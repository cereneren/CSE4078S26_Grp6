"""
train.py — Supervised fine-tuning (LoRA / QLoRA) of Qwen3-4B-Instruct-2507 on the
cleaned, de-duplicated Turkish legal QA SFT set produced by src/preprocess.py.

This is a SEPARATE script from train.py (the team's QLoRA pipeline is left untouched).
Differences in this v2:
  * trains on data/train_sft.jsonl + data/val_sft.jsonl (conversational 'messages'
    format → the model's native chat template), instead of the plain-text prompt;
  * default precision = bf16 LoRA (16-bit base + LoRA); --load_in_4bit switches to QLoRA;
  * LoRA targets attention + MLP projections.

Inputs : data/train_sft.jsonl + data/val_sft.jsonl
Output : models/fine_tuned_v2/final_model  (LoRA adapter + tokenizer)

The official TEST split is NEVER used here (train/val only).

Quick smoke test (validates the whole pipeline in ~1-2 min, no multi-hour wait):
  python -m src.train_v2 --max_steps 5
Full run:
  python -m src.train_v2 --epochs 1
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Attention + MLP projections — adapting the MLP (gate/up/down) too generally gives
# better task adaptation than attention-only.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_sft_config(args) -> SFTConfig:
    """Builds SFTConfig, tolerant to the TRL rename of max_seq_length -> max_length."""
    common = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        bf16=not args.load_in_4bit,
        fp16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    try:
        return SFTConfig(max_seq_length=args.max_seq_len, **common)
    except TypeError:
        return SFTConfig(max_length=args.max_seq_len, **common)


def main():
    p = argparse.ArgumentParser(description="LoRA/QLoRA SFT (v2) for Turkish legal QA.")
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--train_file", default="data/train_sft.jsonl")
    p.add_argument("--val_file", default="data/val_sft.jsonl")
    p.add_argument("--output_dir", default="models/fine_tuned_v2")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--load_in_4bit", action="store_true",
        help="QLoRA: load the base model in 4-bit (use if bf16 OOMs on 12GB).",
    )
    p.add_argument(
        "--max_steps", type=int, default=-1,
        help="Cap total training steps (e.g. 5) for a quick smoke test; -1 = full run.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"Model     : {args.model_name}")
    print(f"Precision : {'4-bit QLoRA' if args.load_in_4bit else 'bf16 LoRA'}")
    print(f"LoRA      : r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"Batch     : {args.batch_size} x grad_accum {args.grad_accum} "
          f"(effective {args.batch_size * args.grad_accum})")
    print(f"Epochs    : {args.epochs}   max_steps: {args.max_steps}   seq_len: {args.max_seq_len}")
    print(f"{'='*60}\n")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model ----
    model_kwargs = dict(trust_remote_code=True, device_map="auto")
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # needed so gradients flow into the frozen base under gradient checkpointing
        model.enable_input_require_grads()

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- data (conversational 'messages'; drop helper columns for a clean collator) ----
    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )
    drop_cols = [c for c in ds["train"].column_names if c != "messages"]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    print(f"Train örnek: {len(ds['train'])}   |   Val örnek: {len(ds['validation'])}\n")

    # ---- train ----
    sft_config = build_sft_config(args)

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_config,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # older TRL used tokenizer= instead of processing_class=
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\nAdaptör kaydedildi: {final_dir}")
    print("Sıradaki adım: evaluate.py ile tam 1500 test setinde before/after koş.")


if __name__ == "__main__":
    main()
