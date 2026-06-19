"""
inference.py — Inference for modern instruct / multimodal LLMs (Gemma 3, Qwen3.5, …).

  • This script:
      - loads via AutoModelForCausalLM, falling back to AutoModelForImageTextToText for
        multimodal checkpoints (Gemma 3, Qwen3.5);
      - always formats inputs with the model's native chat template (folding the system
        prompt into the user turn for models like Gemma that reject a 'system' role);
      - writes the SAME JSONL schema as inference.py, so evaluate.py and
        bertscore_eval.py work on its output unchanged.

Requirements:
  • Gemma 3 → transformers >= 4.51 and `huggingface-cli login` (gated model).
  • Qwen3.5 → transformers v5 (older versions raise 'Unrecognized model'); on Windows the
    custom Mamba/Triton kernels may fail — that is the known wall, not a bug in this script.

NOTE ON COMPARABILITY: models run here use their chat template, whereas the three original
baselines (inference.py) used a plain completion prompt. Numbers from this script are
therefore NOT strictly apples-to-apples with those baselines — footnote it, or re-run all
models the same way if a fully fair table is needed.

Usage:
  python -m src.inference --model "google/gemma-3-4b-it" --sample_size 1500 \
      --max_new_tokens 128 --load_in_4bit

Output: outputs/<model_slug>_inference.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
from tqdm import tqdm

from src.data_prep import load_and_prepare_dataset

SYSTEM_PROMPT = (
    "Sen bir Türk hukuk asistanısın. "
    "Kullanıcının hukuki sorularını doğru ve eksiksiz bir şekilde yanıtla."
)


def _model_slug(model_name: str) -> str:
    """Converts a HF model ID to a safe filename component (matches inference.py)."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)


def _extract_fields(item: dict) -> tuple[str, str, str]:
    """Pulls question / context / reference from a dataset row (multiple field names)."""
    question = (
        item.get("instruction")
        or item.get("question")
        or item.get("Soru")
        or item.get("soru")
        or ""
    )
    context = (
        item.get("input")
        or item.get("context")
        or item.get("Bağlam")
        or item.get("Baglam")
        or item.get("bağlam")
        or item.get("baglam")
        or ""
    )
    reference = (
        item.get("output")
        or item.get("answer")
        or item.get("Cevap")
        or item.get("cevap")
        or ""
    )
    return question, context, reference


def load_model(model_name: str, load_in_4bit: bool = False, device: str | None = None,
               adapter: str | None = None):
    """
    Loads model + (processor or tokenizer).

    Tries AutoModelForCausalLM first and falls back to AutoModelForImageTextToText for
    multimodal checkpoints (Gemma 3, Qwen3.5).
    """
    from transformers import AutoTokenizer

    processor = None
    tokenizer = None

    # Multimodal models expose an AutoProcessor (which wraps a tokenizer); text-only
    # models only have a tokenizer.
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        processor = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"device_map": device or "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_kwargs["torch_dtype"] = "auto"

    from transformers import AutoModelForCausalLM

    model = None
    errors = []
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, **load_kwargs
        )
    except Exception as e:
        errors.append(f"AutoModelForCausalLM -> {e}")
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_name, trust_remote_code=True, **load_kwargs
            )
        except Exception as e2:
            errors.append(f"AutoModelForImageTextToText -> {e2}")
            raise RuntimeError(
                f"Could not load '{model_name}'.\n" + "\n".join(errors)
            )

    if adapter:
        from peft import PeftModel

        print(f"LoRA adaptörü yükleniyor: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tokenizer, processor


def build_inputs(tokenizer, processor, question: str, context: str, device,
                 no_think: bool = False):
    """
    Builds chat-template inputs. Tries a system+user message first; if the model's
    template rejects a 'system' role (e.g. Gemma), folds the system prompt into the
    user turn.

    When no_think=True, passes enable_thinking=False to the chat template. Qwen3 /
    Qwen3.5 emit a reasoning chain by default; for short-answer QA we want the direct
    answer, not the (often English) chain-of-thought.
    """
    if context and context.strip():
        user_content = f"Soru: {question}\n\nBağlam: {context}"
    else:
        user_content = f"Soru: {question}"

    tok = processor if processor is not None else tokenizer

    template_kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    if no_think:
        template_kwargs["enable_thinking"] = False

    message_variants = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"},
        ],
    ]

    last_err = None
    for messages in message_variants:
        try:
            enc = tok.apply_chat_template(messages, **template_kwargs)
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
        except Exception as e:
            last_err = e
    raise last_err


def run_inference(
    model_name: str,
    output_dir: str = "outputs",
    sample_size: int | None = None,
    load_in_4bit: bool = False,
    device: str | None = None,
    max_new_tokens: int = 128,
    dataset_name: str = "Renicames/turkish-law-chatbot",
    no_think: bool = False,
    adapter: str | None = None,
    run_tag: str | None = None,
) -> str:
    print(f"\n{'='*60}")
    print(f"Model         : {model_name}")
    print(f"4-bit         : {load_in_4bit}")
    print(f"max_new_tokens: {max_new_tokens}  (greedy, chat template)")
    print(f"no_think      : {no_think}")
    print(f"adapter       : {adapter or '-'}")
    print(f"{'='*60}\n")

    dataset = load_and_prepare_dataset(dataset_name)
    test_data = dataset["test"]
    if sample_size and sample_size < len(test_data):
        test_data = test_data.select(range(sample_size))
    print(f"Test corpus: {len(test_data)} örnek")

    model, tokenizer, processor = load_model(model_name, load_in_4bit, device, adapter=adapter)

    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{run_tag}" if run_tag else ""
    output_path = os.path.join(output_dir, f"{_model_slug(model_name)}{tag}_inference.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(test_data, desc=f"Inference — {model_name}"):
            question, context, reference = _extract_fields(item)
            enc = build_inputs(
                tokenizer, processor, question, context, model.device, no_think=no_think
            )
            input_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_len:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            record = {
                "model_name": model_name,
                "question": question,
                "reference_answer": reference,
                "generated_answer": generated,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSonuçlar kaydedildi: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Inference for modern instruct/multimodal LLMs (Gemma 3, Qwen3.5) on the "
            "Turkish legal QA test corpus. Output is JSONL compatible with evaluate.py "
            "and bertscore_eval.py."
        )
    )
    parser.add_argument("--model", type=str, required=True, help="HF model ID")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="Limit inference to the first N test samples",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--load_in_4bit", action="store_true",
        help="Load in 4-bit (bitsandbytes). Fine for Gemma 3; for Qwen3.5 prefer fp16.",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--dataset_name", type=str, default="Renicames/turkish-law-chatbot",
    )
    parser.add_argument(
        "--no_think", action="store_true",
        help="Disable thinking/reasoning chains (enable_thinking=False) for Qwen3/Qwen3.5 "
             "so the model returns the direct answer instead of a chain-of-thought.",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Path to a LoRA adapter to load on top of the base model (the fine-tuned "
             "model, e.g. models/fine_tuned/final_model).",
    )
    parser.add_argument(
        "--run_tag", type=str, default=None,
        help="Suffix added to the output filename (e.g. base_full / finetuned_full) so "
             "before/after runs don't overwrite each other.",
    )
    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        load_in_4bit=args.load_in_4bit,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        dataset_name=args.dataset_name,
        no_think=args.no_think,
        adapter=args.adapter,
        run_tag=args.run_tag,
    )
