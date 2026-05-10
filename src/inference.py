"""
inference.py — Reusable inference script for Turkish legal QA.

Generation settings (documented here for reproducibility):
    max_new_tokens : int  = 256   — Maximum tokens the model may generate per answer.
    temperature    : float = 0.0  — Greedy decoding (do_sample=False) for deterministic output.
    do_sample      : bool  = False — No stochastic sampling; temperature is effectively unused.
    device         : str          — Resolved automatically via device_map="auto" (GPU if available,
                                    otherwise CPU).  Override with --device cpu|cuda.
    load_in_4bit   : bool  = False — Optional 4-bit quantization via bitsandbytes to reduce VRAM.

Outputs are written to:
    outputs/<model_slug>_inference.jsonl

Each line is a JSON object with:
    {
        "model_name":       <str>,   # HF model identifier
        "question":         <str>,   # Original question from test corpus
        "reference_answer": <str>,   # Gold answer from test corpus
        "generated_answer": <str>    # Model's generated answer
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_prep import load_and_prepare_dataset

# ---------------------------------------------------------------------------
# Generation hyper-parameters (single source of truth)
# ---------------------------------------------------------------------------
GENERATION_CONFIG: dict = {
    "max_new_tokens": 256,
    "do_sample": False,       # greedy decoding — temperature is not applied
    "temperature": None,      # kept for documentation; ignored when do_sample=False
}

SYSTEM_PROMPT = (
    "Sen bir Türk hukuk asistanısın. "
    "Kullanıcının hukuki sorularını doğru ve eksiksiz bir şekilde yanıtla."
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(question: str, context: str = "") -> str:
    """
    Builds the Turkish legal QA prompt for inference (no answer appended).

    Args:
        question: The legal question from the test corpus.
        context:  Optional background context for the question.

    Returns:
        A formatted prompt string ending with 'Cevap:' for the model to complete.
    """
    if context and context.strip():
        return (
            f"Sistem: {SYSTEM_PROMPT}\n\n"
            f"Soru: {question}\n\n"
            f"Bağlam: {context}\n\n"
            f"Cevap:"
        )
    return (
        f"Sistem: {SYSTEM_PROMPT}\n\n"
        f"Soru: {question}\n\n"
        f"Cevap:"
    )


# ---------------------------------------------------------------------------
# Single-sample generation
# ---------------------------------------------------------------------------

def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    context: str = "",
    max_new_tokens: int = GENERATION_CONFIG["max_new_tokens"],
) -> str:
    """
    Generates an answer for a single question using greedy decoding.

    Args:
        model:          A loaded CausalLM model.
        tokenizer:      Corresponding tokenizer.
        question:       The legal question.
        context:        Optional context string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The generated answer text (prompt stripped).
    """
    prompt = build_prompt(question, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=GENERATION_CONFIG["do_sample"],
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    load_in_4bit: bool = False,
    device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a CausalLM model and its tokenizer.

    Args:
        model_name:   HF model identifier (e.g. 'ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1').
        load_in_4bit: Whether to apply 4-bit quantization (requires bitsandbytes + CUDA).
        device:       Explicit device string ('cpu', 'cuda').  None → device_map='auto'.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {}

    if device:
        load_kwargs["device_map"] = device
    else:
        load_kwargs["device_map"] = "auto"

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **load_kwargs,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Per-model inference loop
# ---------------------------------------------------------------------------

def _model_slug(model_name: str) -> str:
    """Converts a HF model ID to a safe filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)


def run_inference(
    model_name: str,
    output_dir: str = "outputs",
    sample_size: int | None = None,
    load_in_4bit: bool = False,
    device: str | None = None,
    max_new_tokens: int = GENERATION_CONFIG["max_new_tokens"],
    dataset_name: str = "Renicames/turkish-law-chatbot",
) -> str:
    """
    Runs inference for one model on the full test corpus and saves results.

    Args:
        model_name:     HF model identifier.
        output_dir:     Directory to write the JSONL output file.
        sample_size:    If set, only the first N test samples are used.
        load_in_4bit:   Enable 4-bit quantization.
        device:         Explicit device override.
        max_new_tokens: Token budget per generation.
        dataset_name:   HF dataset to use.

    Returns:
        Path to the written JSONL output file.
    """
    print(f"\n{'='*60}")
    print(f"Model  : {model_name}")
    print(f"Cihaz  : {device or 'auto'}")
    print(f"4-bit  : {load_in_4bit}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"do_sample     : {GENERATION_CONFIG['do_sample']}  (greedy decoding)")
    print(f"{'='*60}\n")

    dataset = load_and_prepare_dataset(dataset_name)
    test_data = dataset["test"]

    if sample_size and sample_size < len(test_data):
        test_data = test_data.select(range(sample_size))

    print(f"Test corpus: {len(test_data)} örnek")

    model, tokenizer = load_model(model_name, load_in_4bit=load_in_4bit, device=device)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{_model_slug(model_name)}_inference.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(test_data, desc=f"Inference — {model_name}"):
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
                or item.get("Bagam")
                or item.get("bagam")
                or ""
            )
            reference = (
                item.get("output")
                or item.get("answer")
                or item.get("Cevap")
                or item.get("cevap")
                or ""
            )

            generated = generate_answer(model, tokenizer, question, context, max_new_tokens)

            record = {
                "model_name": model_name,
                "question": question,
                "reference_answer": reference,
                "generated_answer": generated,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSonuçlar kaydedildi: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run inference for one or two models on the Turkish legal QA test corpus. "
            "Results are saved as JSONL files under --output_dir."
        )
    )
    parser.add_argument(
        "--model_a",
        type=str,
        required=True,
        help="HF model ID for Model A (e.g. 'ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1')",
    )
    parser.add_argument(
        "--model_b",
        type=str,
        default=None,
        help="HF model ID for Model B (optional; run after Model A)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store JSONL output files (default: outputs/)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit inference to the first N test samples (useful for quick runs)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model(s) in 4-bit quantization to reduce VRAM usage",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force a specific device (default: auto-detect)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=GENERATION_CONFIG["max_new_tokens"],
        help=f"Max tokens to generate per answer (default: {GENERATION_CONFIG['max_new_tokens']})",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Renicames/turkish-law-chatbot",
        help="HF dataset identifier (default: Renicames/turkish-law-chatbot)",
    )

    args = parser.parse_args()

    run_inference(
        model_name=args.model_a,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        load_in_4bit=args.load_in_4bit,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        dataset_name=args.dataset_name,
    )

    if args.model_b:
        run_inference(
            model_name=args.model_b,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            load_in_4bit=args.load_in_4bit,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            dataset_name=args.dataset_name,
        )
