import argparse
import json
import os
from typing import Optional

import torch
import evaluate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_prep import load_and_prepare_dataset
from src.inference import generate_answer, _model_slug


def extract_fields(item: dict) -> tuple[str, str, str]:
    """
    Extracts question, context, and reference answer from a dataset row.
    Supports multiple possible field names.
    """
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

    return question, context, reference


def load_model_for_evaluation(
    model_name: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
):
    """
    Loads a base model, tokenizer, and optionally a LoRA adapter.
    This mode is slower because it generates answers before calculating ROUGE.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"}

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def save_metrics(results: dict, metrics_path: str):
    """
    Saves ROUGE metrics as JSON.
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_results(title: str, results: dict, sample_count: int, extra_info: Optional[str] = None):
    """
    Prints ROUGE results in a readable format.
    """
    print("\n===============================")
    print(title)

    if extra_info:
        print(extra_info)

    print(f"Samples: {sample_count}")
    print("===============================")

    for key, value in results.items():
        print(f"{key}: {value:.4f}")


def evaluate_from_file(
    input_file: str,
    output_dir: str = "outputs",
    metrics_name: Optional[str] = None,
):
    """
    Fast evaluation mode.

    Reads an existing inference JSONL file and calculates ROUGE using:
    - generated_answer
    - reference_answer

    This avoids loading the model and generating answers again.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {input_file}"
                ) from e

            generated_answer = item.get("generated_answer", "")
            reference_answer = item.get("reference_answer", "")

            predictions.append(generated_answer)
            references.append(reference_answer)

    if not predictions:
        raise ValueError(f"No valid prediction records found in {input_file}")

    results = rouge.compute(predictions=predictions, references=references)

    print_results(
        title="Evaluation Results from Saved Inference File",
        results=results,
        sample_count=len(predictions),
        extra_info=f"Input file: {input_file}",
    )

    os.makedirs(output_dir, exist_ok=True)

    if metrics_name is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        metrics_name = f"{base_name}_rouge_metrics.json"

    metrics_path = os.path.join(output_dir, metrics_name)
    save_metrics(results, metrics_path)

    print(f"\nROUGE metrics saved to: {metrics_path}")

    return results


def run_evaluation(
    model_name: str,
    adapter_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    load_in_4bit: bool = False,
    max_new_tokens: int = 256,
    output_dir: str = "outputs",
    dataset_name: str = "Renicames/turkish-law-chatbot",
):
    """
    Slow evaluation mode.

    Loads the model, generates answers on the test split, then calculates ROUGE.

    Use this only if you do not already have an inference JSONL file.
    If you already ran inference.py, use --input_file instead.
    """
    dataset = load_and_prepare_dataset(dataset_name)
    test_data = dataset["test"]

    if sample_size and sample_size < len(test_data):
        test_data = test_data.select(range(sample_size))

    model, tokenizer = load_model_for_evaluation(
        model_name=model_name,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
    )

    rouge = evaluate.load("rouge")

    predictions = []
    references = []
    records = []

    for item in tqdm(test_data, desc=f"Evaluating {model_name}"):
        question, context, reference = extract_fields(item)

        prediction = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            context=context,
            max_new_tokens=max_new_tokens,
        )

        predictions.append(prediction)
        references.append(reference)

        records.append(
            {
                "model_name": model_name,
                "adapter_path": adapter_path,
                "question": question,
                "context": context,
                "reference_answer": reference,
                "generated_answer": prediction,
            }
        )

    results = rouge.compute(predictions=predictions, references=references)

    print_results(
        title="Evaluation Results",
        results=results,
        sample_count=len(test_data),
        extra_info=(
            f"Model: {model_name}\n"
            f"Adapter: {adapter_path if adapter_path else 'Base Model'}"
        ),
    )

    os.makedirs(output_dir, exist_ok=True)

    model_slug = _model_slug(model_name)
    adapter_suffix = "_finetuned" if adapter_path else "_base"

    predictions_path = os.path.join(
        output_dir,
        f"{model_slug}{adapter_suffix}_evaluation_outputs.jsonl",
    )

    metrics_path = os.path.join(
        output_dir,
        f"{model_slug}{adapter_suffix}_rouge_metrics.json",
    )

    with open(predictions_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    save_metrics(results, metrics_path)

    print(f"\nGenerated answers saved to: {predictions_path}")
    print(f"ROUGE metrics saved to: {metrics_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline or fine-tuned Turkish legal QA model."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help=(
            "Fast mode: evaluate an existing inference JSONL file instead of "
            "loading the model and generating answers again."
        ),
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Base Hugging Face model ID. Required unless --input_file is used.",
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights, if evaluating a fine-tuned model.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit number of test samples evaluated in slow generation mode.",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit to reduce VRAM usage in slow generation mode.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens generated per answer in slow generation mode.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for evaluation outputs and metrics.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Renicames/turkish-law-chatbot",
        help="Hugging Face dataset ID.",
    )

    parser.add_argument(
        "--metrics_name",
        type=str,
        default=None,
        help="Optional output filename for ROUGE metrics when using --input_file.",
    )

    args = parser.parse_args()

    if args.input_file:
        evaluate_from_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            metrics_name=args.metrics_name,
        )
    else:
        if not args.model_name:
            raise ValueError("You must provide --model_name unless using --input_file.")

        run_evaluation(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            sample_size=args.sample_size,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
        )