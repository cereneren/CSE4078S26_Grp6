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
    Evaluates a base or fine-tuned model on the test split using ROUGE.

    The test split should only be used here, after model selection/training.
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

    print("\n===============================")
    print(f"Evaluation Results")
    print(f"Model: {model_name}")
    print(f"Adapter: {adapter_path if adapter_path else 'Base Model'}")
    print(f"Samples: {len(test_data)}")
    print("===============================")

    for key, value in results.items():
        print(f"{key}: {value:.4f}")

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

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated answers saved to: {predictions_path}")
    print(f"ROUGE metrics saved to: {metrics_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline or fine-tuned Turkish legal QA model."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base Hugging Face model ID",
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights, if evaluating a fine-tuned model",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit number of test samples evaluated",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit to reduce VRAM usage",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens generated per answer",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for evaluation outputs and metrics",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Renicames/turkish-law-chatbot",
        help="Hugging Face dataset ID",
    )

    args = parser.parse_args()

    run_evaluation(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        sample_size=args.sample_size,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )