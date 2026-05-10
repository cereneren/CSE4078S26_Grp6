import argparse
import json
import os

from bert_score import score


def load_predictions(input_file: str):
    predictions = []
    references = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            generated = item.get("generated_answer", "")
            reference = item.get("reference_answer", "")

            predictions.append(generated)
            references.append(reference)

    if not predictions:
        raise ValueError(f"No valid records found in {input_file}")

    return predictions, references


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BERTScore from an inference JSONL file."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to inference JSONL file containing generated_answer and reference_answer.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-base-multilingual-cased",
        help="BERTScore model. Default is multilingual BERT.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where BERTScore metrics will be saved.",
    )

    args = parser.parse_args()

    predictions, references = load_predictions(args.input_file)

    P, R, F1 = score(
        predictions,
        references,
        model_type=args.model_type,
        lang="tr",
        verbose=True,
    )

    results = {
        "input_file": args.input_file,
        "samples": len(predictions),
        "bertscore_model": args.model_type,
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }

    print("\nBERTScore Results")
    print("=================")
    print(f"Input file: {args.input_file}")
    print(f"Samples:   {results['samples']}")
    print(f"Model:     {results['bertscore_model']}")
    print(f"Precision: {results['bertscore_precision']:.4f}")
    print(f"Recall:    {results['bertscore_recall']:.4f}")
    print(f"F1:        {results['bertscore_f1']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    input_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = os.path.join(
        args.output_dir,
        f"{input_name}_bertscore_metrics.json",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nBERTScore metrics saved to: {output_path}")


if __name__ == "__main__":
    main()