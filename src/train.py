import argparse
import json
import os
import platform
from typing import Any

import peft
import torch
import transformers
import trl
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from src.data_prep import apply_prompt_template, load_and_prepare_dataset


DATASET_NAME = "Renicames/turkish-law-chatbot"


def deduplicate_training_data(dataset: Dataset) -> Dataset:
    """
    Removes empty examples and exact duplicate formatted training examples.

    Deduplication is performed before the train/validation split so that the
    same exact example cannot appear in both subsets.
    """
    dataframe = dataset.to_pandas()

    dataframe["text"] = dataframe["text"].fillna("").astype(str).str.strip()
    dataframe = dataframe[dataframe["text"].str.len() > 0]

    before_count = len(dataframe)

    dataframe = (
        dataframe.drop_duplicates(subset=["text"], keep="first")
        .reset_index(drop=True)
    )

    after_count = len(dataframe)

    print(f"Samples before exact deduplication: {before_count}")
    print(f"Samples after exact deduplication:  {after_count}")
    print(f"Exact duplicates removed:           {before_count - after_count}")

    return Dataset.from_pandas(dataframe, preserve_index=False)


def get_cuda_configuration() -> tuple[torch.dtype, bool, bool]:
    """
    Determines the most suitable training precision for the current GPU.

    Returns:
        compute_dtype: Datatype used for 4-bit calculations.
        use_fp16: Whether FP16 training should be enabled.
        use_bf16: Whether BF16 training should be enabled.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. QLoRA training with bitsandbytes requires "
            "a CUDA-capable NVIDIA GPU."
        )

    bf16_supported = torch.cuda.is_bf16_supported()

    if bf16_supported:
        compute_dtype = torch.bfloat16
        use_bf16 = True
        use_fp16 = False
    else:
        compute_dtype = torch.float16
        use_bf16 = False
        use_fp16 = True

    print("\nCUDA configuration:")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )
    print(f"BF16 supported: {bf16_supported}")
    print(f"QLoRA compute dtype: {compute_dtype}")

    return compute_dtype, use_fp16, use_bf16


def save_training_configuration(
    output_dir: str,
    configuration: dict[str, Any],
) -> None:
    """
    Saves the experimental setup for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    configuration_path = os.path.join(
        output_dir,
        "training_configuration.json",
    )

    with open(configuration_path, "w", encoding="utf-8") as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Training configuration saved to: {configuration_path}")


def train(
    model_name: str,
    output_dir: str,
    num_train_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lr: float = 2e-4,
    validation_ratio: float = 0.1,
    max_seq_length: int = 512,
    seed: int = 42,
) -> None:
    """
    Fine-tunes a causal language model using QLoRA.

    Methodology:
    - Only the official training split is used for model adaptation.
    - Exact duplicates are removed before splitting.
    - Validation data is created only from the training split.
    - The official test split is not formatted, trained on, or validated on.
    - The official test split must be used only after training.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be greater than 0 and smaller than 1."
        )

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be at least 1."
        )

    if max_seq_length < 1:
        raise ValueError("max_seq_length must be at least 1.")

    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    compute_dtype, use_fp16, use_bf16 = get_cuda_configuration()

    # ---------------------------------------------------------------
    # Dataset preparation
    # ---------------------------------------------------------------
    print("\nLoading dataset...")

    raw_dataset = load_and_prepare_dataset(DATASET_NAME)

    if "train" not in raw_dataset:
        raise KeyError("The dataset does not contain a 'train' split.")

    if "test" not in raw_dataset:
        raise KeyError("The dataset does not contain a 'test' split.")

    official_test_size = len(raw_dataset["test"])

    print(f"Official training split: {len(raw_dataset['train'])} samples")
    print(f"Official test split:     {official_test_size} samples")
    print("The official test split will not be used during training.")

    # Format only the training split. The official test split is deliberately
    # excluded from the training preprocessing pipeline.
    train_only_dataset = DatasetDict(
        {
            "train": raw_dataset["train"],
        }
    )

    formatted_dataset = apply_prompt_template(train_only_dataset)
    full_train_data = formatted_dataset["train"]

    print("\nApplying preprocessing and exact deduplication...")
    full_train_data = deduplicate_training_data(full_train_data)

    print("\nCreating train/validation split from training data only...")

    split_dataset = full_train_data.train_test_split(
        test_size=validation_ratio,
        seed=seed,
        shuffle=True,
    )

    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]

    print(f"Final training samples:   {len(train_data)}")
    print(f"Final validation samples: {len(eval_data)}")
    print(f"Official test samples:    {official_test_size}")
    print("Official test overlap with training: 0 by construction")

    # ---------------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------------
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_length

    # ---------------------------------------------------------------
    # QLoRA quantization
    # ---------------------------------------------------------------
    print("\nConfiguring 4-bit NF4 quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("\nLoading the base model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # ---------------------------------------------------------------
    # LoRA configuration
    # ---------------------------------------------------------------
    print("\nConfiguring the LoRA adapter...")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    effective_batch_size = (
        batch_size * gradient_accumulation_steps
    )

    # ---------------------------------------------------------------
    # SFT configuration
    # ---------------------------------------------------------------
    print("\nConfiguring supervised fine-tuning...")

    training_args = SFTConfig(
        output_dir=output_dir,

        # Dataset configuration
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,

        # Batch configuration
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # Optimization
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        lr_scheduler_type="linear",
        max_grad_norm=1.0,

        # Precision and memory
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        # Logging, evaluation, and checkpoints
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        # Best checkpoint selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Reproducibility
        seed=seed,
        data_seed=seed,
        report_to="none",
    )

    # ---------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------
    print("\nInitializing SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    if hasattr(trainer.model, "print_trainable_parameters"):
        print("\nTrainable parameter information:")
        trainer.model.print_trainable_parameters()

    experiment_configuration = {
        "project": "CSE4078 Spring 2026 Term Project",
        "dataset": DATASET_NAME,
        "model_name": model_name,
        "method": "Supervised Fine-Tuning with QLoRA",
        "official_train_split_size": len(raw_dataset["train"]),
        "official_test_split_size": official_test_size,
        "post_deduplication_size": len(full_train_data),
        "training_samples": len(train_data),
        "validation_samples": len(eval_data),
        "validation_ratio": validation_ratio,
        "test_used_during_training": False,
        "num_train_epochs": num_train_epochs,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size_per_process": effective_batch_size,
        "learning_rate": lr,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "quantization": {
            "bits": 4,
            "type": "NF4",
            "double_quantization": True,
            "compute_dtype": str(compute_dtype),
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            ),
        },
    }

    save_training_configuration(
        output_dir,
        experiment_configuration,
    )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    print("\nStarting training...")
    print(f"Per-device batch size: {batch_size}")
    print(
        "Gradient accumulation steps: "
        f"{gradient_accumulation_steps}"
    )
    print(
        "Effective batch size per process: "
        f"{effective_batch_size}"
    )

    train_result = trainer.train()

    trainer.log_metrics(
        "train",
        train_result.metrics,
    )
    trainer.save_metrics(
        "train",
        train_result.metrics,
    )
    trainer.save_state()

    # Perform and store one final validation evaluation.
    print("\nRunning final validation evaluation...")

    validation_metrics = trainer.evaluate()

    trainer.log_metrics(
        "validation",
        validation_metrics,
    )
    trainer.save_metrics(
        "validation",
        validation_metrics,
    )

    # ---------------------------------------------------------------
    # Save final adapter
    # ---------------------------------------------------------------
    final_model_path = os.path.join(
        output_dir,
        "final_model",
    )

    print(
        "\nSaving the best final LoRA adapter to: "
        f"{final_model_path}"
    )

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print("\nTraining complete.")
    print(f"Final adapter: {final_model_path}")
    print(
        "The official test split must now be evaluated using evaluate.py."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a Turkish legal question-answering model "
            "using QLoRA."
        )
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model identifier to fine-tune.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/fine_tuned",
        help="Directory for checkpoints, metrics, and the final adapter.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training and validation batch size.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps accumulated before an optimizer update.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="Fraction of training data reserved for validation.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum tokenized training sequence length.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for splitting and training.",
    )

    arguments = parser.parse_args()

    train(
        model_name=arguments.model_name,
        output_dir=arguments.output_dir,
        num_train_epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=(
            arguments.gradient_accumulation_steps
        ),
        lr=arguments.lr,
        validation_ratio=arguments.validation_ratio,
        max_seq_length=arguments.max_seq_length,
        seed=arguments.seed,
    )
