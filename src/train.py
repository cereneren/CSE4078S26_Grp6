import argparse
import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer

from src.data_prep import load_and_prepare_dataset, apply_prompt_template


def train(
    model_name: str,
    output_dir: str,
    num_train_epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    validation_ratio: float = 0.1,
    max_seq_length: int = 512,
    seed: int = 42,
):
    """
    Fine-tunes a base causal language model using QLoRA.

    Important methodology note:
    The official test split is NOT used during training or validation.
    A validation set is created only from the training split.
    The test split should be used only after training, inside evaluate.py.
    """

    print("\nLoading dataset...")
    dataset = load_and_prepare_dataset()
    dataset = apply_prompt_template(dataset)

    full_train_data = dataset["train"]

    print("\nCreating train/validation split from the training set only...")
    split_dataset = full_train_data.train_test_split(
        test_size=validation_ratio,
        seed=seed,
        shuffle=True,
    )

    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(eval_data)}")
    print("Official test split is kept untouched for final evaluation.")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("\nConfiguring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print("\nConfiguring LoRA adapter...")
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

    print("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        do_eval=True,
        optim="paged_adamw_8bit",
        fp16=True,
        report_to="none",
        seed=seed,
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()

    final_model_path = os.path.join(output_dir, "final_model")

    print(f"\nSaving final LoRA adapter to: {final_model_path}")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print("\nTraining complete.")
    print("Use evaluate.py on the official test split for final results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a Turkish legal QA model using QLoRA."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model ID to fine-tune",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/fine_tuned",
        help="Directory where checkpoints and final adapter will be saved",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="Ratio of the training set used as validation data",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for SFT training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        validation_ratio=args.validation_ratio,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )