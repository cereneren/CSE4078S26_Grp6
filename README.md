# CSE4078 Spring 2026 Term Project

This repository contains the codebase for evaluating and fine-tuning small open-source LLMs in the 0.3B to 4B parameter range for the Turkish language, applied to the `Renicames/turkish-law-chatbot` dataset.

## Project Structure

```
.
├── src/
│   ├── data_prep.py   # Dataset loading and prompt formatting
│   ├── inference.py   # Reusable inference script; saves per-sample outputs for both models
│   ├── evaluate.py    # ROUGE evaluation (baseline and fine-tuned comparison)
│   ├── train.py       # Fine-tuning script using QLoRA and SFTTrainer
├── tests/             # Pytest directory for basic structure testing
├── outputs/           # Generated inference outputs (created at runtime)
├── requirements.txt   # Dependencies
└── README.md
```

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Run tests to ensure basic logic is intact.
   ```bash
   pytest tests/
   ```

## Workflow & Usage

### 0. Data Preparation (optional)
Download and cache the dataset splits locally as JSONL files:

```bash
python -m src.data_prep
```

### 1. Inference on Both Models
Run inference for Model A and Model B on the **same** test corpus. Outputs are saved under `outputs/` as JSONL files, one per model, with the following fields per sample: `model_name`, `question`, `reference_answer`, `generated_answer`.

**PowerShell:**
```powershell
python -m src.inference `
    --model_a "MODEL_A_HF_ID" `
    --model_b "MODEL_B_HF_ID" `
    --output_dir outputs `
    --max_new_tokens 256
```

**bash / Linux / macOS:**
```bash
python -m src.inference \
    --model_a "MODEL_A_HF_ID" \
    --model_b "MODEL_B_HF_ID" \
    --output_dir outputs \
    --max_new_tokens 256
```

Generation settings (all documented in `src/inference.py → GENERATION_CONFIG`):

| Setting | Value | Notes |
|---|---|---|
| `max_new_tokens` | 256 | Max tokens generated per answer |
| `do_sample` | False | Greedy decoding — deterministic output |
| `temperature` | N/A | Not applied when `do_sample=False` |
| `device` | auto | GPU if available, otherwise CPU |
| `load_in_4bit` | False | Add `--load_in_4bit` flag to reduce VRAM |

To run on a subset for quick testing (PowerShell):

```powershell
python -m src.inference `
    --model_a "MODEL_A_HF_ID" `
    --model_b "MODEL_B_HF_ID" `
    --sample_size 50 `
    --load_in_4bit
```

### 2. Baseline Evaluation
After inference, compute ROUGE scores against reference answers. `--load_in_4bit` can be used if running on limited VRAM.

```powershell
python -m src.evaluate --model_name "YOUR_SELECTED_MODEL_ID_1"
python -m src.evaluate --model_name "YOUR_SELECTED_MODEL_ID_2"
```

### 3. Fine-Tuning
After selecting the best performing model from the baseline evaluation, train it using Supervised Fine-Tuning (SFTTrainer) + PEFT/QLoRA framework over the `train` dataset constraint split only. *The test set is completely unseen.*

```powershell
python -m src.train `
    --model_name "YOUR_SELECTED_MODEL_ID" `
    --output_dir "models/fine_tuned" `
    --epochs 3 `
    --batch_size 4
```

### 4. Post-Training Evaluation
Test the final adapted model over the exact same test dataset.

```powershell
python -m src.evaluate `
    --model_name "YOUR_SELECTED_MODEL_ID" `
    --adapter_path "models/fine_tuned/final_model" `
    --sample_size 1500
```

## Note on Resources
4-bit quantization reduces memory drastically, pushing 4B parameters well under strict GPU memory limits. The dataset split rule logic (`train` strictly mapping to LoRA fit algorithm evaluation subset isolation) executes in `src/train.py`.
