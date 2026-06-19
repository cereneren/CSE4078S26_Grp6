# Fine-Tuning a Small LLM for Turkish Legal Question Answering

**CSE4078 - Spring 2026 Term Project**
**Department of Computer Engineering, Marmara University**

**Group 6:** Ceren Eren, Zorbey Onur Ak, Alp Büyükköse, Kerem Hakkı Koç, Dilan Dilen

> **Academic prototype only:** The generated responses are not professional legal advice and may contain incorrect, incomplete, outdated, or hallucinated information.

---

## Overview

This project investigates whether parameter-efficient supervised fine-tuning can improve the Turkish legal question-answering performance of a small instruction-tuned language model.

Four models with approximately 3-4 billion parameters were screened under the same evaluation protocol. The strongest feasible model, **Qwen3-4B-Instruct-2507**, was then fine-tuned with **LoRA** using a carefully cleaned version of the `Renicames/turkish-law-chatbot` training split.

The final model was evaluated against its own baseline on the official 1,500-example test split using ROUGE, BERTScore, and deterministic error-analysis metrics.

### Main results

| Metric       | Baseline | Fine-tuned | Relative improvement |
| ------------ | -------: | ---------: | -------------------: |
| ROUGE-1      |   0.2013 | **0.5602** |            **+178%** |
| ROUGE-2      |   0.0939 | **0.4542** |            **+384%** |
| ROUGE-L      |   0.1515 | **0.5268** |            **+248%** |
| ROUGE-Lsum   |   0.1630 | **0.5270** |            **+223%** |
| BERTScore F1 |   0.6494 | **0.8294** |             **+28%** |

Fine-tuning also reduced repetitive or degenerate outputs from **13.6% to 0.5%** and increased yes/no polarity accuracy from **81.6% to 95.8%**.

---

## Research Question

> Does LoRA-based supervised fine-tuning improve Turkish legal question answering for a small language model on a shared public benchmark?

The project was designed to:

1. Screen several small instruction-tuned models under the same conditions.
2. Select a model that provides strong baseline quality and remains feasible on a 12 GB GPU.
3. Build a deterministic and parameterized preprocessing pipeline.
4. Fine-tune only the selected model.
5. Compare the baseline and fine-tuned model on the same official test split.
6. Analyze both improvements and remaining legal-reliability risks.

---

## Dataset

The project uses the Hugging Face dataset [Renicames/turkish-law-chatbot](https://huggingface.co/datasets/Renicames/turkish-law-chatbot).

| Split                   | Examples | Purpose                                           |
| ----------------------- | -------: | ------------------------------------------------- |
| Official training split |   13,354 | Source for cleaning, SFT training, and validation |
| Cleaned training subset |   10,709 | LoRA fine-tuning                                  |
| Validation subset       |    1,189 | Validation loss and best-checkpoint selection     |
| Official test split     |    1,500 | Baseline and final evaluation                     |

The official test split was excluded from training, validation, rule-based cleaning, semantic deduplication, embedding generation, and hyperparameter tuning.

A fixed 200-example subset drawn from the official test split was used for baseline model screening. Therefore, the final 1,500-example evaluation is isolated from training but is not completely independent of model selection. This is reported explicitly as a methodological limitation.

The train-validation split uses a fixed random seed of `42`.

---

## Baseline Model Screening

Four instruction-tuned models were evaluated on the same fixed 200-example screening subset using greedy decoding and the same metric implementation.

| Model                      |   ROUGE-1 |   ROUGE-2 |   ROUGE-L | BERTScore F1 |
| -------------------------- | --------: | --------: | --------: | -----------: |
| **Qwen3-4B-Instruct-2507** |     0.230 | **0.108** | **0.183** |    **0.674** |
| Qwen3.5-4B                 | **0.235** |     0.097 |     0.168 |        0.667 |
| Qwen2.5-3B-Instruct        |     0.180 |     0.077 |     0.146 |        0.635 |
| amd/Instella-3B-Instruct   |     0.022 |     0.000 |     0.020 |        0.523 |

### Why Qwen3-4B-Instruct-2507 was selected

* It achieved the strongest BERTScore F1, ROUGE-2, and ROUGE-L scores.
* It was a current-generation dense model with no more than 4 billion parameters.
* It was feasible to fine-tune with LoRA on an NVIDIA RTX 4070 Ti with 12 GB of VRAM.
* Qwen3.5-4B produced a marginally higher ROUGE-1 score, but its MoE/Gated-DeltaNet architecture and software requirements prevented the planned LoRA setup from running reliably within the available hardware constraints.
* Instella produced almost no bigram overlap and was not competitive on this task.

---

## Exploratory Data Analysis

The reference answers are generally short:

| Statistic       | Answer length in words |
| --------------- | ---------------------: |
| Median          |                     20 |
| Mean            |                     24 |
| 90th percentile |                     44 |
| 95th percentile |                     54 |
| 99th percentile |                    101 |
| Maximum         |                    309 |

Based on the token-level analysis:

* `max_new_tokens = 256` was used for generation.
* A 128-token limit could truncate the long tail of the answer distribution.
* A 256-token generation limit covers approximately **99.2%** of the official test references.
* The training maximum sequence length was set to `512` tokens to accommodate the system prompt, question, and answer.

---

## Preprocessing Pipeline

Only the official training split was processed. The official test split remained outside the preprocessing pipeline.

```text
13,354 raw training examples
        |
        v
12,336 after deterministic rule-based cleaning
        |
        v
11,898 after semantic and polarity-aware deduplication
        |
        v
10,709 training / 1,189 validation
```

All cleaning thresholds are exposed as parameters rather than being permanently hard-coded into the experimental design.

### Stage 1 - Deterministic rule-based cleaning

| Step                     | Rule                                          | Removed | Remaining |
| ------------------------ | --------------------------------------------- | ------: | --------: |
| Start                    | Raw official training split                   |       - |    13,354 |
| Length filter            | Remove answers shorter than 6 words           |     829 |    12,525 |
| Exact-pair deduplication | Remove identical question-answer pairs        |       4 |    12,521 |
| Same-question cap        | Keep each normalized question at most 2 times |       2 |    12,519 |
| Same-answer cap          | Keep each normalized answer at most 6 times   |     183 |    12,336 |

Before comparison, every question and answer is normalized using Unicode NFC normalization and surrounding whitespace removal. Deduplication keys are lower-cased and stripped of punctuation and repeated whitespace.

Stage 1 removes **1,018 examples**, corresponding to approximately **7.6%** of the original training split.

### Stage 2 - Semantic deduplication

Each combined `Soru + Cevap` pair is embedded using **Qwen3-Embedding-4B** with:

* bfloat16 precision,
* normalized embeddings,
* batch size `32`.

A union-find clustering procedure groups examples whose cosine similarity is at least `0.955`.

The process produced:

* 9,655 semantic clusters,
* 141 clusters exceeding the configured cap.

### Stage 3 - Polarity-aware capping

Semantic similarity alone can be unsafe for legal data because opposite legal conclusions may be embedded very closely. Examples include:

* `uygundur` / `aykırıdır`
* `yapılabilir` / `yapılamaz`
* `geçerlidir` / `geçersizdir`
* `evet` / `hayır`

To avoid collapsing contradictory examples, every semantic cluster is divided into three rule-based polarity groups:

* `POS`
* `NEG`
* `UNK`

Each polarity group is capped separately at four examples. This prevents positive and negative legal conclusions from being removed as if they were interchangeable duplicates.

| Semantic stage                                 | Examples |
| ---------------------------------------------- | -------: |
| Input after Stage 1                            |   12,336 |
| Removed above the per-cluster/per-polarity cap |      438 |
| Remaining after semantic deduplication         |   11,898 |

The cosine threshold of `0.955` and cap of `4` were selected after manual inspection of removed clusters.

---

## Supervised Fine-Tuning Format

Each cleaned question-answer pair is converted into a three-turn conversation using the selected model's native chat template.

| Role        | Content                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------- |
| `system`    | `Sen bir Türk hukuk asistanısın. Kullanıcının hukuki sorularını doğru ve eksiksiz bir şekilde yanıtla.` |
| `user`      | The legal question                                                                                      |
| `assistant` | The reference answer used as the learning target                                                        |

The final formatted files contain:

* 10,709 training conversations,
* 1,189 validation conversations.

The official test split is not converted into training examples and is never supplied to the trainer.

---

## Fine-Tuning Configuration

The selected model was fine-tuned using supervised fine-tuning with LoRA. The base model was loaded in **16-bit bfloat16**, not 4-bit quantization.

| Parameter                   | Value                                                |
| --------------------------- | ---------------------------------------------------- |
| Base model                  | `Qwen3-4B-Instruct-2507`                             |
| Method                      | Supervised Fine-Tuning with LoRA                     |
| Base-model precision        | 16-bit bfloat16                                      |
| Training epochs             | 4                                                    |
| Selected checkpoint         | Epoch 3                                              |
| Per-device batch size       | 1                                                    |
| Gradient accumulation steps | 16                                                   |
| Effective batch size        | 16                                                   |
| Learning rate               | `2e-4`                                               |
| Scheduler                   | Cosine                                               |
| Warmup ratio                | `0.03`                                               |
| Maximum sequence length     | 512                                                  |
| LoRA rank                   | 16                                                   |
| LoRA alpha                  | 32                                                   |
| LoRA dropout                | 0.05                                                 |
| LoRA target modules         | `q`, `k`, `v`, `o`, `gate`, `up`, `down` projections |
| Optimizer                   | Paged AdamW 8-bit                                    |
| Random seed                 | 42                                                   |
| Hardware                    | NVIDIA RTX 4070 Ti, 12 GB VRAM                       |

### Checkpoint selection

| Epoch | Validation loss |
| ----- | --------------: |
| 1     |           0.470 |
| 2     |           0.396 |
| **3** |       **0.383** |
| 4     |           0.406 |

Epoch 3 produced the lowest validation loss. The increase at epoch 4 was treated as the beginning of overfitting, so the epoch-3 checkpoint was selected for final evaluation.

---

## Evaluation Protocol

The baseline and fine-tuned versions of Qwen3-4B-Instruct-2507 were evaluated on the same official 1,500-example test split.

The evaluation configuration was held constant:

* greedy decoding,
* sampling disabled,
* maximum 256 newly generated tokens,
* identical system prompt and input format,
* ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum,
* BERTScore with `bert-base-multilingual-cased`.

The automatic scores are complemented by deterministic rule-based analyses of:

* answer length,
* legal article-number agreement,
* yes/no polarity,
* repeated 3-gram degeneration.

No LLM-based judge was used for these reported metrics.

---

## Final Results

| Metric              | Baseline | Fine-tuned | Relative change |
| ------------------- | -------: | ---------: | --------------: |
| ROUGE-1             |   0.2013 | **0.5602** |       **+178%** |
| ROUGE-2             |   0.0939 | **0.4542** |       **+384%** |
| ROUGE-L             |   0.1515 | **0.5268** |       **+248%** |
| ROUGE-Lsum          |   0.1630 | **0.5270** |       **+223%** |
| BERTScore Precision |   0.6036 | **0.8275** |        **+37%** |
| BERTScore Recall    |   0.7046 | **0.8328** |        **+18%** |
| BERTScore F1        |   0.6494 | **0.8294** |        **+28%** |

Every reported metric improved after fine-tuning.

The largest relative improvement was observed in ROUGE-2, indicating that the fine-tuned model learned domain-specific multiword expressions and legal phrasing rather than only increasing isolated word overlap.

The smaller relative increase in BERTScore suggests that the baseline model already captured part of the intended meaning, while supervised fine-tuning mainly improved phrasing, conciseness, format, and reference alignment.

---

## Error Analysis

| Deterministic measure        |   Baseline |     Fine-tuned |
| ---------------------------- | ---------: | -------------: |
| Mean answer length           | 89.9 words | **23.5 words** |
| Article-number agreement     |      98.4% |      **98.7%** |
| Yes/no polarity accuracy     |      81.6% |      **95.8%** |
| Repetition/degeneration rate |      13.6% |       **0.5%** |

The main gains came from:

* shorter and more focused answers,
* better yes/no conclusions,
* removal of looping and repetitive degeneration,
* more consistent answer formatting.

Article recognition was already high before fine-tuning and changed only slightly.

---

## Qualitative Improvements

The fine-tuned model frequently improved outputs by:

* removing unnecessary greetings and prompt leakage,
* replacing long and indecisive answers with direct conclusions,
* correcting some baseline polarity errors,
* producing concise definitions that better matched the reference style,
* avoiding repeated phrases and malformed output blocks.

These improvements explain why both lexical-overlap metrics and semantic-similarity metrics increased.

---

## Remaining Risks and Limitations

Despite the large aggregate improvements, the model is not legally reliable.

### 1. Systematic date hallucination introduced by fine-tuning

A targeted analysis of 60 date-related questions found a spurious `2007` pattern:

* The baseline returned the correct year `1982` in 47 cases.
* Fine-tuning changed 20 of those answers to `2007`.
* The baseline did not produce the `2007` error.

This indicates that supervised fine-tuning can overwrite correct base-model knowledge when a misleading pattern exists in the training data.

### 2. Reversed legal conclusions

The model occasionally identifies the relevant legal article but gives the opposite conclusion, such as returning `Evet` when the correct answer is `Hayır`.

This is especially dangerous because the answer may appear confident and legally grounded while its final verdict is incorrect.

### 3. Metric limitations

* ROUGE rewards reference-word overlap but cannot verify legal correctness.
* BERTScore measures semantic similarity but can still reward legally incorrect answers.
* High aggregate polarity accuracy does not eliminate individual high-risk failures.
* No complete professional legal-expert review was performed for all 1,500 outputs.

### 4. Experimental limitations

* One main random seed was used.
* The project did not conduct an exhaustive hyperparameter search.
* The official test split was protected from training and preprocessing, but a 200-example subset was used during baseline model selection.
* Dataset noise and contradictory examples may influence the learned behavior.
* Results are specific to the selected dataset, prompt, models, and evaluation protocol.

---

## Repository Structure

```text
.
├── data/                  # Generated raw, cleaned, and formatted datasets
├── models/                # LoRA adapter, checkpoints, and training metadata
├── outputs/               # Generated answers, metrics, and analysis results
├── src/                   # Preprocessing, training, inference, and evaluation code
├── tests/                 # Automated tests
├── requirements.txt       # Python dependencies
└── README.md
```

Generated datasets, full inference outputs, embedding arrays, intermediate checkpoints, and large model files may be excluded from Git to keep the repository suitable for submission. They can be recreated from the public dataset and the provided scripts/configuration.

The repository should retain the lightweight artifacts required to verify the experiment, including:

* preprocessing and training configuration,
* selected thresholds,
* random seed,
* final aggregate metrics,
* error-analysis summaries,
* LoRA adapter download information when the adapter is stored externally.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/cereneren/CSE4078S26_Grp6.git
cd CSE4078S26_Grp6
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A CUDA-capable NVIDIA GPU is strongly recommended for training. The reported final run used an RTX 4070 Ti with 12 GB of VRAM.

---

## Reproduction Workflow

The full experiment follows this order:

1. Download the public dataset.
2. Save and isolate the official test split.
3. Apply deterministic rule-based cleaning to the official training split.
4. Generate semantic embeddings for cleaned question-answer pairs.
5. Perform cosine-similarity clustering and polarity-aware capping.
6. Split the remaining training data into train and validation subsets with seed `42`.
7. Convert examples to the Qwen chat template.
8. Run baseline inference for the selected model.
9. Fine-tune Qwen3-4B-Instruct-2507 with the reported LoRA configuration.
10. Select the checkpoint with the lowest validation loss.
11. Evaluate the baseline and fine-tuned models on all 1,500 official test examples.
12. Compute ROUGE, BERTScore, and deterministic error-analysis metrics.

Because training and preprocessing scripts may expose their settings through command-line arguments, use the values documented in this README and the saved configuration files when reproducing the final experiment.

---

## Usage (Commands)

```bash
# 0. cache the official dataset splits locally
python -m src.data_prep

# 1. preprocessing  (train split only — the test split is never touched)
python -m src.preprocess_01_regexfilter --min_a_words 6 --max_q_freq 2 --max_a_freq 6   # -> data/train_clean.jsonl  (13,354 -> 12,336)
python -m src.preprocess_02_semanticembed --mode embed                                  # -> data/emb/  (Qwen3-Embedding-4B, bf16)
python -m src.preprocess_03_clustercap --cap 4 --threshold 0.955 --apply                # -> data/train_sft.jsonl, val_sft.jsonl  (12,336 -> 11,898 -> 10,709 / 1,189)

# 2. fine-tune  (LoRA, 16-bit bf16; epoch 3 = lowest validation loss)
python -m src.train --output_dir models/fine_tuned --epochs 4 --max_seq_len 512

# 3. before / after inference on the full 1,500-example test split  (max_new_tokens = 256)
python -m src.inference --model Qwen/Qwen3-4B-Instruct-2507 --max_new_tokens 256 --run_tag a_base_full
python -m src.inference --model Qwen/Qwen3-4B-Instruct-2507 --adapter models/fine_tuned/final_model --max_new_tokens 256 --run_tag finetuned_full

# 4. evaluation + reproducible error analysis
python -m src.evaluate       --input_file outputs/Qwen_Qwen3-4B-Instruct-2507_finetuned_full_inference.jsonl
python -m src.bertscore_eval --input_file outputs/Qwen_Qwen3-4B-Instruct-2507_finetuned_full_inference.jsonl
python -m src.error_analysis \
    --base      outputs/Qwen_Qwen3-4B-Instruct-2507_a_base_full_inference.jsonl \
    --finetuned outputs/Qwen_Qwen3-4B-Instruct-2507_finetuned_full_inference.jsonl
```

`inference.py` defaults to `--max_new_tokens 128`, so pass `256` explicitly to match the reported runs. All thresholds are command-line parameters and the train/validation split uses a fixed seed of `42`.

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT / LoRA
* TRL
* bitsandbytes
* scikit-learn
* ROUGE
* BERTScore
* pytest

---

## Conclusion

Qwen3-4B-Instruct-2507 provided the strongest feasible baseline among the screened small models. LoRA-based supervised fine-tuning improved every reported metric and produced substantially shorter, less repetitive, and more reference-aligned Turkish legal answers.

However, the remaining failure cases show that higher automatic scores do not guarantee legal correctness. The systematic date hallucination and reversed yes/no conclusions demonstrate why the model must be treated as an academic research prototype rather than a deployable legal assistant.

---

## Legal Disclaimer

This repository is intended only for academic and research purposes.

The generated answers do not constitute legal advice. The model may produce incorrect, incomplete, outdated, contradictory, or hallucinated legal information. Consult a qualified legal professional before making legal decisions.
