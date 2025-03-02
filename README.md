# DPO Fine-Tuned Language Model

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/silanm/nlp-a5)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL)


## Overview

This project demonstrates fine-tuning a pre-trained language model (specifically, `gpt2]`) using Direct Preference Optimization (DPO) on the `argilla/distilabel-intel-orca-dpo-pairs` dataset.  

The goal is to improve the model's ability to generate helpful, harmless, and high-quality responses, aligned with human preferences.  

The project includes:

* **Training Script**  
A Python script (`train_dpo_wandb.py`) that performs the DPO fine-tuning using the `trl` library and `transformers`.

* **Weights & Biases Integration:**
Full integration with Weights & Biases (wandb) for experiment tracking, hyperparameter tuning (using sweeps), and model logging.

* **Streamlit Interface:**
A simple web application (`app.py`) built with Streamlit that allows users to interact with the fine-tuned model.

* **Hugging Face Hub Integration:**
Code to push the trained model and tokenizer to the Hugging Face Hub.


## Dataset

* **Name:** `argilla/distilabel-intel-orca-dpo-pairs`
* **Source:** [`huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs`]([https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs)
* **Description:** 


## Model

*   **Base Model:** `gpt2`
*   **Hugging Face Hub Link:** [`huggingface.co/silanm/nlp-a5`](https://huggingface.co/silanm/nlp-a5)


## Training

| Hyperparameter | Value |
|----------------|-------|
| **learning_rate** | log(1e-6) to log(1e-3) |
| **per_device_train_batch_size**  | 2, 4, 8 |
| **gradient_accumulation_steps**  | 1, 2, 4 |
| **max_steps** | 100, 200, 500 |
| **beta** | 0.01, 0.1, 0.2, 0.5 |
| **max_length** | 512 |
| **max_prompt_length** | 256 |
| **max_target_length** | 256 |
| **optimizer** | adamw_torch |
| **warmup_steps** | 50 |
| **gradient_checkpointing** | True |
| **bf16** | True |
| **logging_steps** | 5 |
| **eval_steps** | 50 |

### Weights & Biases Sweeps

*   **Sweep Configuration:** `sweep_config.yaml`
*   **Sweep Link:** [`nlp-a5-sweep/sweeps`](https://wandb.ai/sila-nmht-asian-institute-of-technology/nlp-a5-sweep/sweeps/r6rta83w)
*   **Results:** 


## Results
...

**Evaluation Metrics**: 

**Qualitative Examples**: 

**Discussion**: 


# Acknowledgements

* Professor Chaklam Silpasuwanchai (Asian Institute of Technology)
* Todsavad Tangtortan (Asian Institute of Technology)
* Argilla/distilabel-intel-orca-dpo-pairs Dataset