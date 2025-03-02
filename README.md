# DPO Fine-Tuned Language Model

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/silanm/nlp-a5)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nlp-a5-dpo-model-fxhwriaztrrhvk9jpwckm4.streamlit.app)


## Overview

This project demonstrates fine-tuning a pre-trained language model (specifically, `gpt2]`) using Direct Preference Optimization (DPO) on the `argilla/distilabel-intel-orca-dpo-pairs` dataset.  

The goal is to improve the model's ability to generate helpful, harmless, and high-quality responses, aligned with human preferences.  


## Dataset

* **Name:** `argilla/distilabel-intel-orca-dpo-pairs`
* **Source:** [`huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs`](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs)
* **Description:** 
The argilla/distilabel-intel-orca-dpo-pairs dataset is a collection of prompts and responses designed for preference learning, specifically Direct Preference Optimization (DPO). It contains pairs of responses to a given prompt, with one "chosen" (preferred) response and one "rejected" (less preferred) response. The dataset is derived from a combination of sources, drawing inspiration from the Orca and UltraChat datasets, and focuses on instruction-following and question-answering tasks with an emphasis on helpfulness and harmlessness. It is created using distilabel.

## Model

*   **Base Model:** `gpt2`
*   **Hugging Face Hub Link:** [`huggingface.co/silanm/nlp-a5`](https://huggingface.co/silanm/nlp-a5)


## Training Setup

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
*   **Sweep Link:** [`nlp-a5-sweep/sweeps`](https://wandb.ai/sila-nmht-asian-institute-of-technology/nlp-a5-sweep?nw=nwusersilanmht)


## Training Results

Based on the `wandb_sweep_export.csv` data,

### Evaluation Metrics

We'll primarily focus on **`eval/loss`** (lower is better) and **`eval/rewards/accuracies`** (higher is better) to identify the best runs.


### Top Performers (sorted by eval/rewards/accuracies)

|Run|eval/loss|eval/rewards/accuracies|learning_rate|beta|gradient_accumulation_steps|per_device_train_batch_size|max_steps|
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|daily-sweep-1|0.6584|0.8219|0.0000538|0.2|4|8|500|
|crimson-sweep-4|0.6629|0.8095|0.0000617|0.2|4|8|500|
|woven-sweep-5|0.6912|0.8110|0.0000675|0.2|4|8|500|
|feasible-sweep-2|2.0949|0.8035|0.000169|0.5|4|8|500| 
|soft-sweep-3|2.0004|0.7989|0.000146|0.5|4|8|500|
|magic-sweep-7|1.2066|0.7782|0.000624|0.1|4|8|500|


### Observations

* **`daily-sweep-1 is the best`**: It has the lowest eval/loss and the highest eval/rewards/accuracies. This is the strongest performing run.

    ![AccuracyAndLoss](images/plot.png) 

* crimson-sweep-4 and woven-sweep-5 are also very good: They have very similar performance to daily-sweep-1, with slightly higher eval/loss but still excellent accuracy.

* magic-sweep-7 is in a lower tier: While its eval/loss is better than many runs, it is not the best, and its accuracy is lower than the top three. The concerns about potential overfitting with its very high learning rate remain.

* feasible-sweep-2 and soft-sweep-3: Good but worse than top 3.

* The best runs all use `beta` = 0.2. Runs with beta = 0.01 consistently performed poorly.

* The best runs have `learning_rate` in the 5e-5 to 7e-5 range. This confirms that this range is a good area to focus on.

* `per_device_train_batch_size` = 8 and `gradient_accumulation_steps` = 4 seems to be a good combination.


## Examples

| Type | Prompt | Response |
|------|--------|----------|
| Direct Question Answering | What is the capital city of Thailand? | The capital city, capital of Thailand, is the country's capital. There are 2.7 million people living in the country, according to the official census. The country has a population of 4.6 million. (...) |
| Instruction Following | Give me three tips for improving my time management skills | 1. Get a good, high-level understanding of what your time management requirements are. : This will help you better understand your time manager's expectations and how they may affect your time. <br />2. Ask for advice before you start your day. <br />3. Ask your time-management company about your current job. |
| Open-Ended/Conversational | Why is it important to protect endangered species? | The problem, as scientists understand it, is that most species live in the wild, and these species will die out before they reach the human population. (...) |
| System Prompt | You are a helpful and honest assistant.<br />What are the steps to make a perfect pancake? | (...) is a full-service pancake maker. It has all the ingredients you need, including: 1/4 cup sugar, 2 eggs, 3/4 tsp baking powder. Pour the batter into a glass or bowl and cover with plastic wrap. (...) |


# Acknowledgements

* Professor Chaklam Silpasuwanchai (Asian Institute of Technology)
* Todsavad Tangtortan (Asian Institute of Technology)
* Argilla/distilabel-intel-orca-dpo-pairs Dataset