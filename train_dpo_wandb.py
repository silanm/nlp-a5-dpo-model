# For wandb sweep

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name_or_path = "gpt2"
ignore_bias_buffers = False

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
if ignore_bias_buffers:
    model._ddp_params_and_buffers_to_ignore = [name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool]

model_ref = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def create_prompt(sample):
    return {"prompt": sample["input"]}


train_dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train[:90%]")
eval_dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train[90%:]")

train_dataset = train_dataset.map(create_prompt)
eval_dataset = eval_dataset.map(create_prompt)


wandb.init(project="nlp-a5-sweep")

config = wandb.config

learning_rate = config.learning_rate
per_device_train_batch_size = config.per_device_train_batch_size
gradient_accumulation_steps = config.gradient_accumulation_steps
max_steps = config.max_steps
beta = config.beta
max_length = 512
max_prompt_length = 256
max_target_length = 256
report_to = "wandb"
gradient_checkpointing = True

training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    max_steps=max_steps,
    remove_unused_columns=False,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    evaluation_strategy="steps",
    logging_first_step=True,
    logging_steps=5,
    eval_steps=50,
    output_dir="./nlp-a5",
    optim="adamw_torch",
    warmup_steps=50,
    report_to=report_to,
    bf16=True,
    gradient_checkpointing=gradient_checkpointing,
    save_strategy="steps",
    save_steps=50,
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=beta,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=max_length,
    max_target_length=max_target_length,
    max_prompt_length=max_prompt_length,
    generate_during_eval=True,
)

dpo_trainer.train()

dpo_trainer.save_model(f"./nlp-a5-sweep-beta{dpo_trainer.beta}-lr{dpo_trainer.args.learning_rate}-bs{dpo_trainer.args.per_device_train_batch_size}")

wandb.finish()
