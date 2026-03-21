from models import create_model
from data import tokenized_train, tokenized_dev
from helper import compute_metrics
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")



model = create_model(num_labels=4)

small_train_dataset = tokenized_train.select(range(64))
small_eval_dataset = tokenized_dev.select(range(16))

training_args = TrainingArguments(
    output_dir="./results",    
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    use_cpu=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
trainer.save_model("./saved_model")
