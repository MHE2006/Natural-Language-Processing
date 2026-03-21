from models import create_model
from data import tokenized_train, tokenized_dev
from helper import compute_metrics
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")



model = create_model(num_labels=4)

training_args = TrainingArguments(
    output_dir="./results",    
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_dir="./logs",
    logging_strategy="epoch",
    seed=123
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
trainer.save_model("./saved_model/assignment_3_distilbert")

