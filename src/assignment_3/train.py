from src.assignment_3.models import create_model
from src.assignment_3.data import tokenized_train, tokenized_dev, tokenizer
from src.helper import compute_metrics
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



model = create_model(num_labels=4).to(device)

training_args = TrainingArguments(
    output_dir="./results", 
    use_cpu = False, 
    fp16=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
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

print(f"Model device: {next(model.parameters()).device}")
trainer.train()
trainer.save_model("./saved_model/assignment_3_distilbert")
tokenizer.save_pretrained("./saved_model/assignment_3_distilbert")

