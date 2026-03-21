import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer
from data import tokenized_test, test_set
from helper import compute_metrics, print_misclassified, labels_map




### Assignment 2 Best Model ###



### Assignment 3: DistilBERT ###
model = DistilBertForSequenceClassification.from_pretrained("./saved_model/assignment_3_distilbert")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate(eval_dataset=tokenized_test)
print("Test set results:", results)

