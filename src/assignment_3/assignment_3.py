import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer
from src.assignment_3.data import tokenized_test
from src.helper import compute_metrics, evaluate_on_test, print_metrics, print_misclassified_bert, get_ag_news_split, SEED
from src.assignment_2.models import LSTMClassifier
from src.assignment_2.data import DataPipeline,dataset,dev_split
from src.assignment_2.train import train_model
from src.assignment_3.slice_ev import length_bucket_evaluation, keyword_masking_evalutation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pipeline = DataPipeline(max_length=128, batch_size=64)
train_loader, dev_loader, test_loader = pipeline.prepare_all_loaders(dataset, dev_split)
vocab_size = pipeline.vocab_size

### Assignment 2 Best Model ###

lstm = LSTMClassifier(vocab_size=vocab_size, hidden_size=128, dropout=0.3)
best_lstm, lstm_history = train_model(lstm, train_loader, dev_loader, epochs=10, device=device)
lstm_preds, lstm_labels = evaluate_on_test(best_lstm, test_loader, device, "BiLSTM")

confusion_matrix = print_metrics("BILSTM", lstm_labels, lstm_preds)
print(confusion_matrix)

### Assignment 3: DistilBERT ###

hugging_face_model = "Merijn2006/Assignment_3_NLP"

model = DistilBertForSequenceClassification.from_pretrained(hugging_face_model)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate(eval_dataset=tokenized_test)

preds_output = trainer.predict(tokenized_test)
bert_preds = np.argmax(preds_output.predictions, axis=-1)
bert_labels = preds_output.label_ids

cm = print_metrics("DistilBERT", bert_labels, bert_preds)
print(cm)



print("Slice Evaluation Length Buckets")
length_bucket_evaluation(trainer, tokenized_test)


print("Original accuracy: ")
results = trainer.evaluate(eval_dataset=tokenized_test)


print("Keyword Masking evaluation")
keywords = ["trainer","coach","ball", "stock", "company", "computer", "electricity", "leader", "politics"]
keyword_masking_evalutation(trainer,tokenized_test, tokenizer,keywords)

dataset, dev_split = get_ag_news_split(seed=SEED)
test_texts = [str(t) + " " + str(d) for t, d in zip(dataset['test']['title'], dataset['test']['description'])]
print("Missclassified")
print_misclassified_bert(test_texts, bert_labels, bert_preds, "Bert", 10 )
