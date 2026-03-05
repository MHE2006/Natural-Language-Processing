import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models import CNNClassifier, LSTMClassifier
from train import train_model
from helper import evaluate_on_test, plot_learning_curves, get_ag_news_split, print_misclassified, SEED
from data import DataPipeline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Retrieving data splits from assignment 1 
dataset, dev_split = get_ag_news_split(seed=SEED)


pipeline = DataPipeline(max_length=128, batch_size=64)
train_loader, dev_loader, test_loader = pipeline.prepare_all_loaders(dataset, dev_split)
vocab_size = pipeline.vocab_size

test_texts = [str(t) + " " + str(d) for t, d in zip(dataset['test']['title'], dataset['test']['description'])]

print("Training CNN ")
cnn = CNNClassifier(vocab_size=vocab_size,num_classes=4, dropout=0.3)
best_cnn, cnn_history = train_model(cnn, train_loader, dev_loader, epochs=10, device=device)
    
    
plot_learning_curves(cnn_history, "CNN Baseline")
    
    
cnn_preds, test_labels = evaluate_on_test(best_cnn, test_loader, device, "CNN Baseline")
print_misclassified(test_texts, test_labels, cnn_preds, "CNN Baseline", n=10)

print("Training BiLSTM")
lstm = LSTMClassifier(vocab_size=vocab_size, hidden_size=128, dropout=0.3)
best_lstm, lstm_history = train_model(lstm, train_loader, dev_loader, epochs=10, device=device)
    
plot_learning_curves(lstm_history, "BiLSTM")
lstm_preds, _ = evaluate_on_test(best_lstm, test_loader, device, "BiLSTM")
print_misclassified(test_texts, test_labels, lstm_preds, "BiLSTM", n=10)

print("Ablation: CNN with Dropout 0.0")
cnn_no_drop = CNNClassifier(vocab_size=vocab_size, dropout=0.0)
best_cnn_no_drop, cnn_no_drop_history = train_model(cnn_no_drop, train_loader, dev_loader, epochs=10, device=device)
plot_learning_curves(cnn_no_drop_history, "CNN Ablation with Dropout 0.0")

cnn_no_drop_preds, _ = evaluate_on_test(best_cnn_no_drop, test_loader, device, "CNN Ablation with Dropout 0.0")
print_misclassified(test_texts, test_labels, cnn_no_drop_preds, "CNN Ablation with Dropout 0.0", n=10)