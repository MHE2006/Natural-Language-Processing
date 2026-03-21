import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from datasets import load_dataset

import os
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


SEED = 123

## Helper functions for getting AG NEWS split

def get_ag_news_split(seed=SEED):
    dataset = load_dataset("sh0416/ag_news")
    dev_split = dataset['train'].train_test_split(test_size=0.1, seed=SEED)
    return dataset, dev_split


# Helper function for preprocessing/normalizing text
def preprocess_text(text, lemmatizer, stop_words):
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.lower().split()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

# Helper function to print metrics and return confusion matrix
def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"--- {name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}\n")
    return confusion_matrix(y_true, y_pred)

# Helper function to print misclassified examples
labels_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
def print_misclassified(texts, y_true, y_pred, model_name, n=20):
    print(f" FIRST {n} MISCLASSIFIED: {model_name}")
    count = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            count += 1
            print(f"{count}. [Index {i}]")
            print(f"   Text: {texts[i][:150]}...") 
            print(f"   ACTUAL: {labels_map[y_true[i]]}")
            print(f"   PREDICTED: {labels_map[y_pred[i]]}\n")
        if count >= n:
            break



def evaluate_on_test(model, test_loader, device, model_name="Model"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_inputs, attention_mask, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_inputs, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    mac_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"{model_name} Test Results")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {mac_f1:.4f}\n")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{PLOTS_DIR}/{filename}_confusion_matrix.png", bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {PLOTS_DIR}/{filename}_confusion_matrix.png")

    return all_preds, all_labels

def plot_learning_curves(history, model_name):
    """Plots training loss and dev Macro-F1 score."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(epochs, history['train_loss'], color='tab:red', label='Train Loss', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Dev Macro-F1', color='tab:blue')
    ax2.plot(epochs, history['dev_f1'], color='tab:blue', label='Dev Macro-F1', marker='s')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title(f'{model_name} Learning Curves')
    fig.tight_layout()
    filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{PLOTS_DIR}/{filename}_learning_curves.png", bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {PLOTS_DIR}/{filename}_learning_curves.png")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}
