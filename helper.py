import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def preprocess_text(text, lemmatizer, stop_words):
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.lower().split()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"--- {name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}\n")
    return confusion_matrix(y_true, y_pred)

labels_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

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
