from datasets import load_dataset
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from assignment_2.helper import preprocess_text, print_metrics, print_misclassified

SEED = 123


dataset = load_dataset("sh0416/ag_news")
dev_split = dataset['train'].train_test_split(test_size=0.1, seed=SEED)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


train_texts = [preprocess_text(x,lemmatizer,stop_words) for x in dev_split['train']['description']]
dev_texts = [preprocess_text(x, lemmatizer, stop_words) for x in dev_split['test']['description']]
test_texts = [preprocess_text(x, lemmatizer, stop_words) for x in dataset['test']['description']]

### Vectorization using TF-IDF ###
tfidf = TfidfVectorizer() 
vectorized_train = tfidf.fit_transform(train_texts)
vectorized_dev = tfidf.transform(dev_texts)
vectorized_test = tfidf.transform(test_texts)

train_labels = dev_split['train']['label']
dev_labels = dev_split['test']['label']
test_labels = dataset['test']['label']

### Logistic Regression ###

log_reg = LogisticRegression(random_state=SEED)
log_reg.fit(vectorized_train, train_labels)


### Linear SVM ### 
lin_svm = LinearSVC(random_state=SEED)
lin_svm.fit(vectorized_train,train_labels)


### Predictions of Baseline Models on both dev and test set ###

log_reg_test_prediction = log_reg.predict(vectorized_test)
svm_test_prediction = lin_svm.predict(vectorized_test)

log_reg_dev_pred = log_reg.predict(vectorized_dev)
svm_dev_pred = lin_svm.predict(vectorized_dev)

## Printing metrics and storing confusion matrix in var ##
log_confusion_matrix_dev = print_metrics("Logistic Regression Dev", dev_labels, log_reg_dev_pred)
svm_confusion_matrix_dev =print_metrics("Linear SVM Dev", dev_labels, svm_dev_pred)
log_confusion_matrix_test = print_metrics("Logistic Regression Test", test_labels, log_reg_test_prediction)
svm_confusion_matrix_test = print_metrics("Linear SVM Test", test_labels, svm_test_prediction)

## Printing confusion matrices ## 
print("Log Confusion Matrix Dev : ", log_confusion_matrix_dev)
print("SVM Confusion Matrix Dev: ", svm_confusion_matrix_dev)

print("Log Confusion Matrix Test: ", log_confusion_matrix_test)
print("SVM Confusion Matrix Test: ", svm_confusion_matrix_test)


## Finding Errors ##
print_misclassified(dataset['test']['description'], test_labels, log_reg_test_prediction, "Logistic Regression")
print_misclassified(dataset['test']['description'], test_labels, svm_test_prediction, "Linear SVM")
