from helper import get_ag_news_split, SEED
from transformers import DistilBertTokenizer

## Getting AG News dataset and creating dev split ##
dataset, dev_split = get_ag_news_split(seed=SEED)

train_set = dev_split['train']
dev_set = dev_split['test']
test_set = dataset['test']

## Tokenizing using DistilBERT tokenizer ##
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(data):
    result = tokenizer(
        [t + " " + d for t, d in zip(data['title'], data['description'])],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    result["labels"] = [l - 1 for l in data["label"]]
    return result

def tokenize_dataset(dataset):

    tokenized = dataset.map(tokenize, batched=True,load_from_cache_file=False)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) 
    return tokenized


tokenized_train = tokenize_dataset(train_set)
tokenized_dev = tokenize_dataset(dev_set)
tokenized_test = tokenize_dataset(test_set)
