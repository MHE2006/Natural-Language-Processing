from src.helper import get_ag_news_split, SEED
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

dataset, dev_split = get_ag_news_split(seed=SEED)


class DataPipeline:
    def __init__(self, max_length=128, batch_size=64, tokenizer_name="bert-base-uncased"):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size 

    def _tokenize_and_encode(self, text_list):
        return self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def create_dataloader(self, dataset,shuffle=False):
        texts = [str(t) + " " + str(d) for t, d in zip(dataset['title'], dataset['description'])]
        labels = torch.tensor(dataset['label'])- 1

        encodings = self._tokenize_and_encode(texts)

        tensor_dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
        return DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def prepare_all_loaders(self, dataset, dev_split):
        train_data = dev_split['train']
        dev_data = dev_split['test']  
        test_data = dataset['test']

        train_loader = self.create_dataloader(train_data,shuffle=True)
        dev_loader = self.create_dataloader(dev_data)
        test_loader = self.create_dataloader(test_data)

        return train_loader, dev_loader, test_loader