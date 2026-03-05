import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=4, dropout=0.3):
        super(CNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=5)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x,attention_mask=None):
        x = self.embedding(x) 

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)       
        
        x = x.permute(0, 2, 1) 
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2)
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        
       
        out = torch.cat([x1, x2, x3], dim=1) 
        
        out = self.dropout(out)
        out = self.fc(out) 
        
        return out
    

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_classes=4, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True 
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        embedded = self.embedding(x)
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1)
        
        lstm_out, _ = self.lstm(embedded)
        pooled_out, _ = torch.max(lstm_out, dim=1)

        out = self.dropout(pooled_out)
        out = self.fc(out) 
        return out