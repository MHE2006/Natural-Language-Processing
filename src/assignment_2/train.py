import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import copy

def train_model(model, train_loader, dev_loader, epochs=10, learning_rate=1e-3, patience=3, device="cpu"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'dev_loss': [], 'dev_acc': [], 'dev_f1': []}

    best_dev_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch_inputs, attention_mask, batch_labels in train_loader:
            batch_inputs, attention_mask, batch_labels = batch_inputs.to(device), attention_mask.to(device), batch_labels.to(device)

            
            optimizer.zero_grad()
            outputs = model(batch_inputs, attention_mask=attention_mask)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        model.eval()
        total_dev_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_inputs, attention_mask, batch_labels in dev_loader:
                batch_inputs, attention_mask, batch_labels = batch_inputs.to(device), attention_mask.to(device), batch_labels.to(device)
                
                outputs = model(batch_inputs, attention_mask=attention_mask)
                loss = criterion(outputs, batch_labels)
                total_dev_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
        avg_dev_loss = total_dev_loss / len(dev_loader)
        dev_acc = accuracy_score(all_labels, all_preds)
        dev_f1 = f1_score(all_labels, all_preds, average='macro')
        
        history['dev_loss'].append(avg_dev_loss)
        history['dev_acc'].append(dev_acc)
        history['dev_f1'].append(dev_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")
        
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
                
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model, history


