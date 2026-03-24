import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score



def length_bucket_evaluation(trainer,tokenized_dataset):
    predict_output = trainer.predict(tokenized_dataset)
    all_predictions = np.argmax(predict_output.predictions, axis=-1)
    all_labels = predict_output.label_ids
    attention_masks = tokenized_dataset['attention_mask']

    b1_preds = []
    b1_labels = []
    b2_preds = []
    b2_labels = []
    b3_preds = []
    b3_labels = []
    b4_preds = []
    b4_labels = []


    for i in range(len(all_predictions)):
        length = sum(attention_masks[i]).item()

        pred = all_predictions[i]
        label = all_labels[i]

        if length <= 32:
            b1_preds.append(pred)
            b1_labels.append(label)
        elif length <= 64:
            b2_preds.append(pred)
            b2_labels.append(label)
        elif length <= 96:
            b3_preds.append(pred)
            b3_labels.append(label)
        else:
            b4_preds.append(pred)
            b4_labels.append(label)

    
    print("Bucket under 32 Tokens")
    print(f"Total Examples: {len(b1_labels)}")
    print(f"Accuracy: {accuracy_score(b1_labels, b1_preds):.4f}")
    print(f"Macro-F1: {f1_score(b1_labels, b1_preds, average='macro'):.4f}")
       
       
    print("Bucket between 32 and 64 tokens")
    print(f"Total Examples: {len(b2_labels)}")
    print(f"Accuracy: {accuracy_score(b2_labels, b2_preds):.4f}")
    print(f"Macro-F1: {f1_score(b2_labels, b2_preds, average='macro'):.4f}")

    print("Bucket between 64 and 96 tokens")
    print(f"Total Examples: {len(b3_labels)}")
    print(f"Accuracy: {accuracy_score(b3_labels, b3_preds):.4f}")
    print(f"Macro-F1: {f1_score(b3_labels, b3_preds, average='macro'):.4f}")


    print("Bucket more than 96 tokens")
    print(f"Total Examples: {len(b4_labels)}")
    print(f"Accuracy: {accuracy_score(b4_labels, b4_preds):.4f}")
    print(f"Macro-F1: {f1_score(b4_labels, b4_preds, average='macro'):.4f}")



def keyword_masking_evalutation(trainer, tokenized_dataset, tokenizer, keyword_list):
    
    tokenized_dataset = trainer.data_collator(tokenized_dataset)
    
    keyword_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in keyword_list]
    mask_id = tokenizer.mask_token_id

    keyword_tensor = torch.tensor(keyword_ids)

    all_input_ids = torch.as_tensor(tokenized_dataset['input_ids'])
    all_masks = torch.as_tensor(tokenized_dataset['attention_mask'])
    all_labels = torch.as_tensor(tokenized_dataset['labels'])

    
    keyword_locations = torch.isin(all_input_ids, keyword_tensor)    
    rows_with_keywords = keyword_locations.any(dim=-1)

    orig_inputs = all_input_ids[rows_with_keywords]
    subset_masks = all_masks[rows_with_keywords]
    subset_labels = all_labels[rows_with_keywords]

    masked_inputs = torch.where(torch.isin(orig_inputs, keyword_tensor), mask_id, orig_inputs)
    
    masked_dataset = Dataset.from_dict({
    'input_ids': masked_inputs.cpu().numpy(), 
    'attention_mask': subset_masks.cpu().numpy(), 
    'labels': subset_labels.cpu().numpy()
    })

    mask_preds = np.argmax(trainer.predict(masked_dataset).predictions, axis=-1)
    mask_acc = accuracy_score(subset_labels.flatten(), mask_preds.flatten())

    print(f"Masked Accuracy:   {mask_acc:.4f}")


