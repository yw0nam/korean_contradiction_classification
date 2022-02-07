import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class dataCollator():

    def __init__(self, tokenizer, max_length, with_text=True, inference=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
        self.inference = inference
        
    def __call__(self, samples):
        input_seq = [s['input_seq'] for s in samples]
        label = [s['label'] for s in samples]
        
        input_encoding = self.tokenizer(
            input_seq,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        
        return_value = {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
        }
        if not self.inference:
            encode_label = torch.tensor(list(map(lambda x: int(x), label)))
            return_value['labels'] = encode_label
        if self.with_text:
            return_value['input_seq'] = input_seq
        return return_value


class load_Dataset(Dataset):

    def __init__(self, input_seq, labels):
        self.input_seq = input_seq
        self.labels = labels
    
    def __len__(self):
        return len(self.input_seq)
    
    def __getitem__(self, item):
        input_seq = str(self.input_seq[item])
        label = str(self.labels[item])

        return {
            'input_seq': input_seq,
            'label': label,
        }

def get_datasets(fn, random_state=None):
    with open(fn, 'r') as f:
        lines = f.readlines()

        input_seqs, labels = [], []
        for line in lines:
            if line.strip() != '':
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                input_seq, label = line.strip().split('\t')
                input_seqs += [input_seq]
                labels += [label]
    if random_state:
        train, val = train_test_split(list(zip(input_seqs, labels)), 
                                        test_size=0.1, 
                                        stratify=labels, 
                                        random_state=random_state)

        train_seq = [e[0] for e in train]
        train_label = [e[1] for e in train]

        val_seq = [e[0] for e in val]
        val_label = [e[1] for e in val]

        train_dataset = load_Dataset(train_seq, train_label)
        val_dataset = load_Dataset(val_seq, val_label)
        return train_dataset, val_dataset
    else:
        dataset = load_Dataset(input_seq, labels)
        return dataset

    