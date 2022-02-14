import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class dataCollator():

    def __init__(self, tokenizer, max_length, inference=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inference = inference
        
    def __call__(self, samples):
        premise = [s['premise'] for s in samples]
        hypothesis = [s['hypothesis'] for s in samples]
        label = [s['label'] for s in samples]
        
        input_encoding = self.tokenizer(
            list(premise),
            list(hypothesis),
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length
        )

        
        return_value = {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
        }
        if not self.inference:
            encode_label = torch.tensor(list(map(lambda x: int(x), label)))
            return_value['labels'] = encode_label
        return return_value


class load_Dataset(Dataset):

    def __init__(self, premise, hypothesis, labels):
        self.premise = premise
        self.hypothesis = hypothesis
        self.labels = labels
        
    def __len__(self):
        return len(self.premise)
    
    def __getitem__(self, item):
        premise = str(self.premise[item])
        hypothesis = str(self.hypothesis[item])
        label = str(self.labels[item])

        return {
            'premise': premise,
            'hypothesis': hypothesis,
            'label': label,
        }
