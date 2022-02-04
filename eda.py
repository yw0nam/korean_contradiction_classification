# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
# %%
csv = pd.read_csv('./data/train_data.csv', index_col=False)
csv = csv.drop('index', axis=1)
# %%
test = pd.read_csv('./data/test_data.csv', index_col=False)
test = test.drop('index', axis=1)
# %%
csv['text'] = csv['premise'] +"[SEP]" + csv['hypothesis']
test['text'] = test['premise'] +"[SEP]" + test['hypothesis']
# %%
label_dicts = {
    "entailment" : 0,
    "contradiction": 1,
    "neutral" :2
}
csv['label'] = csv['label'].replace(label_dicts)
test['label'] = test['label'].replace(label_dicts)
# %%
train, val = train_test_split(csv, test_size=500, stratify=csv['label'], random_state=1004)
# %%
train[['text', 'label']].to_csv('./data/train.tsv', sep='\t', index=False, header=None)
val[['text', 'label']].to_csv('./data/val.tsv', sep='\t', index=False, header=None)
test[['text', 'label']].to_csv('./data/test.tsv', sep='\t', index=False, header=None)
# %%
encode_label = torch.nn.functional.one_hot(torch.tensor(csv['label'].iloc[0]), 3)
# %%
temp = ['2', '1', '0', '1', '0', '2', '0', '1', '2', '0', '1', '1', '0', '0', '1', '2', '0', '0', '2', '0', '2', '0', '0', '1', '2', '2', '2', '1', '1', '1', '0', '0', '0', '2', '1', '2', '0', '2', '0', '0', '1', '2', '2', '1', '0', '2', '1', '1', '0', '1', '2', '1', '1', '1', '2', '0', '0', '0', '1', '0', '0', '1', '0', '0', '1', '0', '1', '0', '0', '2', '0', '1', '0', '1', '1', '2', '2', '2', '0', '2', '0', '0', '2', '0', '1', '2', '2', '0', '2', '0', '2', '2', '2', '1', '1', '0', '2', '1', '1', '2', '1', '2', '0', '0', '0', '2', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '1', '1', '0', '0', '1', '1', '0', '2', '0', '2', '0', '0']
# %%
torch.tensor(list(map(lambda x: int(x), temp)))
# %%
