# %%
import pandas as pd
import torch
from datasets import load_dataset
import re
# %%
dataset = load_dataset('kor_nlu', 'nli')
csv = pd.read_csv('./data/train_data.csv')
# %%
data = pd.concat([ 
    pd.DataFrame({
    'premise': dataset['train']['premise'],
    'hypothesis': dataset['train']['hypothesis'],
    'label': dataset['train']['label']
    }), 
    pd.DataFrame({
    'premise': dataset['validation']['premise'],
    'hypothesis': dataset['validation']['hypothesis'],
    'label': dataset['validation']['label']
    })
])
# %%

comp = re.compile(r'[a-zA-Z]')
data = data.drop_duplicates(subset='premise')
data['include_premise_en'] = data['premise'].map(lambda x: 1 if not comp.match(x) else 0)
data['include_hypothesis_en'] = data['hypothesis'].map(lambda x: 1 if not comp.match(x) else 0)
# %%
data['include_premise_en'].value_counts()
# %%
data['include_hypothesis_en'].value_counts()

label_dicts = {
    "entailment" : 0,
    "contradiction": 1,
    "neutral" :2
}

data['label'] = data['label'].replace(label_dicts)
# %%
data.query("include_premise_en == 1 and include_hypothesis_en == 1")[['premise',
                                                                      'hypothesis', 
                                                                      'label']].to_csv('./data/add_extra_data.csv', index=False)
# %%

csv = pd.read_csv('./data/add_extra_data.csv')
label_dicts = {
    "entailment" : 0,
    "contradiction": 1,
    "neutral" :2
}
# %%
csv['label'] = csv['label'].replace(label_dicts)
# %%
csv['label']
# %%
temp = pd.read_csv('./data/train_data.csv')
# %%
temp['premise'].to_list() + csv['premise'].to_list()
# %%
csv['text'] = csv['premise'] + ' '+ csv['hypothesis']
temp['text'] = temp['premise'] + ' '  + temp['hypothesis']
# %%
temp['text_len'] = temp['text'].map(lambda x: len(x))
csv['text_len'] = csv['text'].map(lambda x: len(x))
# %%
print(csv['text_len'].min(), csv['text_len'].max(), csv['text_len'].mean())
print(temp['text_len'].min(), temp['text_len'].max(), temp['text_len'].mean())
# %%
temp.query("text_len <= 30")
# %%
csv.query("text_len >= 26 and text_len <=192")
# %%
temp['text_len'].std()
# %%
upper = int(temp['text_len'].mean() + temp['text_len'].std())
lower = int(temp['text_len'].mean() - temp['text_len'].std())
# %%
csv.query("text_len >= @lower and text_len <= @upper")[['premise','hypothesis', 'label']].to_csv('./data/add_extra_data_clip.csv', index=False)
# %%
csv
# %%
