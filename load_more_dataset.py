# %%
import pandas as pd
import torch
from datasets import load_dataset
import re
# %%
dataset = load_dataset('kor_nlu', 'nli')
csv = pd.read_csv('./data/train_data.csv')
# %%
data = pd.concat([csv, 
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

# %%
