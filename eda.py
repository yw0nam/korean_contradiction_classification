# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--train_fn', required=True)
    p.add_argument('--test_fn', required=True)
    p.add_argument('--valid_size', default=500)
    p.add_argument('--save_path', default="./data/")
    config = p.parse_args()

    return config

def main(config):
    csv = pd.read_csv(config.train_fn, index_col=False)
    csv = csv.drop('index', axis=1)
    test = pd.read_csv(config.test_fn, index_col=False)
    test = test.drop('index', axis=1)

    csv['text'] = csv['premise'] +"[SEP]" + csv['hypothesis']
    test['text'] = test['premise'] +"[SEP]" + test['hypothesis']
    label_dicts = {
        "entailment" : 0,
        "contradiction": 1,
        "neutral" :2
    }
    csv['label'] = csv['label'].replace(label_dicts)
    test['label'] = test['label'].replace(label_dicts)

    train, val = train_test_split(csv, test_size=config.valid_size, stratify=csv['label'], random_state=1004)
    
    train[['text', 'label']].to_csv(os.path.join(config.save_path,'train.tsv'), sep='\t', index=False, header=None)
    val[['text', 'label']].to_csv(os.path.join(config.save_path,'val.tsv'), sep='\t', index=False, header=None)
    test[['text', 'label']].to_csv(os.path.join(config.save_path,'test.tsv'), sep='\t', index=False, header=None)

if __name__ == '__main__':
    config = define_argparser()
    main(config)