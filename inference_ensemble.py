# %%
from transformers import AutoModelForSequenceClassification
from utils import *
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from glob import glob
import os
import argparse
from tqdm import tqdm

label2str = {
    0 : "entailment" ,
    1 : "contradiction",
    2 : "neutral" 
}

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_save_path', required=True, type=str)
    p.add_argument('--model_address', required=True, type=str)
    p.add_argument('--model_root_path', required=True, type=str)
    p.add_argument('--test_fn', default='./data/test_data.csv')
    p.add_argument('--save_path', default="./submission/submission.csv", required=True)
    config = p.parse_args()

    return config

@torch.no_grad()
def predict(model_path, loader):
    output_ls = []
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    model.eval()
    for data in loader:
        input_ids, attention_mask = data['input_ids'].to('cuda'), data['attention_mask'].to('cuda')
        out = model(input_ids=input_ids, attention_mask=attention_mask)[0].detach().to('cpu')
        output_ls.append(torch.nn.functional.softmax(out, dim=1))
    return torch.cat(output_ls)

# %%
def main(config):
    
    data = pd.read_csv(config.test_fn)
    model_save_pathes = config.model_save_path.split('|')
    model_address = config.model_address.split('|')
    
    final_logits = []
    for i in range(len(model_save_pathes)):
        print("Model- %s Inference Start"%model_address[i])
        cv_pathes = glob(os.path.join(config.model_root_path, model_save_pathes[i], '*'))
        tokenizer = AutoTokenizer.from_pretrained(model_address[i])

        collate_arg = {
            'tokenizer': tokenizer,
            'max_length': 512,
            'inference': True,
        }
        test_dataset = load_Dataset(
            data['premise'].to_list(), data['hypothesis'].to_list(), data['label'].to_list()
            )
        loader = DataLoader(test_dataset, batch_size=64,
                            collate_fn=dataCollator(**collate_arg), shuffle=False)

        logits = []
        for path in tqdm(cv_pathes):
            logits.append(predict(path, loader))
        final_logits.append(torch.sum(torch.stack(logits), dim=0) / len(logits))
        
        data['label'] = torch.argmax(
            torch.sum(torch.stack(final_logits), dim=0) / len(final_logits), dim=1)
        data['label'] = data['label'].replace(label2str)
        data['index'] = data.index
        data[['index', 'label']].to_csv(config.save_path, index=False)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
