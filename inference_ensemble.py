# %%
from transformers import AutoModelForSequenceClassification
from utils import *
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from glob import glob
import os
torch.cuda.set_device('cuda:0')

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
def main(data):
    model_save_pathes = [
        "klue-roberta-large",
        "xlm-roberta-large",
        ]


    model_address = [
        "klue/roberta-large",
        "xlm-roberta-large",
        ]

    model_root_path = "./model/experiment_20-fold-cv/"
    
    final_logits = []
    for i in range(len(model_save_pathes)):
        cv_pathes = glob(os.path.join(model_root_path, model_save_pathes[i], '*'))
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
        for path in cv_pathes:
            logits.append(predict(path, loader))
        final_logits.append(torch.sum(torch.stack(logits), dim=0) / len(logits))
    return final_logits

# %%

data = pd.read_csv('./data/test_data.csv')
logit = main(data)

label2str = {
    0 : "entailment" ,
    1 : "contradiction",
    2 : "neutral" 
}
data['label'] = torch.argmax(torch.sum(torch.stack(logit), dim=0) / len(logit), dim=1)
data['label'] = data['label'].replace(label2str)
data['index'] = data.index 
data[['index', 'label']].to_csv('./data/submission_20-fold_cv_ensemble_with_2_model.csv', index=False)
# %%
