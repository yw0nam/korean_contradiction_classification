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
def main():
    model_save_pathes = ["klue-roberta-large",
                         "xlm-roberta-large"]


    model_address = ["klue/roberta-large",
                    "xlm-roberta-large"]

    model_root_path = "./model/experiment_5-fold-cv/"
    
    test = pd.read_csv('./data/test.tsv', sep='\t',
                        header=None, names=['text', 'label'])
    final_logits = []
    for i in range(len(model_save_pathes)):
        cv_pathes = glob(os.path.join(model_root_path, model_save_pathes[i], '*'))
        tokenizer = AutoTokenizer.from_pretrained(model_address[i])

        collate_arg = {
            'tokenizer': tokenizer,
            'max_length': 512,
            'with_text': False,
            'inference': True,
        }
        test_dataset = load_Dataset(test['text'].to_list(), test['label'].to_list())
        loader = DataLoader(test_dataset, batch_size=64,
                            collate_fn=dataCollator(**collate_arg), shuffle=False)

        logits = []
        for path in cv_pathes:
            logits.append(predict(path, loader))
        final_logits.append(torch.sum(torch.stack(logits), dim=0) / 5)
    return final_logits

# %%
logit = main()
# %%
torch.argmax(torch.sum(torch.stack(logit), dim=0) / 2, dim=1)
# %%
test = pd.read_csv('./data/test.tsv', sep='\t',
                header=None, names=['text', 'label'])

# %%
label2str = {
    0 : "entailment" ,
    1 : "contradiction",
    2 : "neutral" 
}
# %%
test['label'] = torch.argmax(torch.sum(torch.stack(logit), dim=0) / 2, dim=1)

# %%
test['label'] = test['label'].replace(label2str)
# %%
test['index'] = test.index 
test[['index', 'label']].to_csv('./data/submission_5-fold_cv_ensemble_with_2_model.csv', index=False)
# %%
