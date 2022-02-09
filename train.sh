CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn xlm-roberta-large \
                                        --save_path ./model/experiment_5-fold-cv/xlm-roberta-large 

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn bert-base-multilingual-uncased \
                                        --save_path ./model/experiment_5-fold-cv/bert-base-multilingual-uncased 

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-base \
                                        --save_path ./model/experiment_5-fold-cv/klue-roberta-base 

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-small \
                                        --save_path ./model/experiment_5-fold-cv/klue-roberta-small 

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-large \
                                        --save_path ./model/experiment_5-fold-cv/klue-roberta-large 