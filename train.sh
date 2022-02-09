CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn xlm-roberta-large \
                                        --save_path ./model/experiment_10-fold-cv/xlm-roberta-large \
                                        --fold 10

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-large \
                                        --save_path ./model/experiment_10-fold-cv/klue-roberta-large \
                                        --fold 10