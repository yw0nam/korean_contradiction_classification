CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-large \
                                        --save_path ./model/klue_roberta-large_random_state_1 \
                                        --n_epochs 20 \
                                        --random_state 1 \

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-large \
                                        --save_path ./model/klue_roberta-large_random_state_11 \
                                        --n_epochs 20 \
                                        --random_state 11

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-large \
                                        --save_path ./model/klue_roberta-large_random_state_111 \
                                        --n_epochs 20 \
                                        --random_state 111

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-large \
                                        --save_path ./model/klue_roberta-large_random_state_1111 \
                                        --n_epochs 20 \
                                        --random_state 1111 

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/csv.tsv --model_fn klue/roberta-large \
                                        --save_path ./model/klue_roberta-large_random_state_1004 \
                                        --n_epochs 20 \
                                        --random_state 1004
