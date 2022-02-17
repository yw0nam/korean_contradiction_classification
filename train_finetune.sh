CUDA_VISIBLE_DEVICES=0,1 python train_without_cv.py --train_fn ./data/add_extra_data.csv --model_fn xlm-roberta-large \
                                        --save_path ./model/experiment_finetune_extra_data/xlm-roberta-large \
                                        --n_epochs 10 \

rm -rf ./model/experiment_finetune_extra_data/xlm-roberta-large/*/checkpoints

CUDA_VISIBLE_DEVICES=0,1 python train_without_cv.py --train_fn ./data/add_extra_data.csv --model_fn klue/roberta-large \
                                        --save_path ./model/experiment_finetune_extra_data/klue-roberta-large \
                                        --n_epochs 10 \

rm -rf ./model/experiment_finetune_extra_data/klue-roberta-large/*/checkpoints
