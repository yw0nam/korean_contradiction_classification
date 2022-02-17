CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/train_data.csv --model_fn ./model/experiment_finetune_extra_data/xlm-roberta-large \
                                        --model_address xlm-roberta-large \
                                        --save_path ./model/experiment_10-fold-cv_after_finetune/xlm-roberta-large \
                                        --fold 10 \
                                        --n_epochs 10
                                        # --add_extra_data ./data/add_extra_data_clip.csv

rm -rf ./model/experiment_10-fold-cv_after_finetune/xlm-roberta-large/*/checkpoints

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/train_data.csv --model_fn ./model/experiment_finetune_extra_data/klue-roberta-large \
                                        --model_address klue/roberta-large \
                                        --save_path ./model/experiment_10-fold-cv_after_finetune/klue-roberta-large \
                                        --fold 10 \
                                        --n_epochs 10
                                        # --add_extra_data ./data/add_extra_data_clip.csv

rm -rf ./model/experiment_10-fold-cv_after_finetune/klue-roberta-large/*/checkpoints