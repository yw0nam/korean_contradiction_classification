CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/add_extra_data.csv --model_fn xlm-roberta-large \
                                        --save_path ./model/experiment_10-fold-cv/xlm-roberta-large \
                                        --fold 10

rm -rf ./model/experiment_10-fold-cv_extra_data/xlm-roberta-large/*/checkpoints

CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/add_extra_data.csv --model_fn klue/roberta-large \
                                        --save_path ./model/experiment_10-fold-cv/klue-roberta-large \
                                        --fold 10

rm -rf ./model/experiment_10-fold-cv_extra_data/klue-roberta-large/*/checkpoints


CUDA_VISIBLE_DEVICES=0,1 python train.py --train_fn ./data/add_extra_data.csv --model_fn Huffon/klue-roberta-base-nli \
                                        --save_path ./model/experiment_10-fold-cv/Huffon-klue-roberta-base-nli \
                                        --fold 10

rm -rf ./model/experiment_10-fold-cv_extra_data/Huffon-klue-roberta-base-nli/*/checkpoints