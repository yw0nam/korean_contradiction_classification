# %%
import os
import shutil
from glob import glob
import argparse

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_root', required=True, type=str)
    p.add_argument('--fold', required=True, type=int)
    config = p.parse_args()

    return config

def main(config):
    for i in range(config.fold):
        path = os.path.join(config.model_root, str(i))
        shutil.copy(os.path.join(glob(path + '/checkpoints/*')[-1], 'trainer_state.json'), path)
        shutil.rmtree(os.path.join(path, 'checkpoints'))
        

if __name__ == '__main__':
    config = define_argparser()
    main(config)