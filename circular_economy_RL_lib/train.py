##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
import multiprocessing
from utils import create_logger

from trainer import Trainer


##########################################################################################
# parameters

logger_params = {
    'log_file': {
        'desc': 'train',
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def main():
    create_logger(**logger_params)
    _print_config()
    print(f'Number of CPUs: {multiprocessing.cpu_count()}')

    trainer = Trainer()

    trainer.learn()

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
    main()
