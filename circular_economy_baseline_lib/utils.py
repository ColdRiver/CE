import warnings
import sys
import time
import pytz
import logging
import os
from datetime import datetime
from config import config
warnings.filterwarnings("ignore")

import random
import numpy as np 
import torch

def seedBasic(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed):
    seedBasic(seed)
    seedTorch(seed)


process_start_time = datetime.now(pytz.timezone('US/Eastern'))
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S")  #+ '{desc}'
#process_start_time.strftime("%Y%m%d_%H%M%S") #+ '{desc}'


def get_result_folder():
    return result_folder

def get_baseline_result_folder(phi = 0.2):
    baseline_result_folder = './result/baselines/' + "mathopt_phi0={}_phi1={}_phi2={}".format(phi[0],
                                                                                                          phi[1], phi[2]) + "pricefactor={}".format(config['price_factor'])
    #import pdb; pdb.set_trace()
    return baseline_result_folder

def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0

class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))
