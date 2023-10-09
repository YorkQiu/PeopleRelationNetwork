# import git
from datetime import datetime
import argparse
import inspect
import sys
import os
import itertools
import importlib
import pickle
import random
import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import numpy as np

from packages.utils import tools

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', '-G', type=str, default="-1")
    parser.add_argument('--config', '-C', type=str, default="")
    parser.add_argument('--is_resume', '-r', action="store_true")
    parser.add_argument('--resume_model', '-RM', type=str, default="")
    parser.add_argument('--resume_records', '-RR', action="store_true")
    parser.add_argument('--force', '-f', action="store_true")
    parser.add_argument('--eval', action='store_true')
    return parser.parse_args()


def init_seed(seed):
    '''
    Set random seed for python, numpy, torch(cpu), torch(gpu).
    Update this function when new library that requires seed.
    '''
    random.seed(seed)
    np.random.seed(random.randrange(2**31))
    torch.manual_seed(random.randrange(2**31))
    torch.cuda.manual_seed_all(random.randrange(2**31))
    # this may lead to less efficiency
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_model(model_name, models_path, cnfs=None):
    is_find = False
    module_files = ['packages.models.' + f[:-3] for f in os.listdir(
        models_path) if os.path.isfile(os.path.join(models_path, f)) and f != '__init__.py']
    models = []
    for f in module_files:
        module = importlib.import_module(f)
        models.append(inspect.getmembers(module, inspect.isclass))
    all_models = itertools.chain(*models)

    for name, val in all_models:
        if name == model_name:
            model = val(cnfs)
            is_find = True
            return model
    try:
        assert is_find
    except AssertionError:
        print("Error! invaild model type!!!")
        sys.exit(1)



def init_data(cnfs):
    """
    all processed data are stored in .pkl form.
    """
    data_path = cnfs['data']['path'][cnfs['dataset']]
    return pickle.load(open(data_path, 'rb'))


def init_loader(cnfs, dataset, batch_size, is_train=False):
    def my_collate_fn(batch):
        if len(batch) == 1:
            return batch[0]
        else:
            reg = {}
            tmp_reg = batch[0]
            keys = tmp_reg.keys()
            for key in keys:
                tmp_list = []
                for b in batch:
                    tmp_list.append(b[key])
                try:
                    reg[key] = torch.cat(tmp_list, dim=0)
                except Exception:
                    print("catch exception.")
                    print(key)
                    for tmp in tmp_list:
                        print(tmp.shape)
                    exit(-1)
            return reg
    data_loader = DataLoader(dataset, batch_sampler=BatchSampler(RandomSampler(dataset, False), batch_size=batch_size,
                                                                 drop_last=is_train), num_workers=cnfs['num_worker'], pin_memory=False, collate_fn=my_collate_fn)
    return data_loader


def init_optimizer_and_scheduler(cnfs, model, is_multi_stage=False, cur_iter=1):
    if cnfs['opt']['type'] == 'sgd':
        opt = optim.SGD(model.parameters(), lr=cnfs['opt']['lr'],
                        momentum=cnfs['opt']['m'], weight_decay=cnfs['opt']['wd'])
        print('opt: SGD')
    elif cnfs['opt']['type'] == 'adam':
        opt = optim.Adam(
            model.parameters(), lr=cnfs["opt"]["lr"], amsgrad=True)
        print('opt: ADAM')
    else:
        print('wrong optimizier!')
        sys.exit(-1)
    if is_multi_stage:
        cnfs["opt"]["lde"] = cnfs["multi_stage"]["lde"][cur_iter-1]
    if isinstance(cnfs["opt"]["lde"], list):
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=cnfs["opt"]["lde"], gamma=cnfs["opt"]["gamma"])
    else:
        scheduler = optim.lr_scheduler.StepLR(
            opt, step_size=cnfs['opt']['lde'], gamma=cnfs['opt']['gamma'])
    return opt, scheduler


def init_record(cnfs, is_resume=False, is_force=False):
    """
    initate the file to record model performance.
    """
    file_name = cnfs['model']["name"] + '.txt'
    file_path = os.path.join(cnfs['record_path'], file_name)
    if is_resume:
        try:
            assert os.path.exists(file_path)
        except AssertionError:
            print("Error! in resume mode, but do not have corresponding record file.")
            sys.exit(-1)
        return file_path
    if os.path.exists(file_path):
        if not is_force:
            print("record file exist! add --force to force a new training progress.")
            sys.exit(-1)
        os.remove(file_path)
    f = open(file_path, 'w+')
    f.close()
    records = {}
    #repo = git.Repo(path=os.getcwd())
    #records["git_hash_code"], records["start_time"] = repo.head.object.hexsha, datetime.today()
    tools.record_info(file_path, records, records.keys())
    return file_path


def init_record_dict():
    return {
        'train_acc': 0.0,
        'best_acc': 0.0,
        'best_epoch': 0,
        'test_acc': 0.0,
        'best_test_acc': 0.0,
        'best_test_epoch': 0,
        'mAP': 0.0,
        'best_mAP': 0.0,
        'best_mAP_epoch': 0,
        'best_iter': 0,
        'best_test_iter': 0,
        'best_mAP_iter': 0,
        'new_mAP': 0.0,
        'best_new_mAP': 0.0,
        'best_new_mAP_iter': 0,
        'best_new_mAP_epoch': 0,
        'val_mAP': 0.0,
        'best_val_mAP': 0.0,
        'best_val_mAP_epoch': 0,
        'early_stop_mAP': 0.0}


def init_paths_and_cuda(cnfs):
    def check_path_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)
    cnfs['record_path'] = os.path.join(
        os.getcwd(), 'records', cnfs['dataset'])
    check_path_exist(cnfs['record_path'])
    cnfs['save_path'] = os.path.join(
        os.getcwd(), 'saves', cnfs['dataset'])
    check_path_exist(cnfs['save_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = cnfs['gpu_id']

