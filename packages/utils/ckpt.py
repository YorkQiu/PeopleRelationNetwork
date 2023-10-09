import os
import torch
import torch.nn as nn
from tqdm import tqdm

def save_checkpoint(state, model_name, cnfs=None):
    epoch = state['epoch']
    
    if epoch == cnfs["epoch"]:
        file_name = model_name + '.pkl'
    else:
        file_name = model_name + '_%d_' % epoch + '.pkl'
    save_name = os.path.join(cnfs["save_path"], file_name)
    print('checkpoint path: %s' % (save_name))
    if os.path.exists(save_name):
        os.remove(save_name)
    torch.save(state, save_name)
    return save_name

def del_tmp_ckpts(files):
    print('deleting tmporary checkpoints.')
    for f in tqdm(files, ncols=80):
        if f.endswith("best.pkl"):
            continue
        os.remove(f)
    print('finished!')

def load_checkpoint(resume_file, scheduler, model, optimizer):
    print("=> loading checkpoint '{}'".format(resume_file))
    checkpoint = torch.load(resume_file)
    start_epoch = checkpoint['epoch']
    checkpoint.pop('epoch')
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    checkpoint.pop('scheduler')
    model.load_state_dict(checkpoint['state_dict'])
    checkpoint.pop('state_dict')
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint.pop('optimizer')
    other_param = checkpoint
    print("=> loaded checkpoint '{}' (epoch {})".format(
        resume_file, start_epoch))
    return start_epoch, scheduler, model, optimizer, other_param

def resume_model(cnfs, model, scheduler, optimizer, records, is_parallel=True):
    resume_file = os.path.join(cnfs["save_path"], cnfs["resume_model"])
    if os.path.isfile(resume_file):
        save_epoch, scheduler, model, optimizer, other_param = load_checkpoint(
            resume_file, scheduler, model, optimizer)
        start_epoch = save_epoch + 1
        if is_parallel:
            model = nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        records.update(other_param)
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))
    return start_epoch, scheduler, model, optimizer, records
