import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from packages.utils.tools import get_mAP


def train(train_loader, model, cnfs, optimizer):
    accuracy = 0.0
    num = 0
    model.train()
    loss_list = []
    for batch_idx, data in enumerate(tqdm(train_loader, ncols=80)):
        optimizer.zero_grad()
        logits, face_logits, ctx_logits, fbg_logits, relats_logits = model(data, cnfs)
        masks = data['masks']
        data_len = len(masks[masks==1])
        labels = data['labels'].cuda()
        loss = get_loss(cnfs, data, logits)
        if cnfs["lambda"]['face'] != 0:
            loss += cnfs["lambda"]['face'] * get_loss(cnfs, data, face_logits)
        if cnfs["lambda"]['ctx'] != 0:
            loss += cnfs["lambda"]['ctx'] * get_loss(cnfs, data, ctx_logits)
        if cnfs["lambda"]['fbg'] != 0:
            loss += cnfs["lambda"]['fbg'] * get_loss(cnfs, data, fbg_logits)
        if cnfs["lambda"]['relat'] != 0:
            loss += cnfs["lambda"]['relat'] * get_loss(cnfs, data, relats_logits)
        accuracy += get_accuracy_num(logits, labels, masks)
        num += data_len
        if loss == 0:
            del logits, loss
            continue
        loss.backward()
        optimizer.step()
        loss_list.append(float(loss.data.cpu()))
    return accuracy / num, np.mean(np.array(loss_list)), np.max(np.array(loss_list))



def eval_or_test(data_loader, model, cnfs=None):
    model.eval()
    probability_list = []
    real_label_list = []
    for batch_idx, data in enumerate(tqdm(data_loader, ncols=80)):
        # num += 1
        with torch.no_grad():
            logit = model(data, cnfs)
        soft_logit = torch.softmax(logit, dim=1)
        probability = soft_logit[:, 1]
        probability = probability.data.cpu().numpy()
        labels = data['labels'].data.cpu().numpy()
        probability_list.append(probability)
        real_label_list.append(labels)
    mAP = get_mAP(probability_list, real_label_list)
    return mAP


def get_accuracy_num(logits, labels, masks):
    softmax_logits = F.softmax(logits, 1)
    prediction = torch.max(softmax_logits, 1)[1]
    prediction = prediction[masks==1]
    labels = labels[masks==1]
    pred_y = prediction.data.cpu().numpy()
    target_y = labels.data.cpu().numpy()
    return sum(pred_y == target_y)


def get_loss(cnfs, data, logits):
    loss = torch.Tensor([0.0]).float().view(()).cuda()
    loss_dict = init_loss_dict()
    loss_weights = cnfs["loss_weights"]
    for key, value in loss_weights.items():
        if key in loss_dict.keys():
            loss += float(value) * loss_dict[key](cnfs, data, logits)
    return loss


def init_loss_dict():
    loss_dict = {}
    loss_dict['ce_loss'] = get_cross_entropy_loss
    return loss_dict


def get_cross_entropy_loss(cnfs, data, logits):
    labels = data['labels'].cuda()
    masks = data['masks'].cuda()
    ce_loss = get_cross_entropy_with_mask(logits, labels, masks) + 1e-6
    return ce_loss


def get_cross_entropy_with_mask(predicts, labels, mask):
    # mask = torch.cat([mask.view(-1, 1), mask.view(-1, 1)], dim=1)
    batch_size = len(mask[mask==1])
    mask = mask.expand(2, mask.shape[0]).transpose(0, 1)
    log_likelihood = - F.log_softmax(predicts, dim=1)
    target = [1-labels, labels]
    target = torch.stack(target, 0)
    target.transpose_(0, 1)
    return torch.sum(torch.mul(torch.mul(log_likelihood, target.float()), mask)) / batch_size
