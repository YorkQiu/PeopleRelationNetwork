# pylint: disable=not-callable
import math
import random
import sys

import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset
from packages.utils.data.transforms import VIPRandomHFlip, VIPRandomResizedCrop


class BaseDataset(Dataset):
    def __init__(self, data, cnfs, is_train=False):
        self.data = data
        self.cnfs = cnfs
        self.is_train = is_train
        self.num_ins = cnfs['num_ins']
        self.data_name = cnfs["dataset"]
        self.mean = torch.Tensor(cnfs["data"]["mean"][self.data_name])
        self.std = torch.Tensor(cnfs["data"]["std"][self.data_name])
        self.padding_method = cnfs["data"]["padding_method"] if "padding_method" in cnfs["data"].keys() else "zero"
        print("padding_method: ", self.padding_method)
        self.to_tensor = T.ToTensor()
        self.face_xfmr = None
        self.ctx_xfmr = None
        self.img_xfmr = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._get_sgl_trn_item(idx) if self.is_train else self._get_sgl_item(idx)

    def _get_sgl_trn_item(self, idx):
        tmp_data = self.data[idx]
        img_path, bboxes, lbls, evt_lbl = tmp_data['img_path'], np.array(
            tmp_data['bboxes'], dtype=int), np.array(tmp_data['labels'], dtype=np.int64), tmp_data['event_label']
        imp_idxs, non_imp_idxs = self.get_idxs(lbls)
        img = Image.open(img_path).convert("RGB")
        img, faces, ctxs, bboxes, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes, rltv_pos, lbls = self.get_augmented_data(
            img, bboxes, imp_idxs, non_imp_idxs)
        bboxes = bboxes.tolist()
        img, faces, ctxs, bboxes, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes, rltv_pos = self.param_to_tensor(
            img, faces, ctxs, bboxes, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes, rltv_pos)
        valid_num = bboxes.shape[0]
        reg = {}
        reg['labels'] = torch.from_numpy(lbls).long()
        self.init_reg(reg, img=img, faces=faces, ctxs=ctxs, coords=coords, ctx_coords=ctx_coords, bboxes=bboxes,
                      ctx_bboxes=ctx_bboxes, norm_bboxes=norm_bboxes, norm_ctx_bboxes=norm_ctx_bboxes, rltv_pos=rltv_pos)
        valid_num = len(lbls)
        assert valid_num <= self.num_ins
        reg['imgs'] = torch.cat(
            [img.detach().clone().unsqueeze(0) for i in range(valid_num)], dim=0)
        reg['event_labels'] = torch.Tensor(
            [evt_lbl for i in range(valid_num)]).long()
        reg['masks'] = torch.ones(valid_num)
        reg['weight'] = torch.Tensor([1]).float()
        if valid_num < self.num_ins:
            if self.padding_method == "zero":
                keys = ['imgs', 'faces', 'ctxs', 'coords', 'ctx_coords', 'bboxes', 'ctx_bboxes',
                        'norm_bboxes', 'norm_ctx_bboxes', 'event_labels', 'labels', 'rltv_pos', 'masks']
                self._zero_pad_all(keys, reg)
            if self.padding_method == "person":
                keys = ['imgs', 'faces', 'ctxs', 'coords', 'ctx_coords', 'bboxes', 'ctx_bboxes',
                        'norm_bboxes', 'norm_ctx_bboxes', 'event_labels', 'rltv_pos', 'masks']
                idxs = np.arange(valid_num)
                choice_idxs = np.random.choice(idxs, self.num_ins - valid_num,  replace=True)
                imp_idxs = np.arange(len(lbls))[lbls == 1]
                tmp_labels = [1 if idx in imp_idxs else 0 for idx in choice_idxs]
                reg['labels'] = torch.cat([reg['labels'], torch.Tensor(tmp_labels).long()], dim=0)
                self._person_pad_all(keys, reg, choice_idxs)
        keys = ['imgs', 'faces', 'ctxs']
        self.norm_data(reg, keys)
        return reg

    def _get_sgl_item(self, idx):
        tmp_data = self.data[idx]
        img_path, bboxes, lbls, evt_lbl = tmp_data['img_path'], tmp_data['bboxes'], torch.Tensor(
            tmp_data["labels"]).long(), tmp_data['event_label']
        reg = {}
        img = Image.open(img_path).convert("RGB")
        faces, ctxs, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes = self.get_face_ctx_coord_normed_bboxes(img, bboxes, self.data_name)
        reg['labels'] = lbls
        valid_num = lbls.shape[0]
        img = img.resize((224, 224), Image.BILINEAR)
        width, height = img.size
        rltv_pos = self.get_rltv_pos(bboxes, width, height)
        img, faces, ctxs, bboxes, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes, rltv_pos = self.param_to_tensor(
            img, faces, ctxs, bboxes, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes, rltv_pos)
        reg['imgs'] = torch.cat([img.detach().clone().unsqueeze(0)
                                     for i in range(valid_num)], dim=0)
        self.init_reg(reg, faces=faces, ctxs=ctxs, coords=coords, ctx_coords=ctx_coords, bboxes=bboxes,
                      ctx_bboxes=ctx_bboxes, norm_bboxes=norm_bboxes, norm_ctx_bboxes=norm_ctx_bboxes, rltv_pos=rltv_pos)
        reg['event_labels'] = torch.Tensor(
            [evt_lbl for i in range(valid_num)]).long()
        reg['masks'] = torch.ones(valid_num)
        reg['img_id'] = tmp_data['img_id']
        reg['img_path'] = img_path
        reg['origin_bboxes'] = tmp_data['bboxes']
        keys = ['imgs', 'faces', 'ctxs']
        self.norm_data(reg, keys)
        return reg

    def get_augmented_data(self, img, bboxes, imp_idxs, non_imp_idxs):
        return [None for i in range(11)]

    def get_rltv_pos(self, bboxes, width, height):
        norm_bboxes = []
        pos_x = []
        pos_y = []
        for bbox in bboxes:
            x, y, w, h = bbox
            area = w * h
            x, y, w, h = self.norm_bbox(bbox, width, height)
            pos_x.append(x)
            pos_y.append(y)
            norm_bboxes.append([x, y, w, h, area])
        mean_x = sum(pos_x) / len(pos_x)
        mean_y = sum(pos_y) / len(pos_y)
        rltv_pos = []
        for box_area in norm_bboxes:
            x, y, w, h, area = box_area
            rltv_x = x - mean_x
            rltv_y = y - mean_y
            min_ratio = 100000000
            for box_area_2 in norm_bboxes:
                x_2, y_2, w_2, h_2, area_2 = box_area_2
                if w_2 == 0 or h_2 ==0:
                    print(x_2,y_2,w_2,h_2,area_2)
                min_ratio = min(min_ratio, area / area_2)
            rltv_pos.append([x, y, w, h, rltv_x, rltv_y, min_ratio])
        return rltv_pos

    def norm_data(self, reg, keys):
        b, c, w, h = reg['faces'].shape
        mean = self.mean
        std = self.std
        mean = torch.cat([mean[0].expand(b, 1, w, h), mean[1].expand(
            b, 1, w, h), mean[2].expand(b, 1, w, h)], dim=1)
        std = torch.cat([std[0].expand(b, 1, w, h), std[1].expand(
            b, 1, w, h), std[2].expand(b, 1, w, h)], dim=1)
        for key in keys:
            reg[key] = reg[key].sub(mean).div(std)
    

    def get_idxs(self, lbls):
        imp_idxs = np.arange(len(lbls))[lbls==1]
        non_imp_idxs = np.arange(len(lbls))[lbls==0]
        assert len(imp_idxs) + len(non_imp_idxs) == len(lbls)
        return imp_idxs, non_imp_idxs

    def choice_bbox(self, imp_idxs, non_imp_idxs, bboxes):
        lbls = np.zeros(len(bboxes), dtype=np.int64)
        lbls[imp_idxs] = 1
        imps = imp_idxs
        if len(lbls) - len(imp_idxs) > self.cnfs["num_ins"] - len(imps):
            choice_idxs = np.random.choice(non_imp_idxs, self.cnfs["num_ins"] - len(imps), replace=False)
            choice_idxs = np.append(choice_idxs, imps)
        else:
            choice_idxs = np.append(non_imp_idxs, imps)
        np.random.shuffle(choice_idxs)
        return bboxes[choice_idxs], lbls[choice_idxs]

    def get_random_choice_idxs(self, imp_idx, node_num):
        idxs = [i for i in range(node_num) if i != imp_idx]
        choice_idxs = list(np.random.choice(
            idxs, self.num_ins-1, replace=False))
        choice_idxs.append(imp_idx)
        return list(np.random.permutation(choice_idxs))

    def choice_tensors(self, idxs, keys, reg):
        for key in keys:
            reg[key] = reg[key][idxs]

    def param_to_tensor(self, *cnfs):
        ret_cnfs = []
        for i, arg in enumerate(cnfs):
            if isinstance(arg, (Image.Image, np.ndarray)):
                ret_cnfs.append(self.to_tensor(arg))
            elif isinstance(arg, list):
                if isinstance(arg[0], (Image.Image, np.ndarray)):
                    tmp_ret = []
                    for tmp in arg:
                        tmp_ret.append(self.to_tensor(tmp))
                    ret_cnfs.append(torch.stack(tmp_ret, dim=0))
                elif isinstance(arg[0], list):
                    ret_cnfs.append(torch.Tensor(arg).float())
        return ret_cnfs

    def _padding(self, tensor, dtype):
        tensor_shape = list(tensor.shape)
        tensor_shape[0] = self.num_ins - tensor_shape[0]
        zeros = torch.zeros(tensor_shape, dtype=dtype)
        tensor = torch.cat([tensor, zeros], dim=0)
        return tensor

    def _zero_pad_all(self, keys, reg):
        for key in keys:
            reg[key] = self._padding(reg[key], reg[key].dtype)

    def _person_padding(self, tensor, choice_idxs):
        tensor = torch.cat([tensor, tensor[choice_idxs]], dim=0)
        return tensor

    def _person_pad_all(self, keys, reg, choice_idxs):
        for key in keys:
            reg[key] = self._person_padding(reg[key], choice_idxs)

    def init_reg(self, reg, **kcnfs):
        for key, val in kcnfs.items():
            reg[key] = val

    def get_face_ctx_coord_normed_bboxes(self, img, bboxes, data_name):
        width, height = img.size
        faces = []
        ctxs = []
        ctx_bboxes = []
        coords = []
        ctx_coords = []
        normed_bboxes = []
        normed_ctx_bboxes = []
        for bbox in bboxes:
            tmp_ctx_bbox = self.get_ctx_bbox(bbox, width, height, data_name)
            ctx_bboxes.append(tmp_ctx_bbox)
            tmp_face, tmp_coord = self.crop_box(bbox, img)
            if self.face_xfmr is not None:
                faces.append(self.face_xfmr(tmp_face))
            else:
                faces.append(tmp_face.resize((224, 224), Image.BILINEAR))
            coords.append(tmp_coord.resize((224, 224), Image.BILINEAR))
            tmp_ctx, tmp_ctx_coord = self.crop_box(tmp_ctx_bbox, img)
            if self.ctx_xfmr is not None:
                ctxs.append(self.ctx_xfmr(tmp_ctx))
            else:
                ctxs.append(tmp_ctx.resize((224, 224), Image.BILINEAR))
            ctx_coords.append(tmp_ctx_coord.resize((224, 224), Image.BILINEAR))
            normed_bboxes.append(self.norm_bbox(bbox, width, height))
            normed_ctx_bboxes.append(
                self.norm_bbox(tmp_ctx_bbox, width, height))
        return faces, ctxs, ctx_bboxes, coords, ctx_coords, normed_bboxes, normed_ctx_bboxes

    def get_ctx_bbox(self, bbox, width, height, data_name):
        x, y, w, h = bbox
        if data_name == 'MS' or data_name == 'EMS':
            ctx_x_min = int(max(0, (x - 5/2*w)))
            ctx_x_max = int(min(width, (x + 7/2*w)))
            ctx_y_min = int(max(0, (y - 3/2*h)))
            ctx_y_max = int(min(height, (y + 13/2*h)))
            ctx_w = ctx_x_max - ctx_x_min
            ctx_h = ctx_y_max - ctx_y_min
        elif data_name == 'NCAA' or data_name == 'ENCAA':
            ctx_x_min = int(max(0, (x - 0.5*w)))
            ctx_x_max = int(min(width, (x + 1.5*w)))
            ctx_y_min = int(max(0, (y - 0.5*h)))
            ctx_y_max = int(min(height, (y + 1.5*h)))
            ctx_w = ctx_x_max - ctx_x_min
            ctx_h = ctx_y_max - ctx_y_min
        return [ctx_x_min, ctx_y_min, ctx_w, ctx_h]

    def crop_box(self, bbox, img):
        width, height = img.size
        x, y, w, h = bbox
        box_img = img.crop((x, y, x+w, y+h))
        canvas = np.zeros((height, width), np.uint8)
        canvas[y:y+h, x:x+w] = 255
        coord = Image.fromarray(np.uint8(canvas))
        return box_img, coord

    def norm_bbox(self, bbox, width, height):
        x, y, w, h = bbox
        norm_x = x / width
        norm_y = y / height
        norm_w = w / width
        norm_h = h / height
        return [norm_x, norm_y, norm_w, norm_h]


class VIPDatasetNewTransforms(BaseDataset):
    def __init__(self, data, cnfs, is_train=False):
        super(VIPDatasetNewTransforms, self).__init__(
            data, cnfs, is_train=is_train)
        self.use_da = cnfs["data"]["data_augment"]["is_use"]
        self.cnfs = cnfs
        self.clr_j = T.ColorJitter(0.2, 0.2, 0.2, 0)
        self.gr_sc = T.RandomGrayscale(p=0.2)
        self.vip_rnd_hflip = VIPRandomHFlip(p=0.5)
        self.vip_rnd_rsz_crop = VIPRandomResizedCrop((224, 224), cnfs=cnfs)
        self.to_tensor = T.ToTensor()

    def get_augmented_data(self, img, bboxes, imp_idxs, non_imp_idxs):
        width, height = img.size
        if self.use_da[0]:
            img = self.clr_j(img)
        if self.use_da[1]:
            img = self.gr_sc(img)
        if self.use_da[2]:
            img, bboxes = self.vip_rnd_hflip(img, bboxes)
        if self.use_da[3]:
            img, faces, ctxs, bboxes, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes, lbls = self.vip_rnd_rsz_crop(
                img, bboxes, self.data_name, imp_idxs, non_imp_idxs)
        else:
            bboxes, lbls = self.choice_bbox(imp_idxs, non_imp_idxs, bboxes)
            faces, ctxs, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes = self.get_face_ctx_coord_normed_bboxes(
                img, bboxes, self.data_name)
            img = img.resize((224, 224), Image.BILINEAR)
        rltv_pos = self.get_rltv_pos(bboxes, width, height)
        return img, faces, ctxs, bboxes, ctx_bboxes, coords, ctx_coords, norm_bboxes, norm_ctx_bboxes, rltv_pos, lbls

