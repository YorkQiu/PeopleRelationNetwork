import sys
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as TF
import math

class VIPRandomHFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            width, height = img.size
            hflip_img = TF.hflip(img)
            hfilp_bboxes = self.hfilp_bboxes(bboxes, width)
            return hflip_img, hfilp_bboxes
        else:
            return img, bboxes

    def hfilp_bboxes(self, bboxes, width):
        ret_bboxes = []
        for bbox in bboxes:
            ret_bboxes.append(self.get_filp_bbox(bbox, width))
        return np.array(ret_bboxes, dtype=int)
    
    def get_filp_bbox(self, bbox, width):
        x, y, w, h = bbox
        c_x = x + w // 2
        hflip_c_x = width - c_x
        hflip_x = hflip_c_x - w // 2
        return [hflip_x, y, w, h]


class VIPRandomResizedCrop(object):
    def __init__(self, size, scale=(0.9, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, cnfs=None, is_single=False):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print('ERROR! scale or ratio should range in (min, max).')
            exit
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.cnfs = cnfs
        self.is_single = is_single

    @staticmethod
    def get_params(img, scale, ratio, imp_bboxs):
        width, height = img.size
        area = height * width
        min_x, max_x, min_y, max_y = width + 10000, -10000, height + 10000, -10000
        for bbox in imp_bboxs:
            x, y, w, h = bbox
            min_x, max_x, min_y, max_y = min(x, min_x), max(
                x+w, max_x), min(y, min_y), max(y+h, max_y)
        imp_x, imp_y, imp_w, imp_h = min_x, min_y, max_x - min_x, max_y - min_y
        c_x = imp_x + imp_w // 2
        c_y = imp_y + imp_h // 2
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            # still don't know why
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < imp_w or h < imp_h:
                continue
            if 0 < w <= width and 0 < h <= height:
                w_upr_bd, w_lwr_bd, h_upr_bd, h_lwr_bd = min(
                    imp_x, width - w), max(0, max_x - w), min(imp_y, height - h), max(0, max_y - h)
                if w_upr_bd >= w_lwr_bd and h_upr_bd >= h_lwr_bd:
                    i = random.randint(w_lwr_bd, w_upr_bd)
                    j = random.randint(h_lwr_bd, h_upr_bd)
                    return i, j, w, h

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
            if w < imp_w or h < imp_h:
                w = width
                h = height
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
            if w < imp_w or h < imp_h:
                w = width
                h = height
        else:  # whole image
            w = width
            h = height
        if (width - w) // 2 > imp_x:
            i = imp_x
        elif (width - w) // 2 + w < imp_x + imp_w:
            i = imp_x + imp_w - w
        else:
            i = (width - w) // 2
        if (height - h) // 2 > imp_y:
            j = imp_y
        elif (height - h) // 2 + h < imp_y + imp_h:
            j = imp_y + imp_h - h
        else:
            j = (height - h) // 2
        return i, j, w, h
    

    def __call__(self, img, bboxes, data_name, imp_idxs, non_imp_idxs, probs=None):
        if self.is_single:
            imp_bboxes = bboxes
        else:
            tmp_imp_idxs = imp_idxs
            imp_bboxes = bboxes[tmp_imp_idxs]
        i, j, w, h = self.get_params(img, self.scale, self.ratio, imp_bboxes)
        cropped_img = img.crop((i, j, i+w, j+h))
        if probs is not None:
            rdr_bboxes, lbls, probs = self.redirect_bboxes(bboxes, i, j, w, h, imp_idxs, non_imp_idxs, probs)
        else:
            rdr_bboxes, lbls = self.redirect_bboxes(bboxes, i, j, w, h, imp_idxs, non_imp_idxs)
        rsz_faces = []
        rsz_ctxs = []
        rsz_coords = []
        rsz_ctx_coords = []
        rdr_ctx_bboxes = []
        for bbox in rdr_bboxes:
            tmp_face, tmp_ctx, ctx_bbox, coord, ctx_coord = self.get_face_and_ctx_by_bbox(bbox, cropped_img, data_name)
            rdr_ctx_bboxes.append(ctx_bbox)
            rsz_faces.append(tmp_face.resize(self.size, self.interpolation))
            rsz_ctxs.append(tmp_ctx.resize(self.size, self.interpolation))
            rsz_coords.append(coord.resize(self.size, self.interpolation))
            rsz_ctx_coords.append(ctx_coord.resize(self.size, self.interpolation))
        rsz_img = cropped_img.resize(self.size, self.interpolation)
        norm_bboxes = []
        norm_ctx_bboxes = []
        cropped_width, cropped_height = cropped_img.size
        for bbox, ctx_bbox in zip(rdr_bboxes, rdr_ctx_bboxes):
            norm_bboxes.append(self.normalize_bbox(bbox, cropped_width, cropped_height))
            norm_ctx_bboxes.append(self.normalize_bbox(ctx_bbox, cropped_width, cropped_height))
        if probs is not None:
            return rsz_img, rsz_faces, rsz_ctxs, rdr_bboxes, rdr_ctx_bboxes, rsz_coords, rsz_ctx_coords, norm_bboxes, norm_ctx_bboxes, lbls, probs
        else:
            return rsz_img, rsz_faces, rsz_ctxs, rdr_bboxes, rdr_ctx_bboxes, rsz_coords, rsz_ctx_coords, norm_bboxes, norm_ctx_bboxes, lbls

    def redirect_bboxes(self, bboxes, l, t, width, height, imp_idxs, non_imp_idxs, probs=None):
        rdr_bboxes = []
        new_lbls = []
        if probs is not None:
            new_probs = []
        else:
            new_probs = None
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            r_x = x - l
            r_y = y - t
            if r_x < 0 or r_x + w > width or r_y < 0 or r_y + h > height:
                continue
            rdr_bboxes.append([r_x, r_y, w, h])
            if i in imp_idxs:
                new_lbls.append(1)
                if probs is not None:
                    new_probs.append(probs[i])
            elif i in non_imp_idxs:
                new_lbls.append(0)
                if probs is not None:
                    new_probs.append(probs[i])
            else:
                print("Error! not have corresponding label.")
                sys.exit(-1)
        rdr_bboxes = np.array(rdr_bboxes, dtype=int)
        new_lbls = np.array(new_lbls, dtype=np.int64)
        new_probs = np.array(new_probs, dtype=float)
        assert len(new_lbls) == len(rdr_bboxes)
        if probs is not None:
            rdr_bboxes, new_lbls, new_probs = self.choice_bbox(rdr_bboxes, new_lbls, new_probs)
            assert len(new_lbls) == len(rdr_bboxes)
            return rdr_bboxes, new_lbls, new_probs
        else:
            rdr_bboxes, new_lbls = self.choice_bbox(rdr_bboxes, new_lbls)
            assert len(new_lbls) == len(rdr_bboxes)
            return rdr_bboxes, new_lbls
    
    def choice_bbox(self, rdr_bboxes, new_lbls, new_probs=None):
        imp_idxs = np.arange(len(new_lbls))[new_lbls==1]
        non_imp_idxs = np.arange(len(new_lbls))[new_lbls==0]
        imps = imp_idxs
        if len(new_lbls) - len(imp_idxs) > self.cnfs["num_ins"] - len(imps):
            choice_idxs = np.random.choice(non_imp_idxs, self.cnfs["num_ins"] - len(imps), replace=False)
            choice_idxs = np.append(choice_idxs, imps)
        else:
            choice_idxs = np.append(non_imp_idxs, imps)
        np.random.shuffle(choice_idxs)
        if new_probs is not None:
            return rdr_bboxes[choice_idxs], new_lbls[choice_idxs], new_probs[choice_idxs]
        else:
            return rdr_bboxes[choice_idxs], new_lbls[choice_idxs]



    def get_face_and_ctx_by_bbox(self, bbox, img, data_name):
        width, height = img.size
        ctx_bbox = self.get_ctx_by_bbox(bbox, width, height, data_name)
        x, y, w, h = bbox
        face = img.crop((x, y, x+w, y+h))
        canvas = np.zeros((height, width), np.uint8)
        canvas[y:y+h, x:x+w] = 255
        coord = Image.fromarray(np.uint8(canvas))
        x, y, w, h = ctx_bbox
        ctx = img.crop((x, y, x+w, y+h))
        canvas = np.zeros((height, width), np.uint8)
        canvas[y:y+h, x:x+w] = 255
        ctx_coord = Image.fromarray(np.uint8(canvas))
        return face, ctx, ctx_bbox, coord, ctx_coord

    def get_ctx_by_bbox(self, bbox, width, height, data_name):
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

    def normalize_bbox(self, bbox, width, height):
        x, y, w, h = bbox
        norm_x = x / width
        norm_y = y / height
        norm_w = w / width
        norm_h = h / height
        return [norm_x, norm_y, norm_w, norm_h]
