import torchvision
import torch
import numpy as np
import torch.nn as nn

from packages.models.blocks.extractors import CoordinateExtractor, PersonFeatureExtractor
from packages.models.blocks.classifiers import FeatureClassifier
from packages.models.blocks.attention import InterPersonAttention


class PRN(nn.Module):
    def __init__(self, cnfs):
        super(PRN, self).__init__()
        # global extractor, output shape: [batch_size, 2048, 7, 7]
        self.cnfs = cnfs
        self.param = cnfs["model"]["param"]
        self.in_dim = self.param["in_dim"]
        self.hidden_dim = self.param["hidden_dim"]
        self.feat_dim = self.param["feat_dim"]
        self.app_type = self.param["app_type"]
        self.attn_param = self.param["attn_param"]
        self.glb_extr = torchvision.models.resnet50(pretrained=True)
        self.glb_extr = nn.Sequential(
            *list(self.glb_extr.children())[:-2])

        # context extractor, output shape: [b, 2048, 7, 7]
        self.ctx_extr = torchvision.models.resnet50(pretrained=True)
        self.ctx_extr = nn.Sequential(
            *list(self.ctx_extr.children())[:-2])
        self.ctx_extr.add_module('1x1conv', nn.Conv2d(2048, self.feat_dim, 1, stride=1, bias=False))
        self.ctx_extr.add_module('bn1', nn.BatchNorm2d(self.feat_dim))
        self.ctx_extr.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))

        # local extractor, output shape: [b, 2048, 7, 7]
        self.lcl_extr = torchvision.models.resnet50(pretrained=True)
        self.lcl_extr = nn.Sequential(
            *list(self.lcl_extr.children())[:-2])
        self.lcl_extr.add_module('1x1conv', nn.Conv2d(2048, self.feat_dim, 1, stride=1, bias=False))
        self.lcl_extr.add_module('bn1', nn.BatchNorm2d(self.feat_dim))
        self.lcl_extr.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))

        # coordinate feature extractor, output shape: [b, 256, 7, 7] input shape [b, 1, 224, 224]
        self.coord_feats_extr = CoordinateExtractor()
        self.pers_feats_extr = PersonFeatureExtractor(2048+256, self.feat_dim)
        self.attn = InterPersonAttention(self.attn_param)

        # classifier
        self.clf = FeatureClassifier(self.in_dim, 2, self.hidden_dim)
        self.face_clf = FeatureClassifier(self.feat_dim, 2, hidden_dim=self.feat_dim // 2)
        self.ctx_clf = FeatureClassifier(self.feat_dim, 2, hidden_dim=self.feat_dim // 2)
        self.fbg_clf = FeatureClassifier(self.feat_dim, 2, hidden_dim=self.feat_dim // 2)
        self.relat_clf = FeatureClassifier(self.feat_dim, 2, hidden_dim=self.feat_dim // 2)
        self.data_keys = ['imgs', 'faces', 'ctxs', 'coords']

    def forward(self, data, cnfs):
        # debug only parameter
        # coordinates is a heatmap shape: [b, 224, 224] add one dimension to [b, 1, 224, 224]
        imgs, faces, face_ctxs, coords = [
            data[key].cuda() for key in self.data_keys]
        bboxes = data['bboxes']
        masks = data['masks']
        edges = self.get_edges(bboxes, masks)
        face_feats = self.lcl_extr(faces)
        face_feats = face_feats.view(-1, self.feat_dim)
        # get face contextrual features
        face_ctx_feats = self.ctx_extr(face_ctxs)
        face_ctx_feats = face_ctx_feats.view(-1, self.feat_dim)
        # get coordinate featrures
        if self.app_type == "face":
            app_feats = face_feats
        elif self.app_type == 'ctx':
            app_feats = face_ctx_feats
        elif self.app_type == 'both':
            app_feats = torch.cat([face_feats, face_ctx_feats], dim=1)
        if self.training:
            inter_pers_relats = []
            for i, edge in enumerate(edges):
                tmp_masks = masks[i*cnfs["num_ins"]: (i+1)*cnfs["num_ins"]]
                tmp_app_feats = app_feats[i*cnfs["num_ins"]: (i+1)*cnfs["num_ins"]]
                zero_idxs = torch.nonzero(tmp_masks == 0)
                if len(zero_idxs) > 0:
                    node_num = int(zero_idxs[0].data.cpu())
                else:
                    node_num = cnfs["num_ins"]
                if node_num == 1:
                    inter_pers_relats.append(torch.zeros(cnfs["num_ins"], self.feat_dim).cuda())
                    continue
                tmp_app_feats = tmp_app_feats[:node_num]
                tmp_inter_relation_feats = self.attn(tmp_app_feats, edge)
                tmp_inter_relation_feats = torch.cat([tmp_inter_relation_feats, torch.zeros(
                    cnfs["num_ins"] - node_num, tmp_inter_relation_feats.shape[-1]).cuda()], dim=0)
                inter_pers_relats.append(tmp_inter_relation_feats)
            inter_pers_relats = torch.cat(inter_pers_relats, dim=0)
        else:
            if app_feats.shape[0] == 1:
                inter_pers_relats = torch.zeros(1, self.feat_dim).cuda()
            else:
                inter_pers_relats = self.attn(app_feats, edges)
        coord_feats = self.coord_feats_extr(coords)
        img_feats = self.glb_extr(imgs)
        aggr_feats = torch.cat([img_feats, coord_feats], 1)
        pers_feats = self.pers_feats_extr(aggr_feats)  # [24, 512, 1, 1]
        pers_feats = pers_feats.view(-1, self.feat_dim)
        final_feats = torch.cat([inter_pers_relats ,pers_feats, face_feats, face_ctx_feats], dim=1)
        logits = self.clf(final_feats)  # [24, 2]
        if self.training:
            if cnfs['lambda']['face'] != 0:
                face_logits = self.face_clf(face_feats)
            else:
                face_logits = None
            if cnfs['lambda']['ctx'] != 0:
                ctx_logits = self.ctx_clf(face_ctx_feats)
            else:
                ctx_logits = None
            if cnfs['lambda']['fbg'] != 0:
                fbg_logits = self.fbg_clf(pers_feats)
            else:
                fbg_logits = None
            if cnfs['lambda']['relat'] != 0:
                relats_logits = self.relat_clf(inter_pers_relats)
            else:
                relats_logits = None
            return logits, face_logits, ctx_logits, fbg_logits, relats_logits
        else:
            return logits
    
    def get_edges(self, bboxes, masks):
        def get_center(bboxes):
            bboxes = bboxes.data.cpu().numpy()
            x, y, w, h = bboxes
            return (x+w) // 2, (y+h) // 2
        
        def get_edge(dx, dy):
            abs_dx, abs_dy, abs_dx_dy = np.abs(dx), np.abs(dy), np.abs(dx+dy)
            sqrt_dx_dy = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
            arctan, arctan2 = np.arctan(dy / (dx+1e-6)), np.arctan2(dy, (dx+1e-6))
            return torch.Tensor([abs_dx, abs_dy, abs_dx_dy, sqrt_dx_dy, arctan, arctan2])

        if self.training:
            batch_size = bboxes.shape[0] // self.cnfs["num_ins"]
            num_ins = self.cnfs['num_ins']
            edges = []
            for b in range(batch_size):
                tmp_edges = torch.zeros(num_ins, num_ins, 6)
                tmp_bboxes = bboxes[b*num_ins: (b+1)*num_ins]
                tmp_masks = masks[b*num_ins: (b+1)*num_ins]
                zero_idxs = torch.nonzero(tmp_masks == 0)
                if len(zero_idxs) > 0:
                    node_num = int(zero_idxs[0].data.cpu())
                else:
                    node_num = num_ins
                tmp_bboxes = tmp_bboxes[:node_num]
                for i in range(node_num):
                    ctr_x1, ctr_y1 = get_center(tmp_bboxes[i])
                    for j in range(node_num):
                        if i == j:
                            continue
                        ctr_x2, ctr_y2 = get_center(tmp_bboxes[j])
                        tmp_edges[i, j] = get_edge(ctr_x2 - ctr_x1, ctr_y2 - ctr_y1)
                edges.append(tmp_edges.cuda())
            return edges
        else:
            num_ins = bboxes.shape[0]
            edges = torch.zeros(num_ins, num_ins, 6)
            for i in range(num_ins):
                ctr_x1, ctr_y1 = get_center(bboxes[i])
                for j in range(num_ins):
                    if i == j:
                        continue
                    ctr_x2, ctr_y2 = get_center(bboxes[j])
                    edges[i, j] = get_edge(ctr_x2 - ctr_x1, ctr_y2 - ctr_y1)
            return edges.cuda()


