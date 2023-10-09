import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, params):
        super(ScaledDotProductAttention, self).__init__()
        self.x_in_dim = params["x_in_dim"]
        self.y_in_dim = params["y_in_dim"]
        self.out_dim = params["out_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.W_Q = nn.Linear(self.x_in_dim, self.hidden_dim)
        self.W_K = nn.Linear(self.y_in_dim, self.hidden_dim)
        self.W_V = nn.Linear(self.y_in_dim, self.out_dim)

    def forward(self, x, y, need_weight=False):
        qry = self.W_Q(x) # B' x hidden_dim
        key = self.W_K(y) # B x hidden_dim
        val = self.W_V(y) # B x out_dim
        S = torch.matmul(qry, key.transpose(0,1))
        S = S / math.sqrt(self.hidden_dim)
        S = torch.softmax(S, dim=1) # B' x B
        if need_weight:
            return torch.matmul(S, val), S
        else:
            return torch.matmul(S, val)


class InterPersonAttention(nn.Module):
    def __init__(self, params):
        super(InterPersonAttention, self).__init__()
        self.in_dim = params['in_dim']
        self.hidden_dim = params['hidden_dim']
        self.out_dim = params['out_dim']
        self.sub_param = params['sub_param']
        self.edge_extr = nn.Linear(self.in_dim + 6, self.hidden_dim)
        self.out_fc = nn.Linear(self.hidden_dim + self.in_dim, self.out_dim)
        self.attention = ScaledDotProductAttention(self.sub_param)
    
    def forward(self, app_feats, edges, need_attn=False):
        node_num = app_feats.shape[0]
        final_feats = []
        attn_weights = []
        for i in range(node_num):
            tmp_edge_feats = []
            for j in range(node_num):
                if i == j:
                    continue
                tmp_edge_feats.append(torch.relu(self.edge_extr(torch.cat([edges[i,j], app_feats[j]], dim=0))))
            tmp_edge_feats = torch.stack(tmp_edge_feats, dim=0)
            if need_attn:
                edge_feats, attn_weight = self.attention(app_feats[i].view(1, -1), tmp_edge_feats, True)
                attn_weights.append(attn_weight)
            else:
                edge_feats = self.attention(app_feats[i].view(1, -1), tmp_edge_feats)
            final_feats.append(torch.relu(self.out_fc(torch.cat([app_feats[i].view(1, -1), edge_feats], dim=1))))
        final_feats = torch.cat(final_feats, dim=0)
        if need_attn:
            attn_weights = torch.cat(attn_weights, dim=0)
            assert attn_weights.shape[0] == attn_weights.shape[1]+1, print(attn_weights.shape)
            return final_feats, attn_weights
        else:
            return final_feats


