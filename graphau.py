# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class DirectAU(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DirectAU, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.gamma = config['gamma']
        self.decay_weight = config['decay_weight']
        self.encoder_name = config['encoder']

        # define layers and loss
        if self.encoder_name == 'MF':
            self.encoder = MFEncoder(self.n_users, self.n_items, self.embedding_size)
        elif self.encoder_name == 'LightGCN':
            self.n_layers = config['n_layers']
            self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
            self.norm_adj = self.get_norm_adj_mat().to(self.device)
            self.encoder = LGCNEncoder(self.n_users, self.n_items, self.embedding_size, self.norm_adj, self.n_layers)
        else:
            raise ValueError('Non-implemented Encoder.')

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_all_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, user, item):
        user_e, item_e, all_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1), all_e

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None or self.restore_all_e is not None:
            self.restore_user_e, self.restore_item_e, self.restore_all_e = None, None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e, all_e = self.forward(user, item)

        align = [self.alignment(user_e, item_e)]
        uniform = [(self.uniformity(user_e) + self.uniformity(item_e)) / 2]

        for i in range(2, self.encoder.n_layers + 1):
            user_e_agg = all_e[0][i][user]
            item_e_agg = all_e[1][i][item]
            align.append((self.alignment(user_e, item_e_agg) + self.alignment(user_e_agg, item_e)) / 2)
        align = torch.mean(self.decay_weight * torch.stack(align))
        # align = align[-1]

        loss = align + self.gamma * uniform[0]
        return loss

        align = self.alignment(user_e, item_e)
        user_uniform, user_std = self.uniformity(user_e)
        item_uniform, item_std = self.uniformity(item_e)
        uniform = self.gamma * (user_uniform + item_uniform) / 2
        # print(f'align:{align}\tuser_uniform:{user_uniform.detach().cpu().item()}\titem_uniform:{item_uniform.detach().cpu().item()}\tuser_std:{user_std}\titem_std:{item_std}\tgamma:{self.gamma}')
        return align + uniform

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.encoder_name == 'LightGCN':
            if self.restore_user_e is None or self.restore_item_e is None or self.restore_all_e is None:
                self.restore_user_e, self.restore_item_e, self.restore_all_e = self.encoder.get_all_embeddings()
            user_e = self.restore_user_e[user]
            all_item_e = self.restore_item_e
        else:
            user_e = self.encoder.user_embedding(user)
            all_item_e = self.encoder.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

class MFEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(MFEncoder, self).__init__()
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.item_embedding = nn.Embedding(item_num, emb_size)

    def forward(self, user_id, item_id):
        u_embed = self.user_embedding(user_id)
        i_embed = self.item_embedding(item_id)
        return u_embed, i_embed

    def get_all_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings, item_embeddings


class LGCNEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size, norm_adj, n_layers=3):
        super(LGCNEncoder, self).__init__()
        self.n_users = user_num
        self.n_items = item_num
        self.n_layers = n_layers
        self.norm_adj = norm_adj

        self.user_embedding = torch.nn.Embedding(user_num, emb_size)
        self.item_embedding = torch.nn.Embedding(item_num, emb_size)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        # 保存每一层的embedding
        user_each_layer_embeddings, item_each_layer_embeddings = [], []
        for layer_idx in range(self.n_layers):
            user_layer_embeddings, item_layer_embeddings = torch.split(embeddings_list[layer_idx], [self.n_users, self.n_items])
            user_each_layer_embeddings.append(user_layer_embeddings)
            item_each_layer_embeddings.append(item_layer_embeddings)
        return user_all_embeddings, item_all_embeddings, (user_each_layer_embeddings, item_each_layer_embeddings)

    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings, all_layer_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed, all_layer_embeddings