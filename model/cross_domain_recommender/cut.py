# -*- coding: utf-8 -*-

r"""
CUT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
import pickle as pkl


from recbole.model.init import xavier_uniform_initialization, xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import EmbLoss, RegLoss, BPRLoss
from recbole.model.general_recommender.mf import MF
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.general_recommender.simplex import SimpleX

from CUT.model.crossdomain_recommender import CrossDomainRecommender

class CCLLoss(nn.Module):
    def __init__(self, margin=0.5, negative_weight=10):
        super(CCLLoss, self).__init__()
        self.margin = margin
        self.negative_weight=negative_weight

    def forward(self, pos_cos, neg_cos):
        # CCL loss
        pos_loss = torch.relu(1 - pos_cos)
        neg_loss = torch.relu(neg_cos - self.margin)
        neg_loss = neg_loss.mean(1, keepdim=True) * self.negative_weight
        CCL_loss = (pos_loss + neg_loss).mean()
        return CCL_loss
class CUT(CrossDomainRecommender):
    r""" Contrastive User Embedding Transformation
    """

    #input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CUT, self).__init__(config, dataset)
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        
        self.tau = 0.1

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.alpha = config['alpha']
        self.lamda = config['lambda']
        self.gamma = config['gamma']
        self.delta = config['delta']
        self.transform_weight = config['transform_weight']
        self.loss_term = config['loss_term']
        self.cosine_threshold = config['cosine_threshold']
        self.single_model_config = config['single_model']
        self.train_neg_sample_num = config['train_neg_sample_args']['sample_num']
        self.joint_learning = config['joint_learning']
        self.user_transform = config['user_transform']
        self.additional_user_samp = config['additional_user_samp']
        self.raw_embedding_loss = config['raw_embedding_loss']
        self.checkpoint_dir = config["checkpoint_dir"]
        self.sim_emb_name = config['sim_emb_name']
        self.phase_count = 0

        # define layers and loss
        if self.single_model_config['name'] in ['MF','LightGCN']:
            self.user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
            self.item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
            self.target_user_embedding = nn.Embedding(self.target_num_users, self.embedding_size)
            self.target_item_embedding = nn.Embedding(self.target_num_items, self.embedding_size)

            if self.single_model_config['name'] == 'MF':
                pass

            elif self.single_model_config['name'] == 'LightGCN':
                # load dataset info
                self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
                self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)

                # load parameters info
                self.latent_dim = config["embedding_size"]  # int type:the embedding size of lightGCN
                self.n_layers = self.single_model_config["n_layers"]  # int type:the layer num of lightGCN

                # storage variables for full sort evaluation acceleration
                self.restore_user_e = None
                self.restore_item_e = None

                # generate intermediate data
                self.target_norm_adj_matrix = self.get_norm_adj_mat_t(self.target_interaction_matrix).to(self.device)
                self.all_norm_adj_matrix = self.get_norm_adj_mat_a(self.source_interaction_matrix,self.target_interaction_matrix).to(self.device)

                self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        elif self.single_model_config['name'] == 'SimpleX':
            #model config
            self.single_model_config['embedding_size'] = config['embedding_size']
            self.single_model_config['train_neg_sample_args'] = config['train_neg_sample_args']
            self.single_model_config["device"] = config["device"]

            #data config
            self.single_model_config["USER_ID_FIELD"] = 'target_'+config['target_domain']["USER_ID_FIELD"]
            self.single_model_config["ITEM_ID_FIELD"] = 'target_'+config['target_domain']["ITEM_ID_FIELD"]
            self.single_model_config["NEG_PREFIX"] = config['target_domain']["NEG_PREFIX"]

            self.target_model = SimpleX(self.single_model_config,dataset.target_domain_dataset)

            self.single_model_config["USER_ID_FIELD"] = 'total_'+config['target_domain']["USER_ID_FIELD"]
            self.single_model_config["ITEM_ID_FIELD"] = 'total_'+config['target_domain']["ITEM_ID_FIELD"]
            self.both_model = SimpleX(self.single_model_config,dataset)

            self.restore_user_e = None
        
        if config['user_transform']:
            self.user_transform_matrix_r = nn.Parameter(torch.zeros(self.embedding_size, self.embedding_size))#transform source user to target user embedding
        else:
            self.user_transform_matrix_r = torch.zeros(self.embedding_size, self.embedding_size).to(self.device)#donot transform source user to target user embedding
        self.user_transform_matrix = torch.eye(self.embedding_size).to(self.device)#transform source user to target user embedding
        self.sigmoid = nn.Sigmoid()
        if config['loss_type'] == 'CE':
            self.input_type = InputType.POINTWISE
            self.loss = nn.BCEWithLogitsLoss()
        elif config['loss_type'] == 'BPR':
            self.input_type = InputType.PAIRWISE
            if self.single_model_config['name'] == 'SimpleX':
                self.loss = CCLLoss(margin=self.single_model_config['margin'], negative_weight=self.single_model_config['negative_weight'])
            else:
                self.loss = BPRLoss()            

        self.target_reg_loss = EmbLoss()
        self.source_reg_loss = EmbLoss()
        self.param_reg_loss = RegLoss()

        # parameters initialization
        if self.single_model_config['name'] != 'SimpleX':
            self.apply(xavier_normal_initialization)
        #torch.nn.init.xavier_normal_(self.user_transform_matrix_r) #This will derade model performance comparing with constant initialization
        torch.nn.init.constant_(self.user_transform_matrix_r, 0)
        #torch.nn.init.constant_(self.target_user_embedding.weight, 0.001)
        #torch.nn.init.constant_(self.target_item_embedding.weight, 0.001)
    def get_norm_adj_mat_t(self, interaction_matrix):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.target_num_users + self.target_num_items, self.target_num_users + self.target_num_items), dtype=np.float32
        )
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.target_num_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.target_num_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
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
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL
    
    def get_norm_adj_mat_a(self, interaction_matrix_s, interaction_matrix_t):
        # build adj matrix
        A = sp.dok_matrix(
            (self.total_num_users + self.total_num_items, self.total_num_users + self.total_num_items), dtype=np.float32
        )
        inter_S = interaction_matrix_s
        inter_S_t = interaction_matrix_s.transpose()
        inter_T = interaction_matrix_t
        inter_T_t = interaction_matrix_t.transpose()
        data_dict = dict(
            zip(zip(inter_S.row, inter_S.col + self.total_num_users), [1] * inter_S.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_S_t.row + self.total_num_users, inter_S_t.col),
                    [1] * inter_S_t.nnz,
                )
            )
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_T.row, inter_T.col + self.total_num_users),
                    [1] * inter_T.nnz,
                )
            )
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_T_t.row + self.total_num_users, inter_T_t.col),
                    [1] * inter_T_t.nnz,
                )
            )
        )
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
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL
    
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        if self.phase == 'TARGET':
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
            #user_embeddings = torch.cat([user_embeddings, torch.zeros((self.total_num_users-self.target_num_users,self.embedding_size),device=self.device)], dim=0)
            #item_embeddings = torch.cat([item_embeddings, torch.zeros((self.total_num_items-self.target_num_items,self.embedding_size),device=self.device)], dim=0)
        else:
            user_embeddings = self.user_embedding.weight
            if self.raw_embedding_loss:
                user_embeddings = self.transform_raw_user(torch.arange(user_embeddings.shape[0],device=self.device),user_embeddings)
            item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def set_phase(self, phase):
        self.phase = phase
        self.phase_count += 1
        sim_emb_path = os.path.join(self.checkpoint_dir, self.sim_emb_name)
        if phase == 'TARGET':
            if self.single_model_config['name'] == 'SimpleX':
                for param in self.both_model.parameters():
                    param.requires_grad = False
                for param in self.target_model.parameters():
                    param.requires_grad = True
            else:
                self.target_user_embedding.requires_grad = True
                self.target_item_embedding.requires_grad = True
                self.user_embedding.requires_grad = False
                self.item_embedding.requires_grad = False
            if self.user_transform:
                self.user_transform_matrix_r.requires_grad = False
            if self.sim_emb_name != None:
                if not os.path.exists(sim_emb_path):
                    self.skip_target = 0
                else:
                    self.skip_target = 1

        elif phase == 'BOTH':
            if self.phase_count%2 == 1:
                if self.phase_count == 3: #First both phase, freeze target model parameter, derive target user embedding ground truth
                    if self.skip_target == 0:
                        if self.single_model_config['name'] != 'SimpleX':
                            self.target_user_embedding_normed = self.target_user_embedding.weight - torch.mean(self.target_user_embedding.weight)
                        else:
                            self.target_user_embedding_normed = self.target_model.user_emb.weight - torch.mean(self.target_model.user_emb.weight)
                        self.target_user_embedding_normed_no_grad = self.target_user_embedding_normed.detach()
                        with open(sim_emb_path,'wb') as f:
                            pkl.dump(self.target_user_embedding_normed_no_grad.cpu(),f)
                    else:
                        with open(sim_emb_path,'rb') as f:
                            self.target_user_embedding_normed_no_grad = pkl.load(f)
                            self.target_user_embedding_normed_no_grad = self.target_user_embedding_normed_no_grad.to(self.device)
                    if self.single_model_config['name'] != 'SimpleX':
                        self.user_embedding.requires_grad = True
                        self.item_embedding.requires_grad = True
                        self.target_user_embedding.requires_grad = False
                        self.target_item_embedding.requires_grad = False
                    else:
                        for param in self.both_model.parameters():
                            param.requires_grad = True
                        for param in self.target_model.parameters():
                            param.requires_grad = False
                    if self.joint_learning == True:
                        if self.user_transform:
                            self.user_transform_matrix_r.requires_grad = True
                else:
                    if self.single_model_config['name'] != 'SimpleX':
                        self.user_embedding.requires_grad = True
                        self.item_embedding.requires_grad = True
                    else:
                        for param in self.both_model.parameters():
                            param.requires_grad = True
                if self.joint_learning == False:
                    if self.user_transform:
                        self.user_transform_matrix_r.requires_grad = False
            elif self.phase_count%2 == 0:
                if self.single_model_config['name'] != 'SimpleX':
                    self.user_embedding.requires_grad = False
                    self.item_embedding.requires_grad = False
                else:
                    for param in self.both_model.parameters():
                        param.requires_grad = False
                if self.joint_learning == False:
                    if self.user_transform:
                        self.user_transform_matrix_r.requires_grad = True
        elif phase == 'OVERLAP':
            if self.single_model_config['name'] != 'SimpleX':
                self.user_embedding.requires_grad = False
                self.item_embedding.requires_grad = False
                self.target_user_embedding.requires_grad = False
                self.target_item_embedding.requires_grad = False
            else:
                for param in self.both_model.parameters():
                    param.requires_grad = False
                for param in self.target_model.parameters():
                    param.requires_grad = False
            if self.user_transform:
                self.user_transform_matrix_r.requires_grad = False
            pass


    def get_user_embedding(self, user, train, raw=0):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        model_name =  self.single_model_config['name']
        if model_name == 'MF':
            if self.phase != 'TARGET':
                return self.user_embedding(user)
            else:
                return self.target_user_embedding(user)
        elif model_name == 'LightGCN':
            if raw == 1:
                if self.phase != 'TARGET':
                    return self.user_embedding(user)
                else:
                    return self.target_user_embedding(user)
            if train:
                return self.user_gcn_embeddings[user]
            else:
                if self.restore_user_e is None or self.restore_item_e is None:
                    self.restore_user_e, self.restore_item_e = self.gcnforward()
                return self.restore_user_e[user]
        elif model_name == 'SimpleX':
            if self.phase != 'TARGET':
                simplex_model = self.both_model
            else:
                simplex_model = self.target_model
            if raw == 1:
                return simplex_model.user_emb(user)
            history_item = simplex_model.history_item_id[user]
            history_len = simplex_model.history_item_len[user]
            user_e = simplex_model.user_emb(user) # [user_num, embedding_size]
            if self.raw_embedding_loss:
                user_e = self.transform_raw_user(user,user_e)
            history_item_e = simplex_model.item_emb(history_item) # [user_num, max_history_len, embedding_size]
            UI_aggregation_e = simplex_model.get_UI_aggregation(user_e, history_item_e, history_len) # [user_num, embedding_size]
            UI_aggregation_e = simplex_model.dropout(UI_aggregation_e)
            return UI_aggregation_e

            
    def get_item_embedding(self, item, train, raw=0):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        model_name =  self.single_model_config['name']
        if model_name == 'MF':
            if self.phase != 'TARGET':
                return self.item_embedding(item)
            else:
                return self.target_item_embedding(item)
        elif model_name == 'LightGCN':
            if raw == 1:
                if self.phase != 'TARGET':
                    return self.item_embedding(item)
                else:
                    return self.target_item_embedding(item)
            if train:
                return self.item_gcn_embeddings[item]
            else:
                if self.restore_user_e is None or self.restore_item_e is None:
                    self.restore_user_e, self.restore_item_e = self.gcnforward()
                return self.restore_item_e[item]
        elif model_name == 'SimpleX':
            if self.phase != 'TARGET':
                simplex_model = self.both_model
            else:
                simplex_model = self.target_model
            return simplex_model.item_emb(item)
            

    def full_sort_get_all_item_weight(self):
        model_name =  self.single_model_config['name']
        if model_name == 'MF':
            if self.phase != 'TARGET':
                return self.item_embedding.weight
            else:
                return self.target_item_embedding.weight
        elif model_name == 'LightGCN':
            if self.restore_user_e is None or self.restore_item_e is None:
                self.restore_user_e, self.restore_item_e = self.gcnforward()
            return self.restore_item_e
        elif model_name == 'SimpleX':
            if self.phase != 'TARGET':
                simplex_model = self.both_model
            else:
                simplex_model = self.target_model
            return simplex_model.item_emb.weight


    def source_forward(self, user, item, train, neg=True):
        if self.single_model_config['name'] == 'SimpleX':
            user_number = int(len(user) / self.both_model.neg_seq_len)
            if neg:
                # get the sequence of neg items
                neg_item_seq = item.reshape((self.both_model.neg_seq_len, -1))
                item = neg_item_seq.T
                user_e = self.restore_user_e
                self.restore_user_e = None
            else:
                item = item[0:user_number].unsqueeze(1)
                # user's id
                user = user[0:user_number]
                user_e = self.get_user_embedding(user, train)
                if self.restore_user_e != None:
                    raise ValueError('overwrite restore_user_e!')
                self.restore_user_e = user_e
            item_e = self.get_item_embedding(item, train)
            return self.both_model.get_cos(user_e, item_e)
        else:
            user_e = self.get_user_embedding(user, train)
            item_e = self.get_item_embedding(item, train)
            return torch.mul(user_e, item_e).sum(dim=1)

    def target_forward(self, user, item, train, neg=True):
        if self.single_model_config['name'] == 'SimpleX':
            user_number = int(len(user) / self.both_model.neg_seq_len)
            # user's id
            user = user[0:user_number]
            if neg:
                # get the sequence of neg items
                neg_item_seq = item.reshape((self.both_model.neg_seq_len, -1))
                item = neg_item_seq.T
                user_e = self.restore_user_e
                self.restore_user_e = None
            else:
                item = item[0:user_number].unsqueeze(1)
                user_e = self.get_user_embedding(user, train)
                if self.restore_user_e != None:
                    raise ValueError('overwrite restore_user_e!')
                self.restore_user_e = user_e
            t_user_e = self.transform_user(user, user_e)
            item_e = self.get_item_embedding(item, train)
            return self.both_model.get_cos(t_user_e, item_e)
        else:
            user_e = self.get_user_embedding(user, train)
            t_user_e = self.transform_user(user, user_e)
            item_e = self.get_item_embedding(item, train)
            return torch.mul(t_user_e, item_e).sum(dim=1)

    def target_train_forward(self, user, item, neg=True):
        if self.single_model_config['name'] == 'SimpleX':
            user_number = int(len(user) / self.target_model.neg_seq_len)
            if neg:
                # get the sequence of neg items
                neg_item_seq = item.reshape((self.target_model.neg_seq_len, -1))
                item = neg_item_seq.T
                user_e = self.restore_user_e
                self.restore_user_e = None
            else:
                item = item[0:user_number].unsqueeze(1)
                # user's id
                user = user[0:user_number]
                user_e = self.get_user_embedding(user, 1)
                if self.restore_user_e != None:
                    raise ValueError('overwrite restore_user_e!')
                self.restore_user_e = user_e
            item_e = self.get_item_embedding(item, 1)
            return self.target_model.get_cos(user_e, item_e)
        else:
            user_e = self.get_user_embedding(user, 1)
            item_e = self.get_item_embedding(item, 1)
            return torch.mul(user_e, item_e).sum(dim=1)
    
    def gcnforward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        norm_adj_matrix = self.all_norm_adj_matrix if self.phase != 'TARGET' else self.target_norm_adj_matrix

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_num = self.total_num_users if self.phase != 'TARGET' else self.target_num_users
        item_num = self.total_num_items if self.phase != 'TARGET' else self.target_num_items
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [user_num, item_num]
        )
        return user_all_embeddings, item_all_embeddings

    def transform_user(self, user, user_e):#
        if not self.raw_embedding_loss:
            if user != None:
                target_sign = (user<self.target_num_users+1).unsqueeze(1).float()
                target_users = user_e * target_sign
                nontarget_users = user_e * (1-target_sign)
                transformed_target_users = torch.mm(target_users,self.user_transform_matrix+self.transform_weight*self.user_transform_matrix_r)
                return nontarget_users + transformed_target_users
            else:
                return torch.mm(user_e,self.user_transform_matrix+self.transform_weight*self.user_transform_matrix_r)
        else:
            return user_e
        
    def transform_raw_user(self, user, user_e):#
        if self.raw_embedding_loss:
            if user != None:
                target_sign = (user<self.target_num_users+1).unsqueeze(1).float()
                target_users = user_e * target_sign
                nontarget_users = user_e * (1-target_sign)
                transformed_target_users = torch.mm(target_users,self.user_transform_matrix+self.transform_weight*self.user_transform_matrix_r)
                return nontarget_users + transformed_target_users
            else:
                return torch.mm(user_e,self.user_transform_matrix+self.transform_weight*self.user_transform_matrix_r)
        else:
            return user_e

    def calculate_loss(self, interaction):
        if self.single_model_config['name'] == 'LightGCN':
            if self.restore_user_e is not None or self.restore_item_e is not None:
                self.restore_user_e, self.restore_item_e = None, None
            self.user_gcn_embeddings, self.item_gcn_embeddings = self.gcnforward()
        elif self.single_model_config['name'] == 'SimpleX':
            if self.restore_user_e is not None:
                self.restore_user_e = None
            if self.phase == 'BOTH':
                self.target_model.eval()
                if self.phase_count == 4:
                    self.both_model.eval()

        user_id = [None,None]#source, target
        item_id = [None,None]#source, target
        label = [None,None]#source, target
        predict = [0,0]
        loss_all = 0
        if self.input_type == InputType.PAIRWISE:
            neg_item_id = [None,None]
            neg_predict = [0,0]
        if self.phase != 'SOURCE':
            if self.input_type == InputType.PAIRWISE:
                neg_item_id[1] = interaction[self.TARGET_NEG_ITEM_ID]
            user_id[1] = interaction[self.TARGET_USER_ID]
            item_id[1] = interaction[self.TARGET_ITEM_ID]
            label[1] = interaction[self.TARGET_LABEL]
            #return self.target_model.calculate_loss(interaction)#change

        if self.phase != 'TARGET':
            if self.input_type == InputType.PAIRWISE:
                neg_item_id[0] = interaction[self.SOURCE_NEG_ITEM_ID]
            user_id[0] = interaction[self.SOURCE_USER_ID]
            item_id[0] = interaction[self.SOURCE_ITEM_ID]
            label[0] = interaction[self.SOURCE_LABEL]

        if user_id[0] != None:
            predict[0] = self.source_forward(user_id[0], item_id[0], 1, neg=False)
            if self.input_type == InputType.POINTWISE:
                loss_s = self.loss(predict[0], label[0]) + \
                    self.lamda * self.source_reg_loss(self.get_user_embedding(user_id[0],1,raw=1),
                                                    self.get_item_embedding(item_id[0],1,raw=1))
            elif self.input_type == InputType.PAIRWISE:
                neg_predict[0] = self.source_forward(user_id[0], neg_item_id[0], 1, neg=True)
                loss_s = self.loss(predict[0], neg_predict[0]) + \
                    self.lamda * self.source_reg_loss(self.get_user_embedding(user_id[0],1,raw=1),
                                                    self.get_item_embedding(item_id[0],1,raw=1),
                                                    self.get_item_embedding(neg_item_id[0],1,raw=1))
        if user_id[1] != None and self.phase != 'TARGET':
            predict[1] = self.target_forward(user_id[1], item_id[1], 1, neg=False)
            if self.input_type == InputType.POINTWISE:
                loss_t = self.loss(predict[1], label[1]) + \
                    self.gamma * self.target_reg_loss(self.get_user_embedding(user_id[1],1,raw=1),
                                                    self.get_item_embedding(item_id[1],1,raw=1))
            elif self.input_type == InputType.PAIRWISE:
                neg_predict[1] = self.target_forward(user_id[1], neg_item_id[1], 1, neg=True)
                loss_t = self.loss(predict[1], neg_predict[1]) + \
                    self.gamma * self.target_reg_loss(self.get_user_embedding(user_id[1],1,raw=1),
                                                    self.get_item_embedding(item_id[1],1,raw=1),
                                                    self.get_item_embedding(neg_item_id[1],1,raw=1))
            
            if (self.loss_term == 0) or (self.delta == 0):
                loss_contrastive = torch.tensor(0, device = self.device)
            else:
                user_number = int(len(user_id[1]) / self.train_neg_sample_num)
                squeezed_user_id = user_id[1][0:user_number]

                '''squeezed_user_id = user_id[0][0:user_number]
                s_sign = (squeezed_user_id>self.target_num_users).float()
                s_users = (squeezed_user_id-(self.target_num_users-self.overlapped_num_users)) * s_sign
                o_users = squeezed_user_id * (1-s_sign)
                squeezed_user_id = (o_users + s_users).long()
                squeezed_user_ids = user_id[0][0:user_number]'''
            
                additional_user = torch.randint(0,self.target_num_users,(self.additional_user_samp,),device=self.device)
                squeezed_user_id = torch.cat((squeezed_user_id,additional_user),dim=0)
                bs = len(squeezed_user_id)
                t_user_id = squeezed_user_id
                same_user_matrix = t_user_id.repeat(bs,1)-t_user_id.repeat(bs,1).T
                sim_type = 'cosine'
                if sim_type == 'dot':
                    batch_sim = torch.mm(self.target_user_embedding_normed_no_grad[squeezed_user_id],self.target_user_embedding_normed_no_grad[squeezed_user_id].T)
                    pos_sign = torch.where(batch_sim>0,torch.ones_like(batch_sim,device=self.device),torch.zeros_like(batch_sim,device = self.device))
                elif sim_type == 'cosine':
                    batch_sim = torch.mm(self.target_user_embedding_normed_no_grad[squeezed_user_id],self.target_user_embedding_normed_no_grad[squeezed_user_id].T)
                    batch_norm = torch.norm(self.target_user_embedding_normed_no_grad[squeezed_user_id], p=2, dim=1)
                    batch_norm = torch.where(batch_norm>0, batch_norm, torch.ones_like(batch_norm,device = self.device))
                    batch_sim = batch_sim/torch.mm(batch_norm.unsqueeze(1),batch_norm.unsqueeze(1).T) + 1e-9
                    pos_sign = torch.where(batch_sim>self.cosine_threshold,torch.ones_like(batch_sim,device = self.device),torch.zeros_like(batch_sim,device = self.device))
                
                pos_sign = torch.where(same_user_matrix==0, torch.zeros_like(pos_sign,device = self.device), pos_sign)
                pos_num = torch.sum(pos_sign, dim=1)
                real_pos_num = pos_num
                pos_num = torch.max(pos_num,torch.ones_like(pos_num, device = self.device))

                #loss calculate
                loss_term = self.loss_term
                if self.single_model_config['name'] == 'SimpleX':
                    orig_dropout = self.both_model.dropout.p
                    self.both_model.dropout.p = 0 #change: when calculating contrastive loss term, dropout is deactivated

                if torch.sum(real_pos_num, dim=0)>0:
                    if loss_term == 1:#supervised contrastive loss, out
                        if self.raw_embedding_loss:
                            t_user_e = self.transform_raw_user(squeezed_user_id,self.get_user_embedding(squeezed_user_id,1,raw=1))
                            #t_user_e = self.get_user_embedding(squeezed_user_ids,1,raw=1)

                        else:
                            user_e = self.get_user_embedding(squeezed_user_id,1)
                            t_user_e = self.transform_user(squeezed_user_id, user_e)#[bs, embed]
                        sim_matrix = torch.mm(t_user_e,t_user_e.transpose(0,1))/self.tau#[bs, bs]
                        max_wo_same_sim_matrix = torch.where(same_user_matrix==0, -100000*torch.ones_like(sim_matrix,device=self.device), sim_matrix)#perform same user mask two times because same user dot product should not be counted in the max calculation to avoid log(0) in torch.log(torch.sum(all_matrix, dim=1)
                        sim_matrix_max = torch.unsqueeze(torch.max(max_wo_same_sim_matrix, dim=1)[0],1)
                        sim_matrix_norm = torch.exp(max_wo_same_sim_matrix - sim_matrix_max)
                        all_matrix = torch.where(same_user_matrix==0, torch.zeros_like(sim_matrix,device=self.device), sim_matrix_norm)
                        all_term = torch.sum(torch.log(torch.sum(all_matrix, dim=1))+sim_matrix_max.view(-1), dim=0)
                        pos_matrix = torch.where(pos_sign>0, sim_matrix, torch.zeros_like(sim_matrix,device = self.device))
                        pos_term = torch.sum(torch.sum(pos_matrix, dim=1)/pos_num, dim=0)
                        loss_contrastive = all_term - pos_term
                        #loss_contrastive = torch.tensor(0, device = self.device)
                    elif loss_term == 2:#supervised contrastive loss, simple version
                        if self.raw_embedding_loss:
                            t_user_e = self.transform_raw_user(squeezed_user_id,self.get_user_embedding(squeezed_user_id,1,raw=1))
                        else:
                            user_e = self.get_user_embedding(squeezed_user_id,1)
                            t_user_e = self.transform_user(squeezed_user_id, user_e)#[bs, embed]
                        sim_matrix = torch.mm(t_user_e,t_user_e.transpose(0,1))#[bs, bs]
                        all_matrix = torch.where(same_user_matrix==0, torch.zeros_like(sim_matrix,device=self.device), sim_matrix)
                        all_term = torch.sum(all_matrix)/(bs*(bs-1))
                        pos_matrix = torch.where(pos_sign>0, all_matrix, torch.zeros_like(all_matrix,device = self.device))
                        pos_term = torch.sum(torch.sum(pos_matrix, dim=1)/pos_num, dim=0)
                        loss_contrastive = all_term - pos_term
                else:
                    loss_contrastive = torch.tensor(0, device = self.device)

                if self.single_model_config['name'] == 'SimpleX':
                    self.both_model.dropout.p = orig_dropout

            loss_all += loss_contrastive.view(-1) * self.delta + self.lamda * self.param_reg_loss([self.user_transform_matrix_r])
            
        elif user_id[1] != None and self.phase == 'TARGET':
            predict[1] = self.target_train_forward(user_id[1], item_id[1], neg=False)
            if self.input_type == InputType.POINTWISE:
                loss_t = self.loss(predict[1], label[1]) + \
                    self.gamma * self.target_reg_loss(self.get_user_embedding(user_id[1],1,raw=1),
                                                    self.get_item_embedding(item_id[1],1,raw=1))
            elif self.input_type == InputType.PAIRWISE:
                neg_predict[1] = self.target_train_forward(user_id[1], neg_item_id[1], neg=True)
                loss_t = self.loss(predict[1], neg_predict[1]) + \
                    self.gamma * self.target_reg_loss(self.get_user_embedding(user_id[1],1,raw=1),
                                                    self.get_item_embedding(item_id[1],1,raw=1),
                                                    self.get_item_embedding(neg_item_id[1],1,raw=1))
            if self.single_model_config['name'] == 'LightGCN':
                self.user_gcn_embeddings, self.item_gcn_embeddings = [None, None]
            return loss_t

        if user_id[0] != None:
            loss_all += loss_s.view(-1) * self.alpha
        if user_id[1] != None:
            loss_all += loss_t.view(-1) * (1 - self.alpha)

        self.user_gcn_embeddings, self.item_gcn_embeddings = [None, None]
        return loss_all

        '''
        def predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            item = interaction[self.SOURCE_ITEM_ID]
            p = self.source_forward(user, item)
        elif self.phase == 'BOTH' or self.phase =='OVERLAP':
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            p = self.target_forward(user, item)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            p = self.target_train_forward(user, item)
        return self.sigmoid(p)
        '''

    def full_sort_predict(self, interaction):
        '''if self.single_model_config['name'] == 'LightGCN':
            if self.restore_user_e is not None or self.restore_item_e is not None:
                self.restore_user_e, self.restore_item_e = None, None'''
        aie = self.full_sort_get_all_item_weight()
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            user_e = self.get_user_embedding(user,0)
            overlap_item_e = aie[:self.overlapped_num_items]
            source_item_e = aie[self.target_num_items:]#overlap->target->source
            all_item_e = torch.cat([overlap_item_e, source_item_e], dim=0)
        elif self.phase == 'BOTH' or self.phase =='OVERLAP':
            user = interaction[self.TARGET_USER_ID]
            user_e = self.get_user_embedding(user,0)
            all_item_e = aie[:self.target_num_items]
            user_e = self.transform_user(user, user_e)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            user_e = self.get_user_embedding(user,0)
            all_item_e = aie[:self.target_num_items]
        if self.single_model_config['name'] in ['MF','LightGCN']:
            score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        elif self.single_model_config['name'] in ['SimpleX']:
            user_e = F.normalize(user_e, dim=1)
            all_item_e = F.normalize(all_item_e, dim=1)
            score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return self.sigmoid(score).view(-1)