import math
import torch
import torch.nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features // num_heads
        self.linears = torch.nn.ModuleList([torch.nn.Linear(in_features, self.out_features) for _ in range(num_heads)])
        self.tanh = torch.nn.Tanh()
        self.final_linear = torch.nn.Linear(self.out_features * num_heads, 1, bias=False)

    def forward(self, x):
        heads = [self.tanh(linear(x)) for linear in self.linears]
        heads = torch.cat(heads, dim=-1)
        weight = self.final_linear(heads)
        return weight

class RelatEntAtt(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(RelatEntAtt, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.gamma = torch.nn.Parameter(
            torch.Tensor([12]),
            requires_grad=False
        )
        self.epsilon = 2.0
        self.embedding_range = torch.nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.in_features]),
            requires_grad=False
        )
        self.project = MultiHeadAttention(in_features, out_features, num_heads=4)
        # self.W = torch.nn.Parameter(torch.Tensor(num_relations, in_features, out_features))
        self.a = torch.nn.Parameter(torch.Tensor(2*in_features, out_features))
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, h, r, adj_agg):
        E = h.size(0)
        R = r.size(0)
        # adj_agg = torch.sum(adj, dim=0)  # [E,E,R]->[E,R]
        # adj_agg = torch.mean(adj, dim=0)  # [E,E,R]->[E,R]
        # adj_agg = torch.max(adj, dim=0)[0]  # [E,E,R]->[E,R]
        adj_agg = F.softmax(adj_agg, dim=1)
        a_input = (h.repeat(1, R).view(R * E, -1) * r.repeat(E, 1)).view(E, R, -1) # [E,R,in]
        # a_input = torch.cat([h.repeat(1, R).view(R * E, -1), r.repeat(E, 1)], dim=1).view(E, R, -1) # [E,R,2*in]
        # a_input = torch.matmul(a_input, self.a)  # E*E*in
        # a_input = (ComplEx(h.repeat(1, R).view(R * E, -1), r.repeat(E, 1))).view(E, R, -1) # [E,R,in]
        # a_input = RotatE(h.repeat(1, R).view(R * E, -1), r.repeat(E, 1), self.embedding_range).view(E, R, -1) # [E,R,in]

        attention = self.leakyrelu(adj_agg.unsqueeze(-1) * a_input) # [E,R,1]*[E,R,in]->[E,R,in]
        # attention = adj_agg.unsqueeze(-1) * a_input # [E,R,1]*[E,R,in]->[E,R,in]
        attention_E = F.softmax(attention, dim=0)
        attention_R = F.softmax(attention, dim=1)
        attention_E = F.dropout(attention_E, self.dropout, training=self.training)
        attention_R = F.dropout(attention_R, self.dropout, training=self.training)
        alpha = self.linear(attention_R).view(R, E, -1)# [R,E,1]
        # alpha = self.project(attention_R).view(R, E, -1)# [R,E,1]
        alpha = F.softmax(alpha, dim=1) # 同种关系下对异质实体进行加权
        h_prime = torch.sum(attention_E * h.repeat(1, R).view(E, R, -1), dim=1)
        # h_prime = torch.mean(attention_E * h.repeat(1, R).view(E, R, -1), dim=1)
        # h_prime = torch.max(attention_E * h.repeat(1, R).view(E, R, -1), dim=1)[0]
        r_prime = torch.sum(attention_R * r.repeat(E, 1).view(E, R, -1), dim=0)
        # r_prime = torch.mean(attention_R * r.repeat(E, 1).view(E, R, -1), dim=0)
        # r_prime = torch.max(attention_R * r.repeat(E, 1).view(E, R, -1), dim=0)[0]

        # h_prime = torch.matmul(attention, torch.ones_like(r)) # [E,R]*[R,in]=[E,in]
        # r_prime = torch.matmul(attention.transpose(1, 0), torch.ones_like(h)) # [R,E]*[E,in]=[R,in]
        if self.concat:
            # return F.elu(h_prime), F.elu(r_prime)
            return F.elu(h_prime), F.elu(r_prime), alpha
        else:
            # return h_prime, r_prime
            return h_prime, r_prime, alpha

class RelatEntAtt3D(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(RelatEntAtt3D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # self.W = torch.nn.Parameter(torch.Tensor(num_relations, in_features, out_features))
        self.a = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, h, r, adj):
        E = h.size(0)
        R = r.size(0)
        a_input = (h.repeat(1, R * E).view(E * E * R, -1) * r.repeat(E * E, 1) * h.repeat(E, 1).repeat(1, R).view(
            E * E * R, -1)).view(E, E, R, -1)  # [E,E,R,in]
        adj_agg = F.softmax(adj, dim=1)
        # print(adj_agg)
        # attention = adj_agg.unsqueeze(-1) * a_input  # [E,E,R,1]*[E,E,R,in]->[E,E,R,in]
        attention = self.leakyrelu(adj_agg.unsqueeze(-1) * a_input)  # [E,E,R,1]*[E,E,R,in]->[E,E,R,in]
        attention = attention.sum(dim=0)  # [E,E,R,in]->[E,R,in]
        attention_E = F.softmax(attention, dim=0)
        attention_R = F.softmax(attention, dim=1)
        alpha = self.linear(attention_R).view(R, E, -1)  # [R,E,1]
        alpha = F.softmax(alpha, dim=1)#同种关系下对异质实体进行加权
        h_prime = torch.sum(attention_E * h.repeat(1, R).view(E, R, -1), dim=1)
        r_prime = torch.sum(attention_R * r.repeat(E, 1).view(E, R, -1), dim=0)

        # h_prime = torch.matmul(attention, torch.ones_like(r)) # [E,R]*[R,in]=[E,in]
        # r_prime = torch.matmul(attention.transpose(1, 0), torch.ones_like(h)) # [R,E]*[E,in]=[R,in]
        if self.concat:
            # return F.elu(h_prime), F.elu(r_prime)
            return F.elu(h_prime), F.elu(r_prime), alpha
        else:
            # return h_prime, r_prime
            return h_prime, r_prime, alpha

class CenterNeighAtt(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations, dropout, alpha, concat=True):
        super(CenterNeighAtt, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # self.W = torch.nn.Parameter(torch.Tensor(num_relations, in_features, out_features))
        self.a = torch.nn.Parameter(torch.Tensor(in_features, 1))
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, h, adj):
        E = h.size(0)
        adj_agg = torch.sum(adj, dim=0)  # [R,E,E]->[E,E]
        adj_agg = F.softmax(adj_agg, dim=1)
        a_input = (h.repeat(1, E).view(E * E, -1) * h.repeat(E, 1)).view(E, E, -1) # [E,E,in]

        attention = self.leakyrelu(adj_agg.unsqueeze(-1) * a_input)  # [E,R,1]*[E,R,in]->[E,R,in]
        # attention = adj_agg.unsqueeze(-1) * a_input  # [E,E,1]*[E,E,in]->[E,E,in]
        attention = F.softmax(attention, dim=0)
        attention = F.dropout(attention, self.dropout, training=self.training)
        alpha = self.linear(attention).view(E, E, -1)  # [E,E,1]
        alpha = alpha.sum(dim=0).unsqueeze(0)  # [1,E,1]
        alpha = F.softmax(alpha, dim=1)
        # h_prime = torch.sum(attention * h.repeat(1, E).view(E, E, -1), dim=1)
        # h_prime = torch.mean(attention * h.repeat(1, E).view(E, E, -1), dim=1)

        h_prime = torch.sum(attention * h.repeat(E, 1).view(E, E, -1), dim=1)
        # h_prime = torch.mean(attention * h.repeat(E, 1).view(E, E, -1), dim=1)

        # h_prime = torch.matmul(attention, h) # [E,E]*[E,in]=[E,in]
        if self.concat:
            return F.elu(h_prime),alpha
        else:
            return h_prime,alpha

class HGNLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, num_entities, num_relations, device, dropout, alpha, bias=True):
        super(HGNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_entities = num_entities
        self.device = device

        self.weight_ent = Parameter(torch.empty(in_features, out_features, device=self.device))
        self.weight_rel = Parameter(torch.empty(in_features, out_features, device=self.device))

        if bias:
            self.bias = Parameter(torch.empty(out_features))

        self.project = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(out_features, 1, bias=False)
        )
        self.relatentatt = RelatEntAtt(in_features, out_features, dropout=dropout, alpha=alpha, concat=False)
        self.relatentatt3d = RelatEntAtt3D(in_features, out_features, dropout=dropout, alpha=alpha, concat=False)
        self.centerneighatt = CenterNeighAtt(in_features, out_features, num_relations, dropout=dropout, alpha=alpha, concat=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_ent)
        torch.nn.init.xavier_uniform_(self.weight_rel)

        stdv = 1. / math.sqrt(self.weight_ent.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, ent_mat, rel_mat, adjacencies, A):
        supports = []
        for adjacency in adjacencies:
            support = torch.spmm(adjacency, ent_mat)
            supports.append(support)
            del adjacency  # 释放显存
            torch.cuda.empty_cache()  # 清除未使用显存
        supports = torch.cat(supports).view(self.num_relations, self.num_entities, self.in_features)  # R*E*in

        # A = torch.stack([adj.to_dense() for adj in adjacencies])# R*E*E
        # A_adj = torch.stack([torch.where(e > 0, torch.ones_like(e), torch.zeros_like(e)) - torch.eye(self.num_entities) for e in A]).to(self.device)
        # supports = torch.matmul(A, ent_mat)# R*E*in
        # A_rel = A.permute(1, 2, 0)
        ent_mat, rel_mat, alpha = self.relatentatt(ent_mat, rel_mat, A)
        # ent_mat, rel_mat, alpha = self.relatentatt3d(ent_mat, rel_mat, A_rel)
        # weight = self.project(rel_mat)
        # alpha = torch.sigmoid(weight).unsqueeze(1)  # R*1*1
        supports = torch.mul(supports, alpha)  # (R*E*in, R*E*1)= R*E*in
        # A = torch.mul(A, alpha)  # (R*E*E, R*E*1)= R*E*E
        # ent_mat,alpha = self.centerneighatt(ent_mat, A)
        # supports = torch.mul(supports, alpha)

        # ent_output = torch.matmul(supports, self.weight_ent)  # (R*E*in, in*out)=R*E*out
        ent_output = torch.sum(supports, 0)  # sum/GCN
        # ent_output = torch.mean(supports, 0)  # sum/GCN
        # ent_output = torch.max(supports, 0)[0]  # sum/GCN
        # ent_output, rel_mat = self.relatentatt(ent_output, rel_mat, A_rel)
        # ent_output = torch.max(ent_output, 0)[0]  # max/pooling: [0] return value; [1] return index
        # ent_output = torch.mean(ent_output, 0)  # mean

        ent_output = ent_mat + torch.matmul(ent_output, self.weight_ent)

        rel_output = torch.mm(rel_mat, self.weight_rel)  # (R*in, in*out) = R*out
        return ent_output, rel_output


class IAGNN(torch.nn.Module):

    def __init__(self, data, ent_dim, rel_dim,device, **kwargs):
        super(IAGNN, self).__init__()

        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.reshape_H = 2
        self.reshape_W = ent_dim

        # for CNN
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_height = kwargs["filt_height"]
        self.filt_width = kwargs["filt_width"]
        self.dropout = kwargs["dropout"]
        self.alpha = kwargs["alpha"]
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout(kwargs["feature_map_dropout"])

        self.emb_e = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.emb_r = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        # filt_dim = self.in_channels * self.out_channels * self.filt_height * self.filt_width
        # self.filter = torch.nn.Embedding(data.relations_num, filt_dim, padding_idx=0)
        # self.rel_filt = Parameter(torch.FloatTensor(ent_dim, filt_dim))
        filter_dim = self.in_channels * self.out_channels * self.filt_height * self.filt_width
        self.filter = torch.nn.Embedding(data.relations_num, filter_dim, padding_idx=0)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)

        # for GNN
        self.gc1 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num, device, self.dropout, self.alpha)
        self.gc2 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num,device, self.dropout, self.alpha)
        self.gc3 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num,device, self.dropout, self.alpha)
        self.drop_rate = kwargs["drop_rate"]
        self.bn3 = torch.nn.BatchNorm1d(ent_dim)
        self.bn4 = torch.nn.BatchNorm1d(ent_dim)
        self.bn5 = torch.nn.BatchNorm1d(ent_dim)

        # for prediction
        self.bn2 = torch.nn.BatchNorm1d(ent_dim)
        fc_length = (self.reshape_H - self.filt_height + 1) * \
                    (self.reshape_W - self.filt_width + 1) * \
                    self.out_channels
        self.fc = torch.nn.Linear(fc_length, ent_dim)
        self.register_parameter('bias', Parameter(torch.zeros(data.entities_num)))
        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_e.weight.data)
        torch.nn.init.xavier_normal_(self.emb_r.weight.data)
        # torch.nn.init.xavier_normal_(self.rel_filt)
        torch.nn.init.xavier_normal_(self.filter.weight.data)

    def hgn(self, adjacencies, A):
        # layer 1
        # print('adjacencies',adjacencies)
        # print('self.emb_r.weight.shape',self.emb_r.weight.shape)#torch.Size([92, 200])
        # print('self.emb_e.weight.shape',self.emb_e.weight.shape)#torch.Size([135, 200])
        embedded_ent, embedded_rel = self.gc1(self.emb_e.weight, self.emb_r.weight, adjacencies, A)
        # print('embedded_ent.shape',embedded_ent.shape)#torch.Size([135, 200])
        # print('embedded_rel.shape',embedded_rel.shape)#torch.Size([92, 200])
        embedded_ent = torch.relu(self.bn3(embedded_ent))
        # embedded_rel = torch.relu(self.bn3(embedded_rel))
        # print('embedded_ent.shape', embedded_ent.shape)#torch.Size([135, 200])
        embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
        # embedded_rel = F.dropout(embedded_rel, self.drop_rate, training=self.training)

        # # layer 2
        # embedded_ent, embedded_rel = self.gc2(embedded_ent, embedded_rel, adjacencies, A)
        # embedded_ent = torch.relu(self.bn4(embedded_ent))
        # embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
        # # layer 3
        # embedded_ent, embedded_rel = self.gc3(embedded_ent, embedded_rel, adjacencies, A)
        # embedded_ent = torch.relu(self.bn5(embedded_ent))
        # embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
        return embedded_ent, embedded_rel

    def forward(self, e1, rel, embedded_ent, embedded_rel):
        # e1_embedded = embedded_ent[e1].reshape(-1, 1, self.reshape_H, self.reshape_W)
        # x = self.bn0(e1_embedded)
        # x = self.inp_drop(x)
        # x = x.permute(1, 0, 2, 3)

        # print('e1.shape',e1.shape)#torch.Size([128])
        # print('rel.shape',rel.shape)#torch.Size([128])
        # print('embedded_ent.shape',embedded_ent.shape)#torch.Size([135, 200])
        # print('embedded_rel.shape',embedded_rel.shape)#torch.Size([92, 200])
        ent_emb = embedded_ent[e1].reshape(-1, 1, self.ent_dim)
        rel_emb = embedded_rel[rel].reshape(-1, 1, self.rel_dim)
        # print('ent_emb.shape',ent_emb.shape)#torch.Size([128, 1, 200])
        # print('rel_emb.shape',rel_emb.shape)#torch.Size([128, 1, 200])

        x = torch.cat([ent_emb, rel_emb], 1).reshape(-1, 1, self.reshape_H, self.reshape_W)
        # print('x1.shape', x.shape)  # torch.Size([128, 1, 2, 200])
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = x.permute(1, 0, 2, 3)
        # print('x2.shape', x.shape)# torch.Size([1, 128, 2, 200])

        f = self.filter(rel)
        # print('f1.shape', f.shape)#torch.Size([128, 216])
        f = f.reshape(ent_emb.size(0) * self.in_channels * self.out_channels, 1, self.filt_height,
                      self.filt_width)
        # print('f2.shape', f.shape)#torch.Size([4608, 1, 2, 3])
        x = F.conv2d(x, f, groups=ent_emb.size(0))

        # print('x3.shape', x.shape)#torch.Size([1, 4608, 1, 198])
        x = x.reshape(ent_emb.size(0), self.out_channels, self.reshape_H - self.filt_height + 1,
                      self.reshape_W - self.filt_width + 1)
        # print('x4.shape', x.shape)#torch.Size([128, 36, 1, 198])

        # f = torch.mm(embedded_rel[rel], self.rel_filt)
        # f = f.reshape(e1_embedded.size(0) * self.in_channels * self.out_channels, 1,
        #               self.filt_height, self.filt_width)
        # # f = self.filter(rel).reshape(e1_embedded.size(0) * self.in_channels * self.out_channels, 1,
        # #                              self.filt_height, self.filt_width)
        # x = F.conv2d(x, f, groups=e1_embedded.size(0))
        # x = x.reshape(e1_embedded.size(0), self.out_channels, self.reshape_H - self.filt_height + 1,
        #               self.reshape_W - self.filt_width + 1)

        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_map_drop(x)
        # print('x5.shape', x.shape)#torch.Size([128, 36, 1, 198])

        x = x.reshape(ent_emb.size(0), -1)
        # print('x6.shape', x.shape)#torch.Size([128, 7128])
        x = self.fc(x)
        # print('x7.shape', x.shape)  # torch.Size([128, 200])
        x = self.bn2(x)
        # x = torch.relu(x)
        x = self.hidden_drop(x)

        x = torch.mm(x, embedded_ent.transpose(1, 0))
        # print('x7.shape', x.shape)  # torch.Size([128, 135])
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred

# class IAGNN(torch.nn.Module):
#
#     def __init__(self, data, ent_dim, rel_dim, device, **kwargs):#ConvTransE
#         super(IAGNN, self).__init__()
#         self.ent_dim = ent_dim
#         self.rel_dim = rel_dim
#         self.reshape_H = 2
#         self.reshape_W = ent_dim
#
#         self.emb_e = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
#         self.emb_r = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
#         self.in_channels = kwargs["in_channels"]
#         self.out_channels = kwargs["out_channels"]
#         self.filt_height = kwargs["filt_height"]
#         self.filt_width = kwargs["filt_width"]
#         self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
#         self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
#         self.feature_map_drop = torch.nn.Dropout(kwargs["feature_map_dropout"])
#         self.loss = torch.nn.BCELoss()
#
#         # for GNN
#         self.dropout = kwargs["dropout"]
#         self.alpha = kwargs["alpha"]
#         self.gc1 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num, device, self.dropout, self.alpha)
#         self.gc2 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num,device, self.dropout, self.alpha)
#         self.gc3 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num,device, self.dropout, self.alpha)
#         self.drop_rate = kwargs["drop_rate"]
#         self.bn3 = torch.nn.BatchNorm1d(ent_dim)
#         self.bn4 = torch.nn.BatchNorm1d(ent_dim)
#         self.bn5 = torch.nn.BatchNorm1d(ent_dim)
#
#         self.conv1 = torch.nn.Conv1d(self.filt_height, self.out_channels, self.filt_width, stride=1, padding=int(math.floor(self.filt_width/2)))
#         self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
#         self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
#         self.bn2 = torch.nn.BatchNorm1d(ent_dim)
#         self.register_parameter('bias', Parameter(torch.zeros(data.entities_num)))
#         fc_length = self.out_channels*ent_dim
#         self.fc = torch.nn.Linear(fc_length, ent_dim)
#
#     def init(self):
#         torch.nn.init.xavier_normal_(self.emb_e.weight.data)
#         torch.nn.init.xavier_normal_(self.emb_r.weight.data)
#
#     def hgn(self, adjacencies, A):
#         # layer 1
#         # print('adjacencies',adjacencies)
#         # print('self.emb_r.weight.shape',self.emb_r.weight.shape)#torch.Size([92, 200])
#         # print('self.emb_e.weight.shape',self.emb_e.weight.shape)#torch.Size([135, 200])
#         embedded_ent, embedded_rel = self.gc1(self.emb_e.weight, self.emb_r.weight, adjacencies, A)
#         # print('embedded_ent.shape',embedded_ent.shape)#torch.Size([135, 200])
#         # print('embedded_rel.shape',embedded_rel.shape)#torch.Size([92, 200])
#         embedded_ent = torch.relu(self.bn3(embedded_ent))
#         # embedded_rel = torch.relu(self.bn3(embedded_rel))
#         # print('embedded_ent.shape', embedded_ent.shape)#torch.Size([135, 200])
#         embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
#         # embedded_rel = F.dropout(embedded_rel, self.drop_rate, training=self.training)
#
#         # # layer 2
#         # embedded_ent, embedded_rel = self.gc2(embedded_ent, embedded_rel, adjacencies, A)
#         # embedded_ent = torch.relu(self.bn4(embedded_ent))
#         # embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
#         # # layer 3
#         # embedded_ent, embedded_rel = self.gc3(embedded_ent, embedded_rel, adjacencies, A)
#         # embedded_ent = torch.relu(self.bn5(embedded_ent))
#         # embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
#         return embedded_ent, embedded_rel
#
#     def forward(self, e1, rel, embedded_ent, embedded_rel):
#         ent_emb = embedded_ent[e1].reshape(-1, 1, self.ent_dim)
#         rel_emb = embedded_rel[rel].reshape(-1, 1, self.rel_dim)
#         stacked_inputs = torch.cat([ent_emb, rel_emb], 1).reshape(-1, 1, self.reshape_H, self.reshape_W)
#         stacked_inputs = self.bn0(stacked_inputs)
#         stacked_inputs = stacked_inputs.squeeze(1)
#         x= self.inp_drop(stacked_inputs)
#         x= self.conv1(x)
#         x= x.unsqueeze(2)
#         x= self.bn1(x)
#         x= F.relu(x)
#         x = self.feature_map_drop(x)
#         x = x.reshape(ent_emb.size(0), -1)
#         x = self.fc(x)
#         x = self.hidden_drop(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = torch.mm(x, embedded_ent.transpose(1, 0))
#         pred = F.sigmoid(x)
#
#         return pred

# class IAGNN(torch.nn.Module):
#
#     def __init__(self, data, ent_dim, rel_dim,device, **kwargs):
#         super(IAGNN, self).__init__()
#         self.ent_dim = ent_dim
#         self.rel_dim = rel_dim
#         self.reshape_H = 2
#         self.reshape_W = ent_dim
#
#         # for CNN
#         self.in_channels = kwargs["in_channels"]
#         self.out_channels = kwargs["out_channels"]
#         self.filt_height = kwargs["filt_height"]
#         self.filt_width = kwargs["filt_width"]
#         self.dropout = kwargs["dropout"]
#         self.alpha = kwargs["alpha"]
#         self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
#         self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
#         self.feature_map_drop = torch.nn.Dropout(kwargs["feature_map_dropout"])
#
#         self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
#         self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
#         self.bn2 = torch.nn.BatchNorm1d(ent_dim)
#
#         # for GNN
#         self.gc1 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num, device, self.dropout, self.alpha)
#         self.gc2 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num,device, self.dropout, self.alpha)
#         self.bn3 = torch.nn.BatchNorm1d(ent_dim)
#         self.bn4 = torch.nn.BatchNorm1d(ent_dim)
#         self.drop_rate = kwargs["drop_rate"]
#
#         self.embed_drop = torch.nn.Dropout(kwargs["input_dropout"])
#         self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
#         self.feature_drop = torch.nn.Dropout(kwargs["feature_map_dropout"])
#
#         self.emb_e = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
#         self.emb_r = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
#         self.conv = torch.nn.Conv2d(self.in_channels, out_channels=self.out_channels, kernel_size=(5, 5), stride=1, padding=0, bias=False)
#
#         # fc_length = (self.reshape_H - self.filt_height + 1) * \
#         #             (self.reshape_W - self.filt_width + 1) * \
#         #             self.out_channels
#         # self.fc = torch.nn.Linear(fc_length, ent_dim)
#         flat_sz_h = 20 - 5 + 1
#         flat_sz_w = 20 - 5 + 1
#         self.flat_sz = flat_sz_h * flat_sz_w * self.out_channels
#         self.fc = torch.nn.Linear(self.flat_sz, ent_dim)
#         self.register_parameter('bias', Parameter(torch.zeros(data.entities_num)))
#         self.loss = torch.nn.BCELoss()
#
#     def init(self):
#         torch.nn.init.xavier_normal_(self.emb_e.weight.data)
#         torch.nn.init.xavier_normal_(self.emb_r.weight.data)
#
#     def hgn(self, adjacencies):
#         # layer 1
#         # print('adjacencies',adjacencies)
#         # print('self.emb_r.weight.shape',self.emb_r.weight.shape)#torch.Size([92, 200])
#         # print('self.emb_e.weight.shape',self.emb_e.weight.shape)#torch.Size([135, 200])
#         embedded_ent, embedded_rel = self.gc1(self.emb_e.weight, self.emb_r.weight, adjacencies)
#         # print('embedded_ent.shape',embedded_ent.shape)#torch.Size([135, 200])
#         # print('embedded_rel.shape',embedded_rel.shape)#torch.Size([92, 200])
#         embedded_ent = torch.relu(self.bn3(embedded_ent))
#         # embedded_rel = torch.relu(self.bn3(embedded_rel))
#         # print('embedded_ent.shape', embedded_ent.shape)#torch.Size([135, 200])
#         embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
#         # embedded_rel = F.dropout(embedded_rel, self.drop_rate, training=self.training)
#
#         # layer 2
#         # embedded_ent, embedded_rel = self.gc2(embedded_ent, embedded_rel, adjacencies)
#         # embedded_ent = torch.relu(self.bn4(embedded_ent))
#         # embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
#         return embedded_ent, embedded_rel
#
#     def forward(self, e1, rel, embedded_ent, embedded_rel):
#         ent_emb = embedded_ent[e1].reshape(-1, 1, self.ent_dim)
#         rel_emb = embedded_rel[rel].reshape(-1, 1, self.rel_dim)
#         stack_inp = torch.cat([ent_emb, rel_emb], 1)
#         stack_inp = torch.transpose(stack_inp, 2, 1).reshape(-1, 1, 20, 20)
#         x = self.embed_drop(stack_inp)
#         x = self.bn0(x)
#         x = self.conv(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.feature_drop(x)
#         x = x.view(-1, self.flat_sz)
#         x = self.fc(x)
#         x = self.hidden_drop(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#
#         x = torch.mm(x, embedded_ent.transpose(1, 0))
#         x = torch.sigmoid(x)
#
#         return x

# class IAGNN(torch.nn.Module):
#
#     def __init__(self, data, ent_dim, rel_dim,device, **kwargs):
#         super(IAGNN, self).__init__()
#
#         self.ent_dim = ent_dim
#         self.rel_dim = rel_dim
#
#         # for CNN
#         self.dropout = kwargs["dropout"]
#         self.alpha = kwargs["alpha"]
#
#         self.emb_e = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
#         self.emb_r = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
#
#         # for GNN
#         self.gc1 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num, device, self.dropout, self.alpha)
#         self.gc2 = HGNLayer(ent_dim, ent_dim, data.entities_num, data.relations_num,device, self.dropout, self.alpha)
#         self.drop_rate = kwargs["drop_rate"]
#         self.bn3 = torch.nn.BatchNorm1d(ent_dim)
#         self.bn4 = torch.nn.BatchNorm1d(ent_dim)
#
#
#         self.register_parameter('bias', Parameter(torch.zeros(data.entities_num)))
#         self.loss = torch.nn.BCELoss()
#
#     def init(self):
#         torch.nn.init.xavier_normal_(self.emb_e.weight.data)
#         torch.nn.init.xavier_normal_(self.emb_r.weight.data)
#
#     def hgn(self, adjacencies):
#         # layer 1
#         # print('adjacencies',adjacencies)
#         # print('self.emb_r.weight.shape',self.emb_r.weight.shape)#torch.Size([92, 200])
#         # print('self.emb_e.weight.shape',self.emb_e.weight.shape)#torch.Size([135, 200])
#         embedded_ent, embedded_rel = self.gc1(self.emb_e.weight, self.emb_r.weight, adjacencies)
#         # print('embedded_ent.shape',embedded_ent.shape)#torch.Size([135, 200])
#         # print('embedded_rel.shape',embedded_rel.shape)#torch.Size([92, 200])
#         embedded_ent = torch.relu(self.bn3(embedded_ent))
#         # embedded_rel = torch.relu(self.bn3(embedded_rel))
#         # print('embedded_ent.shape', embedded_ent.shape)#torch.Size([135, 200])
#         embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
#         # embedded_rel = F.dropout(embedded_rel, self.drop_rate, training=self.training)
#
#         # # layer 2
#         # embedded_ent, embedded_rel = self.gc2(embedded_ent, embedded_rel, adjacencies)
#         # embedded_ent = torch.relu(self.bn4(embedded_ent))
#         # embedded_ent = F.dropout(embedded_ent, self.drop_rate, training=self.training)
#         return embedded_ent, embedded_rel
#
#     def forward(self, e1, rel, embedded_ent, embedded_rel):#DistMult
#         ent_emb = embedded_ent[e1].reshape(-1, 1, self.ent_dim)
#         rel_emb = embedded_rel[rel].reshape(-1, 1, self.rel_dim)
#
#         obj_emb = ent_emb * rel_emb
#         x = torch.mm(obj_emb, embedded_ent.transpose(1, 0))
#
#         return torch.sigmoid(x)


