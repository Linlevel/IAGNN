import torch
from torch.nn import functional as F
# h = torch.Tensor([[1,2],[4,5],[7,8]])
# r = torch.Tensor([[1,0],[0,1]])
# hi = h.repeat(1, 3).view(3 * 3, -1)
# hi = h.repeat(1, 2).view(2 * 3, -1)
# # hj = h.repeat(3, 1)
# ri = r.repeat(3, 1)
# h_agg = hi * hj
# hr_agg = hi * ri
# hi = h.repeat(1, 2 * 3).view(3 * 2 * 3, -1)
# hj = h.repeat(3, 1).repeat(1, 2).view(3 * 2 * 3, -1)
# ri = r.repeat(3 * 3, 1)
# print(hi)
# print(ri)
# print(hj)


def normalization(adjacencies):
    # 获取稀疏矩阵的尺寸
    size = adjacencies.size()

    # 获取非零元素的索引和值
    indices = adjacencies._indices()
    values = adjacencies._values()

    # 计算度
    row_indices = indices[0]
    degrees = torch.bincount(row_indices, minlength=size[0])

    # 计算D^-0.5
    degree_rsq = torch.pow(degrees, -0.5)

    # 构造归一化后的稀疏矩阵
    degree_rsq_values = degree_rsq[row_indices]
    normalized_values = degree_rsq_values * values * degree_rsq[row_indices]

    nor_adj = torch.sparse_coo_tensor(indices, normalized_values, size)

    return nor_adj
def get_adj(alpha):
    adjacencies = []
    dia_rows = dia_columns = [i for i in range(4)]
    dia_value = [alpha for i in range(4)]
    a = 1.0
    train_data_id = [
        [0,0,1],
        [0,2,2],
        [0,1,3],
        [1,2,2],
        [2,1,3],
        [2,0,0]
    ]
    for i in range(3):
        rows, columns, values = [], [], []
        for h, r, t in train_data_id:
            if i == r:
                rows.append(h)
                columns.append(t)
                values.append(a - alpha)
        rows = rows + dia_rows
        columns = columns + dia_columns
        values = values + dia_value
        # # 计算度
        row_indices = torch.LongTensor(rows)
        degrees = torch.bincount(row_indices, minlength=4)
        # 计算D^-0.5
        degree_rsq = torch.pow(degrees, -0.5)
        # 构造归一化后的稀疏矩阵
        degree_rsq_values = degree_rsq[row_indices]
        normalized_values = degree_rsq_values * torch.FloatTensor(values) * degree_rsq[row_indices]
        sparse_matrix = torch.sparse_coo_tensor(torch.LongTensor([rows, columns]), normalized_values, [4, 4])
        # sparse_matrix = normalization(sparse_matrix)
        adjacencies.append(sparse_matrix)
    return adjacencies

def get_adj_r(alpha):
    adjacencies = []
    dia_columns = [i for i in range(3)]
    dia_value = [alpha for i in range(3)]
    a = 1.0
    train_data_id = [
        [0,0,1],
        [0,2,2],
        [0,1,3],
        [1,2,2],
        [2,1,3],
        [2,0,0]
    ]
    dense_matrix_sum = torch.zeros(4, 3)
    for i in range(4):
        rows, columns, values = [], [], []
        for h, r, t in train_data_id:
            if i == h:
                rows.append(t)
                columns.append(r)
                values.append(a - alpha)
        dia_rows = [i for j in range(3)]
        rows = rows + dia_rows
        columns = columns + dia_columns
        values = values + dia_value
        # 计算度
        column_indices = torch.LongTensor(columns)
        degrees = torch.bincount(column_indices, minlength=4)
        # 计算D^-0.5
        degree_rsq = torch.pow(degrees, -0.5)
        # 构造归一化后的稀疏矩阵
        degree_rsq_values = degree_rsq[column_indices]
        normalized_values = degree_rsq_values * torch.FloatTensor(values) * degree_rsq[column_indices]
        sparse_matrix = torch.sparse_coo_tensor(torch.LongTensor([rows, columns]), normalized_values, [4, 3])
        dense_matrix = sparse_matrix.to_dense()
        dense_matrix_sum += dense_matrix
        adjacencies.append(sparse_matrix)
    return adjacencies, dense_matrix_sum

adj_nnz = get_adj(0.5)
adj_nnz_r, adj_nnz_r_sum = get_adj_r(0.5)
# adj_1 = get_adj(1.0)
# print(adj_nnz)
# print(adj_nnz_r)
A = torch.stack([adj.to_dense() for adj in adj_nnz])#[R,E,E]
A_r = torch.stack([adj.to_dense() for adj in adj_nnz_r])#[R,E,E]
# A_adj = torch.stack([torch.where(e > 0, torch.ones_like(e), torch.zeros_like(e))-torch.eye(4) for e in A])
# print(A_adj)
# print(A)
# print(A_r)
# print(adj_nnz_r_sum)
A_rel = A.permute(1, 2, 0)
# print(A_rel)
# h = torch.nn.Embedding(4, 2, padding_idx=0).weight
# r = torch.nn.Embedding(3, 2, padding_idx=0).weight
h = torch.Tensor([[1,2],[4,5],[7,8],[6,8]])
r = torch.Tensor([[1,0],[0,1],[1,1]])
E = h.size(0)
R = r.size(0)

adj_agg = torch.sum(A_rel, dim=0)  # [E,E,R]->[E,R]
# a_input = (h.repeat(1, R).view(R * E, -1) * r.repeat(E, 1)).view(E, R, -1)  # [E,R,in]
adj_agg = F.softmax(adj_agg, dim=1)
print(adj_agg)
# # attention = self.leakyrelu(adj_agg.unsqueeze(-1) * a_input) # [E,R,1]*[E,R,in]->[E,R,in]
# attention = adj_agg.unsqueeze(-1) * a_input  # [E,R,1]*[E,R,in]->[E,R,in]
# attention_E = F.softmax(attention, dim=0)
# attention_R = F.softmax(attention, dim=1)
# linear = torch.nn.Linear(2, 1)
# alpha = linear(attention_R).view(R, E, -1)# [R,E,1]
# alpha = F.softmax(alpha, dim=0)#同质实体下对不同关系进行加权
# # alpha = F.softmax(alpha, dim=1)#同种关系下对异质实体进行加权
# h_prime = torch.sum(attention_E * h.repeat(1, R).view(E, R, -1), dim=1)
# r_prime = torch.sum(attention_R * r.repeat(E, 1).view(E, R, -1), dim=0)
# print(alpha)
# print(h)
# print(r)
# print(h_prime)
# print(r_prime)


# a_input = (h.repeat(1, R * E).view(E * E * R, -1) * r.repeat(E * E, 1) * h.repeat(E, 1).repeat(1, R).view(E * E * R, -1)).view(E, E, R, -1)  # [E,E,R,in]
# adj_agg = F.softmax(A_rel, dim=1)
# # print(adj_agg)
# attention = adj_agg.unsqueeze(-1) * a_input  # [E,E,R,1]*[E,E,R,in]->[E,E,R,in]
# attention = attention.sum(dim=0) # [E,E,R,in]->[E,R,in]
# attention_E = F.softmax(attention, dim=0)
# attention_R = F.softmax(attention, dim=1)
# linear = torch.nn.Linear(2, 1)
# alpha = linear(attention_R).view(R, E, -1) # [R,E,1]
# alpha = F.softmax(alpha, dim=0)#同质实体下对不同关系进行加权
# # alpha = F.softmax(alpha, dim=1)#同种关系下对异质实体进行加权
# h_prime = torch.sum(attention_E * h.repeat(1, R).view(E, R, -1), dim=1)
# r_prime = torch.sum(attention_R * r.repeat(E, 1).view(E, R, -1), dim=0)
# print(alpha)