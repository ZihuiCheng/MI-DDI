import torch
#from torch import nn
import pandas as pd
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
from stream_sub import drug2emb_encoder
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import (GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_mean_pool,
                                max_pool_neighbor_x,
                                global_add_pool)

from layers import (
    CoAttentionLayer,
    RESCAL,
    RESCAL
)
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv import MessagePassing
import math
from torch.nn import Linear, GRU, Parameter
from torch.nn.functional import leaky_relu
from torch_geometric.nn import Set2Set, NNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn.init import kaiming_uniform_, zeros_

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
         tensor.data.fill_(0)


class MultiHeadTripletAttention(MessagePassing):
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        super(MultiHeadTripletAttention, self).__init__(aggr='add', node_dim=0, **kwargs)  # aggr='mean'
        # node_dim = 0 for multi-head aggr support
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)  # time consuming 13s
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)  # time consuming 12.14s
        alpha = leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        # return x_j * alpha
        # return self.prelu(alpha * e_ij * x_j)
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        return '{node_channels}, {node_channels}, heads={heads}'.format(**self.__dict__)


class Block(torch.nn.Module):
    def __init__(self, dim, edge_dim, heads=4, time_step=3):
        super(Block, self).__init__()
        self.time_step = time_step
        self.conv = MultiHeadTripletAttention(dim, edge_dim, heads)  # GraphMultiHeadAttention
        self.gru = GRU(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        for i in range(self.time_step):
            m = F.celu(self.conv.forward(x, edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.ln(x.squeeze(0))
        return x



from torch import Tensor
class BaseGGNN(MessagePassing):
    def __init__(self, state_size: int, num_layers: int,
                 aggr: str = 'add',
                 bias: bool = True, total_edge_types: int = 10,
                 use_resnet=False,subnum: int=20,atomnum: int=20):
        super(BaseGGNN, self).__init__(aggr=aggr)

        self.state_size = state_size
        self.out_channels = state_size
        self.num_layers = num_layers
        self.use_resnet = use_resnet
        self.subnum=subnum
        self.atomnum=atomnum
        self.weight = nn.Parameter(Tensor(num_layers, state_size, state_size))

        if self.use_resnet:
            self.mlp_2 = torch.nn.Sequential(
                torch.nn.Linear(state_size, state_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(state_size, state_size),
            )
            self.mlp_1 = torch.nn.Sequential(
                torch.nn.Linear(state_size, state_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(state_size, state_size),
            )
        else:
            self.rnn = torch.nn.GRUCell(state_size, state_size, bias=bias)

        # edge_type_tensor should be of the type (e, D, D), where e is the
        # total number of edge types
        # and D is the feature size
        self.edge_type_weight = nn.Parameter(
            torch.zeros(total_edge_types, state_size, state_size),
            requires_grad=True)
        self.edge_type_bias = nn.Parameter(torch.zeros(1, state_size),
                                    requires_grad=True)

        self.reset_parameters()

        self.mlp = nn.ModuleList([nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, 65)
                                  ])
        self.demb = Embeddings(23532, 384, self.subnum, 0.1)
        self.pemb = Embeddings(23532, 384, self.subnum, 0.1)

        self.d_encoder = Encoder_MultipleLayers(2, 384, 1536,
                                                12, 0.1,
                                                0.1)
        self.p_encoder = Encoder_MultipleLayers(2, 384, 1536,
                                                12, 0.1,
                                                0.1)


        self.mlp1 = nn.Linear(384, 128)
        self.mlp2 = nn.Linear(128, 128)
        self.decoder = nn.Sequential(
            # nn.Linear(self.flatten_dim, 512),
            # nn.ReLU(True),
            #
            # nn.BatchNorm1d(512),
            nn.Linear(2*(self.subnum+self.atomnum), 65),
            # nn.ReLU(True),
            #
            # nn.BatchNorm1d(256),
            # nn.Linear(256, 128),
            # nn.ReLU(True),
            # nn.BatchNorm1d(128),
            # # output layer
            # nn.Linear(128, 1)
        )
        df = pd.read_csv('ssidata/drug_smiles.csv', header=0)
        key = []
        value = []
        for i in df["drug_id"]:  # “number”用作键
            key.append(i)
        for j in df["smiles"]:  # “score”用作值
            value.append(j)
        self.dic = dict(zip(key, value))
        # for line in df.values:
        #     self.dic = {}
        #     for item, data in zip(df, line.tolist()):
        #         self.dic[item] = data
        #print(self.dic)



        self.mlp = nn.ModuleList([nn.Linear(2*(self.subnum+self.atomnum), 128),
                              nn.ELU(),
                              nn.Dropout(p=0.1),
                              nn.Linear(128, 128),
                              nn.ELU(),
                              nn.Dropout(p=0.1),
                              nn.Linear(128, 65)
                              ])
    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)

        return vectors
        
 

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_type_weight)
        glorot(self.edge_type_bias)
        if self.use_resnet:
            for layer in self.mlp_1:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.mlp_2:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            self.rnn.reset_parameters()

    def forward(self,triples):
        """
        h_data, t_data, rels = triples
        x1,edge_index1,edge_attr1=h_data.x,h_data.edge_index,h_data.edge_attr
        edge_attr: tensor of size (n, e) - n is the number of edges, e is the
        total number of edge types
        """#
        h_data, t_data, rels = triples
        atomnum=self.atomnum
        subnum=self.subnum
        #print(len(h_data.smiles))
        #print(self.dic[h_data.smiles])
        #print(self.dic[t_data.smiles])
        #print(self.dic)
        d=[]
        p=[]
        for i in range(len(h_data.smiles)):
            d.append(self.dic[h_data.smiles[i]])
            p.append(self.dic[t_data.smiles[i]])


        d_v, input_mask_d = drug2emb_encoder(d,subnum)
        p_v, input_mask_p = drug2emb_encoder(p,subnum)
        d_v=np.array(d_v)
        d_v=torch.from_numpy(d_v).cuda()
        p_v = np.array(p_v)
        p_v= torch.from_numpy(p_v).cuda()
        input_mask_d  = np.array(input_mask_d )
        input_mask_d  = torch.from_numpy(input_mask_d ).cuda()
        input_mask_p = np.array(input_mask_p)
        input_mask_p = torch.from_numpy(input_mask_p).cuda()


        ex_d_mask = input_mask_d.unsqueeze(1).unsqueeze(2).cuda()
        ex_p_mask = input_mask_p.unsqueeze(1).unsqueeze(2).cuda()

        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        d_emb = self.demb(d_v)  # batch_size x seq_length x embed_size
        p_emb = self.pemb(p_v)
        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...

        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        # print("1111111",d_encoded_layers)
        p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())
        # print("11111111111",p_encoded_layers)
        d_encoded_layers = self.mlp1(d_encoded_layers)
        p_encoded_layers = self.mlp1(p_encoded_layers)

        x1, edge_index1, edge_attr1,batch1 = h_data.x, h_data.edge_index, h_data.edge_attr,h_data.batch

        x2, edge_index2, edge_attr2,batch2 = t_data.x, t_data.edge_index, t_data.edge_attr,t_data.batch
        if x1.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x1.size(-1) < self.out_channels:
            zero = x1.new_zeros(x1.size(0), self.out_channels - x1.size(-1))
            x1 = torch.cat([x1, zero], dim=1)
        if x2.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x2.size(-1) < self.out_channels:
            zero = x2.new_zeros(x2.size(0), self.out_channels - x2.size(-1))
            x2 = torch.cat([x2, zero], dim=1)

        for i in range(self.num_layers):
            m1 = torch.matmul(x1, self.weight[i])
            m2 = torch.matmul(x2, self.weight[i])
            m1 = self.propagate(edge_index1, x=m1,edge_attr=edge_attr1,size=None)
            m2 = self.propagate(edge_index2, x=m2, edge_attr=edge_attr2, size=None)
            if self.use_resnet:
                x1 = self.mlp_2(m1 + self.mlp_1(x1))
                x2 = self.mlp_2(m2 + self.mlp_1(x2))
            else:
                x1 = self.rnn(m1, x1)
                x2 = self.rnn(m2, x2)
        max=0
        zhi=atomnum
        #print(111111111111)
        #print(batch1[-1])

            # f.write('{0}\t{1}\t{2}\t{7}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\n'.format(
            #     args.in_file[5:8], args.seed, args.aggregator, loss_test.item(), acc_test, f1_test, recall_test, args.feature_type))

        for i in range(128):
            a=x1[torch.where(batch1==i)]

            # print(a.shape)
            if a.shape[0]<zhi:
                pad=nn.ZeroPad2d(padding=(0, 0, 0, zhi-a.shape[0]))

                a=pad(a).view(1,zhi,-1)
            else:
                a=a[0:zhi].view(1,zhi,-1)
            if i==0:
                a1=a
            else:
                a1=torch.cat([a1,a],dim=0)
        for i in range(128):
            a=x2[torch.where(batch2==i)]

            if a.shape[0]<zhi:

                pad=nn.ZeroPad2d(padding=(0, 0, 0, zhi-a.shape[0]))
                a=pad(a).view(1,zhi,-1)
            else:
                a=a[0:zhi].view(1,zhi,-1)
            if i==0:
                a2=a
            else:
                a2=torch.cat([a2,a],dim=0)

        #x1=global_mean_pool(x1, batch1)
        #x2 = global_mean_pool(x2, batch2)

        a11=a1
        a22=a2
        a1= self.mlp2(a1)
        a2 = self.mlp2(a2)
        d_encoded_layers1=d_encoded_layers
        p_encoded_layers1 = p_encoded_layers
        # print("ddddd", d_encoded_layers1.shape)
        # print("a1111", a1.shape)
        if d_encoded_layers.size(0) < 128:
            # print("ddddd", d_encoded_layers.shape)
            zero = d_encoded_layers.new_zeros(64-d_encoded_layers.size(0),d_encoded_layers.size(1), d_encoded_layers.size(-1))
            d_encoded_layers1 = torch.cat([d_encoded_layers, zero], dim=0)
            # print("ddddd",d_encoded_layers1.shape)
            # print("a1111", a1.shape)
        if p_encoded_layers.size(0) < 128:
            # print("ddddd", d_encoded_layers.shape)
            zero = p_encoded_layers.new_zeros(64 - p_encoded_layers.size(0), p_encoded_layers.size(1),
                                              p_encoded_layers.size(-1))
            p_encoded_layers1 = torch.cat([p_encoded_layers, zero], dim=0)

        d_aug = torch.cat([d_encoded_layers1, a1], dim=1)
        p_aug = torch.cat([p_encoded_layers1, a2], dim=1)
        #print(d_aug .shape)
        #print(p_aug.shape)
        # print("222222",d_aug.size())
        # print("3333333333",p_aug.size())
        # d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1)  # repeat along protein size
        # p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.max_d, 1, 1)  # repeat along drug size
        i = torch.matmul(d_aug, p_aug.permute(0, 2, 1))
        #print(i.shape)# interaction
        d_aug_prime = torch.matmul(i, p_aug)
        p_aug_prime = torch.matmul(i.permute(0, 2, 1), d_aug)
        #print(d_aug_prime .shape)
        #print(p_aug_prime .shape)
        # print(d_aug_prime.size())
        # print(p_aug_prime.size())
        solute_features = torch.cat((d_aug, d_aug_prime), dim=2)
        solvent_features = torch.cat((p_aug, p_aug_prime), dim=2)
        #print(solute_features.shape)
        #print(solvent_features .shape)
        # print("aaaaaaaaaaaa", solute_features.shape)
        # print(solvent_features.shape)
        solute_features = torch.sum(solute_features, dim=2)
        solvent_features = torch.sum(solvent_features, dim=2)
        #print(solute_features.shape)
        #print(solvent_features.shape)
        # print("bbbbbbbbbb", solute_features.shape)
        # print(solvent_features.shape)
        final_features = torch.cat((solute_features, solvent_features), 1)
        #print(final_features.shape)
        #score = self.decoder(final_features)
        #print(score.shape)
        # return score
        # # if x2.size(-1) > self.out_channels:
        # #     raise ValueError('The number of input channels is not allowed to '
        # #                      'be larger than the number of output channels')
        # #
        # # if x2.size(-1) < self.out_channels:
        # #     zero = x2.new_zeros(x2.size(0), self.out_channels - x2.size(-1))
        # #     x2 = torch.cat([x2, zero], dim=1)
        #
        # # for i in range(self.num_layers):
        # #     m = torch.matmul(x2, self.weight[i])
        # #     m = self.propagate(edge_index2, x=m,edge_attr=edge_attr2,size=None)
        # #     if self.use_resnet:
        # #         x2 = self.mlp_2(m + self.mlp_1(x2))
        # #     else:
        # #         x2 = self.rnn(m, x2)
        # # x2=global_mean_pool(x2, batch2)
        #
        # xall = torch.cat((x1, x2), 1)
        # # aa=xall
        # # aa=torch.matmul(xall,rela.T)
        # #print(xall.shape)
        scores = self.MLP(final_features, 1)  # self.lin1(xall)

        return scores, rels
        #,i,d_v,p_v,a11,a22

        #print(x1.shape)

        #return x

    def message(self, x_j, edge_attr):
        """
        edge_attr: tensor of size (n, e) - n is the number of edges, e is the
        total number of edge types
        """

        return x_j if edge_attr is None else \
            torch.bmm(
                torch.einsum("ab,bcd->acd", (edge_attr, self.edge_type_weight)),
                x_j.unsqueeze(-1)).squeeze() + \
            self.edge_type_bias.repeat(x_j.size(0), 1)
    def get_weight(self, triples,zhongzi):
        #print(triples)
        h_data, _, _= triples
        x1, edge_index1, edge_attr1, batch1 = h_data.x, h_data.edge_index, h_data.edge_attr, h_data.batch

        if x1.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x1.size(-1) < self.out_channels:
            zero = x1.new_zeros(x1.size(0), self.out_channels - x1.size(-1))
            x1 = torch.cat([x1, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x1, self.weight[i])
            m = self.propagate(edge_index1, x=m, edge_attr=edge_attr1, size=None)
            if self.use_resnet:
                x1 = self.mlp_2(m + self.mlp_1(x1))
            else:
                x1 = self.rnn(m, x1)
        x1 = global_mean_pool(x1, batch1)

        xall = x1#self.out(x)

        #repr_h = torch.stack(xall,dim=-2)
        repr_h=xall.view((572,-1))
        #print(repr_h.shape)
        np.save('drug_emb_trimnetnew'+str(zhongzi)+'junheng-bampnn.npy',repr_h.cpu())

        kge_heads = repr_h


        return kge_heads
    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            # if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        # if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states