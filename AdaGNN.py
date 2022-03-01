import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F



class AdagnnWith(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(AdagnnWith, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.learnable_diag_1 = Parameter(torch.FloatTensor(in_features))
        if bias: 
            self.bias = Parameter(torch.FloatTensor(out_features))

        else:
            self.resister_parameters('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.normal_(self.learnable_diag_1, mean=0, std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forword(self, input, adj):
        e1 = torch.matmul(adj, input)
        alpha = torch.diag(self.learnable_diag_1)
        e2 = torch.matmul(e1, alpha + torch.eye(self.in_features, self.in_features).cuda())
        e4 = torch.sub(input, e2)
        e5 = torch.matmul(e4, self.weight)
        output = e5

        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AdagnnWithout(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(AdagnnWith, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.learnable_diag_1 = Parameter(torch.FloatTensor(in_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))

        else:
            self.resister_parameters('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.learnable_diag_1, mean=0, std=0.01)
    
    def forword(self, input, adj):
        e1 = torch.matmul(adj, input)
        alpha = torch.diag(self.learnable_diag_1)
        e2 = torch.matmul(e1, alpha + torch.eye(self.in_features, self.in_features).cuda())
        e4 = torch.sub(input, e2)
        # e5 = torch.matmul(e4, self.weight)
        output = e4

        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

        
class Adagnn(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, nlayer=2):
        super(Adagnn, self).__init__()

        self.should_train_1 = AdagnnWith(nfeat, nhid)
        assert nlayer - 2 >= 0
        self.hidden_layers = nn.ModuleList([
            AdagnnWithout(nhid, nhid, bias=False)
            for i in range(nlayer - 2)
        ])
        self.should_train_2 = AdagnnWith(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, l_sym):


        x = F.relu(self.should_train_1(x, l_sym))  # .relu
        x = F.dropout(x, self.dropout, training=self.training)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, l_sym)
            x = self.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.should_train_2(x, l_sym)  # + res1
        return F.log_softmax(x, dim=1)