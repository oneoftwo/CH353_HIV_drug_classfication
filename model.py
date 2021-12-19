import torch
from torch import nn

import config as CONFIG
'ff'

# Vanilla MPNN
class VanillaMPNN(nn.Module):

    def __init__(self, in_h_dim, in_e_dim, hid_h_dim, n_layer):
        super().__init__()
        self.embedding = nn.Linear(in_h_dim, hid_h_dim, bias=False)
        self.mpnn_layers = nn.ModuleList([MPNNLayer(hid_h_dim, in_e_dim, \
                use_dropout=CONFIG.use_dropout, use_activation=True) \
                for _ in range(n_layer)])
        self.mpnn_last = MPNNLayer(hid_h_dim, in_e_dim)
        self.fc = nn.Sequential(
                nn.Linear(hid_h_dim, hid_h_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(hid_h_dim, 2))
        self.act = nn.LeakyReLU()

    def forward(self, h, e, adj, atom_N):
        h = self.embedding(h)
        for layer in self.mpnn_layers:
            h, e, adj = layer(h, e, adj)
            # h_ = h_ + h
        h, e, adj = self.mpnn_last(h, e, adj)
        # readout, divide by atom number
        h = h.sum(dim=1) # h[b hd]
        # h = torch.div(h, atom_N.unsqueeze(1).repeat(1, h.size(1)))
        h = self.fc(h)
        return h


# MPNN Block with skip connection
class MPNNBlock(nn.Module):

    def __init__(self, h_dim, e_dim, use_dropout=False):
        super().__init__()
        self.mpnn_1 = MPNNLayer(h_dim, e_dim, use_dropout=False)
        self.mpnn_2 = MPNNLayer(h_dim, e_dim, use_dropout=False)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.3)

    def forward(self, h, e, adj): # adj[b N N]
        h_, e, adj = self.mpnn_1(h, e, adj)
        h__, e, adj = self.mpnn_2(h_, e, adj)
        h = h + h__
        if self.use_dropout:
            h = self.dropout(h)
        return h, e, adj


# MPNN single layer
class MPNNLayer(nn.Module):

    def __init__(self, h_dim, e_dim, use_dropout=False, use_activation=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_activation = use_activation
        self.h_dim = h_dim
        self.message_make = MPNNMakeMessage(h_dim, e_dim, h_dim)
        self.fc_h = nn.Linear(h_dim * 2, h_dim, bias=False)
        if use_dropout:
            self.dropout = nn.Dropout(0.3)
        if use_activation:
            self.act = nn.LeakyReLU()
        self.gru_cell = nn.GRUCell(h_dim, h_dim)

    def forward(self, h, e, adj): # adj[b N N]
        b, N = h.size(0), h.size(1)
        m = self.message_make(h, e, adj) # [b N N m]
        m = m.mean(2) # [b N nf]
        
        m_reshape = m.view(-1, self.h_dim)
        h_reshape = h.view(-1, self.h_dim)
        new_h = self.gru_cell(h_reshape, m_reshape)
        new_h = new_h.view(b, N, self.h_dim)
        '''
        cat = torch.cat([h, m], dim=2)
        new_h = self.fc_h(cat)
        '''

        if self.use_dropout:
            new_h = self.dropout(new_h)
        if self.use_activation:
            new_h = self.act(new_h)
        return new_h, e, adj
        

# layer for making message in MPNN
class MPNNMakeMessage(nn.Module):

    def __init__(self, h_dim, e_dim, m_dim):
        super().__init__()
        self.fc_m = nn.Linear(2 * h_dim + e_dim, m_dim)
        
    def forward(self, h, e, adj):
        b, N = h.size(0), h.size(1)
        h_repeat_1 = h.unsqueeze(1).repeat(1, N, 1, 1) # [b N N nf]
        h_repeat_2 = h.unsqueeze(2).repeat(1, 1, N, 1) # [b N N nf]
        cat = torch.cat([h_repeat_1, h_repeat_2, e], dim=3) # [b N N nf*2+ef]
        m = self.fc_m(cat)
        adj_ = adj.unsqueeze(3).repeat(1, 1, 1, m.size(3))
        m = m * adj_
        return m


#==============================================================================#
# sanity check ================================================================#
if __name__ == '__main__':
    h = torch.rand(4, 10, 13)
    e = torch.rand(4, 10, 10, 4)
    adj = torch.zeros(10, 10)
    adj[0,1] = 1
    adj = adj.unsqueeze(0).repeat(4, 1, 1)
       
    test = VanillaMPNN(13, 4, 64, 4)
    print(test)
    out = test(h, e, adj, 10)
    print(out)
#==============================================================================#s
