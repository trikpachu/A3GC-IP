import torch.nn
from torch.nn.functional import relu
from config import *
import articulate as art
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
import numpy as np

def reverse_not_working(lst: List[Tensor]) -> List[Tensor]:
    return lst[::-1]

def reverse(lst: List[Tensor]) -> List[Tensor]:
    new_lst = []
    for i in range(len(lst)-1, -1, -1):
        new_lst.append(lst[i])
    return new_lst

class DIP(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(DIP, self).__init__()
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        x_ = self.dropout(x)
        x_ = self.linear1(x_)
        x_ = relu(x_)
        x_, h = self.rnn(x_, h)
        return self.linear2(x_), h

class AAGC(torch.nn.Module):
    '''
    First X*W, then A*X. A learnable.
    '''
    def __init__(self, units_in, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0):
        super(AAGC, self).__init__()
        if activation_fn == 'linear':
            self.activation_fn = torch.nn.Identity()
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            raise ValueError('only support linear and tanh activations for now')

        self.dropout = torch.nn.Dropout(dropout)

        self.gcn_kernel = Parameter(torch.zeros((units_out, units_in), dtype=torch.float32))
        self.adj = Parameter(adjacency_matrix.t())
        self.gcn_bias = Parameter(torch.zeros(units_out, dtype=torch.float32))

        torch.nn.init.xavier_uniform_(self.gcn_kernel)

    def forward(self, input):
        x = self.dropout(input)
        x = torch.einsum('bsnf,nm->bsmf', x, self.adj.t())
        x = torch.matmul(x, self.gcn_kernel.t()) + self.gcn_bias
        x = self.activation_fn(x)
        return x

class AAGC_LSTM_cell(jit.ScriptModule):
    '''
    LSTM like cell with adjacency activation for every usage of the input term
    '''
    def __init__(self, units_in, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0, recurrent_dropout=0.0):
        super(AAGC_LSTM_cell, self).__init__()
        if activation_fn == 'linear':
            self.activation_fn = torch.nn.Identity()
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            raise ValueError('only support linear and tanh activations for now')

        self.dropout = torch.nn.Dropout(dropout)
        self.recurrent_dropout = torch.nn.Dropout(recurrent_dropout)

        self.gcn_kernel_i = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_f = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_c = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_o = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.adjacency_i = Parameter(adjacency_matrix.t())
        self.adjacency_f = Parameter(adjacency_matrix.t())
        self.adjacency_c = Parameter(adjacency_matrix.t())
        self.adjacency_o = Parameter(adjacency_matrix.t())
        self.gcn_bias_i = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_f = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_c = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_o = Parameter(torch.zeros(units_out, dtype=torch.float32))

        torch.nn.init.xavier_uniform_(self.gcn_kernel_i)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_f)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_c)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_o)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        x = self.dropout(input)
        hx = self.recurrent_dropout(hx)
        x_s = torch.cat((x, hx), dim=2)
        x_i = torch.einsum('bnf,nm->bmf', x_s, self.adjacency_i.t())
        x_i = torch.matmul(x_i, self.gcn_kernel_i.t()) + self.gcn_bias_i
        x_i = torch.sigmoid(x_i)
        x_f = torch.einsum('bnf,nm->bmf', x_s, self.adjacency_f.t())
        x_f = torch.matmul(x_f, self.gcn_kernel_f.t()) + self.gcn_bias_f
        x_f = torch.sigmoid(x_f)
        x_c = torch.einsum('bnf,nm->bmf', x_s, self.adjacency_c.t())
        x_c = torch.matmul(x_c, self.gcn_kernel_c.t()) + self.gcn_bias_c
        x_c = torch.tanh(x_c)
        x_o = torch.einsum('bnf,nm->bmf', x_s, self.adjacency_o.t())
        x_o = torch.matmul(x_o, self.gcn_kernel_o.t()) + self.gcn_bias_o
        x_o = torch.sigmoid(x_o)

        cy = (x_f * cx) + (x_i * x_c) 
        
        hy = x_o * torch.tanh(cy)
        hyo = self.activation_fn(hy)

        return hyo, (hy, cy)

class A3GC_LSTM_cell(jit.ScriptModule):
    '''
    LSTM like cell with adjacency activation for every usage of the input term
    '''
    def __init__(self, units_in, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0, recurrent_dropout=0.0):
        super(A3GC_LSTM_cell, self).__init__()
        if activation_fn == 'linear':
            self.activation_fn = torch.nn.Identity()
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            raise ValueError('only support linear and tanh activations for now')

        num_nodes = adjacency_matrix.shape[-1]
        assert num_nodes == 15

        self.dropout = torch.nn.Dropout(dropout)
        self.recurrent_dropout = torch.nn.Dropout(recurrent_dropout)

        self.gcn_kernel_i = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_f = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_c = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_o = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.adjacency_i = Parameter(adjacency_matrix.t())
        self.adjacency_f = Parameter(adjacency_matrix.t())
        self.adjacency_c = Parameter(adjacency_matrix.t())
        self.adjacency_o = Parameter(adjacency_matrix.t())
        self.gcn_bias_i = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_f = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_c = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_o = Parameter(torch.zeros(units_out, dtype=torch.float32))

        self.attention_w = Parameter(torch.zeros((units_out, units_out), dtype=torch.float32))
        self.attention_wq = Parameter(torch.zeros((units_out, units_out), dtype=torch.float32))
        self.attention_wh = Parameter(torch.zeros((units_out, units_out), dtype=torch.float32))
        self.attention_u = Parameter(torch.zeros((1, units_out), dtype=torch.float32))
        self.attention_bs = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.attention_bu = Parameter(torch.zeros(num_nodes, dtype=torch.float32))

        torch.nn.init.xavier_uniform_(self.gcn_kernel_i)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_f)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_c)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_o)

        torch.nn.init.xavier_uniform_(self.attention_w)
        torch.nn.init.xavier_uniform_(self.attention_wq)
        torch.nn.init.xavier_uniform_(self.attention_wh)
        torch.nn.init.xavier_uniform_(self.attention_u)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        x = self.dropout(input)
        hx = self.recurrent_dropout(hx)
        x_s = torch.cat((x, hx), dim=2)
        x_i = torch.einsum('bnf,nm->bmf', x_s, self.adjacency_i.t())
        x_i = torch.matmul(x_i, self.gcn_kernel_i.t()) + self.gcn_bias_i
        x_i = torch.sigmoid(x_i)
        x_f = torch.einsum('bnf,nm->bmf', x_s, self.adjacency_f.t())
        x_f = torch.matmul(x_f, self.gcn_kernel_f.t()) + self.gcn_bias_f
        x_f = torch.sigmoid(x_f)
        x_c = torch.einsum('bnf,nm->bmf', x_s, self.adjacency_c.t())
        x_c = torch.matmul(x_c, self.gcn_kernel_c.t()) + self.gcn_bias_c
        x_c = torch.tanh(x_c)
        x_o = torch.einsum('bnf,nm->bmf', x_s, self.adjacency_o.t())
        x_o = torch.matmul(x_o, self.gcn_kernel_o.t()) + self.gcn_bias_o
        x_o = torch.sigmoid(x_o)

        cy = (x_f * cx) + (x_i * x_c) 
        
        hy = x_o * torch.tanh(cy)

        q_t = torch.relu(torch.sum(torch.matmul(hy, self.attention_w.t()), dim=1, keepdim=True)) # shape [batch, 1, units]
        wh_ht = torch.matmul(hy, self.attention_wh.t()) # shape [batch, nodes, units]
        wq_qt = torch.matmul(q_t, self.attention_wq.t()) # shape [batch, 1, units]
        qht = wh_ht + wq_qt  # shape [batch, nodes, units]
        qht = qht + self.attention_bs # shape [batch, nodes, units]
        qht = torch.tanh(qht)
        a_t = torch.matmul(qht, self.attention_u.t()) # shape [batch, nodes, 1]
        a_t = torch.squeeze(a_t, dim=2) # shape [batch, nodes]
        a_t = a_t + self.attention_bu # shape [batch, nodes]
        a_t = torch.unsqueeze(a_t, dim=-1)
        a_t = torch.sigmoid(a_t)

        hy_att = hy * a_t
        hy = hy + hy_att

        hyo = self.activation_fn(hy)

        return hyo, (hy, cy)

class AGC_LSTM_cell(jit.ScriptModule):
    '''
    LSTM like cell with adjacency activation for every usage of the input term
    '''
    def __init__(self, units_in, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0, recurrent_dropout=0.0):
        super(AGC_LSTM_cell, self).__init__()
        if activation_fn == 'linear':
            self.activation_fn = torch.nn.Identity()
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            raise ValueError('only support linear and tanh activations for now')

        num_nodes = adjacency_matrix.shape[-1]
        assert num_nodes == 15

        self.dropout = torch.nn.Dropout(dropout)
        self.recurrent_dropout = torch.nn.Dropout(recurrent_dropout)

        self.adjacency = Parameter(adjacency_matrix.t(), requires_grad=False)
        self.gcn_kernel_i = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_f = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_c = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_kernel_o = Parameter(torch.zeros((units_out, units_in+units_out), dtype=torch.float32))
        self.gcn_bias_i = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_f = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_c = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.gcn_bias_o = Parameter(torch.zeros(units_out, dtype=torch.float32))

        self.attention_w = Parameter(torch.zeros((units_out, units_out), dtype=torch.float32))
        self.attention_wq = Parameter(torch.zeros((units_out, units_out), dtype=torch.float32))
        self.attention_wh = Parameter(torch.zeros((units_out, units_out), dtype=torch.float32))
        self.attention_u = Parameter(torch.zeros((1, units_out), dtype=torch.float32))
        self.attention_bs = Parameter(torch.zeros(units_out, dtype=torch.float32))
        self.attention_bu = Parameter(torch.zeros(num_nodes, dtype=torch.float32))

        torch.nn.init.xavier_uniform_(self.gcn_kernel_i)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_f)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_c)
        torch.nn.init.xavier_uniform_(self.gcn_kernel_o)

        torch.nn.init.xavier_uniform_(self.attention_w)
        torch.nn.init.xavier_uniform_(self.attention_wq)
        torch.nn.init.xavier_uniform_(self.attention_wh)
        torch.nn.init.xavier_uniform_(self.attention_u)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        x = self.dropout(input)
        hx = self.recurrent_dropout(hx)
        x_s = torch.cat((x, hx), dim=2)
        x_s = torch.einsum('nm,bmf->bnf', self.adjacency.t(), x_s)

        x_i = torch.matmul(x_s, self.gcn_kernel_i.t()) + self.gcn_bias_i
        x_i = torch.sigmoid(x_i)
        x_f = torch.matmul(x_s, self.gcn_kernel_f.t()) + self.gcn_bias_f
        x_f = torch.sigmoid(x_f)
        x_c = torch.matmul(x_s, self.gcn_kernel_c.t()) + self.gcn_bias_c
        x_c = torch.tanh(x_c)
        x_o = torch.matmul(x_s, self.gcn_kernel_o.t()) + self.gcn_bias_o
        x_o = torch.sigmoid(x_o)

        cy = (x_f * cx) + (x_i * x_c) 
        
        hy = x_o * torch.tanh(cy)

        q_t = torch.relu(torch.sum(torch.matmul(hy, self.attention_w.t()), dim=1, keepdim=True)) # shape [batch, 1, units]
        wh_ht = torch.matmul(hy, self.attention_wh.t()) # shape [batch, nodes, units]
        wq_qt = torch.matmul(q_t, self.attention_wq.t()) # shape [batch, 1, units]
        qht = wh_ht + wq_qt  # shape [batch, nodes, units]
        qht = qht + self.attention_bs # shape [batch, nodes, units]
        qht = torch.tanh(qht)
        a_t = torch.matmul(qht, self.attention_u.t()) # shape [batch, nodes, 1]
        a_t = torch.squeeze(a_t, dim=2) # shape [batch, nodes]
        a_t = a_t + self.attention_bu # shape [batch, nodes]
        a_t = torch.unsqueeze(a_t, dim=-1)
        a_t = torch.sigmoid(a_t)

        hy_att = a_t * hy
        hy = hy + hy_att

        hyo = self.activation_fn(hy)

        return hyo, (hy, cy)

class G_GRU_cell(jit.ScriptModule):
    '''
    LSTM like cell with adjacency activation for every usage of the input term
    '''
    def __init__(self, units_in, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0, recurrent_dropout=0.0):
        super(G_GRU_cell, self).__init__()
        if activation_fn == 'linear':
            self.activation_fn = torch.nn.Identity()
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            raise ValueError('only support linear and tanh activations for now')

        num_nodes = adjacency_matrix.shape[-1]
        assert num_nodes == 15

        #self.dropout = torch.nn.Dropout(dropout)
        #self.recurrent_dropout = torch.nn.Dropout(recurrent_dropout)

        self.a = Parameter(adjacency_matrix, requires_grad=False)

        self.dense_r_in = torch.nn.Linear(units_in, units_out, bias=True)
        self.dense_u_in = torch.nn.Linear(units_in, units_out, bias=True)
        self.dense_c_in = torch.nn.Linear(units_in, units_out, bias=True)
        self.dense_r_hid = torch.nn.Linear(units_out, units_out, bias=False)
        self.dense_u_hid = torch.nn.Linear(units_out, units_out, bias=False)
        self.dense_c_hid = torch.nn.Linear(units_out, units_out, bias=False)


        self.adjacency = Parameter(adjacency_matrix.t())
        self.gcn_kernel = Parameter(torch.zeros((units_out, units_out), dtype=torch.float32))
        #self.adj_mul = Parameter(torch.ones((num_nodes, num_nodes), dtype=torch.float32))
        #self.adj_add = Parameter(torch.zeros((num_nodes, num_nodes), dtype=torch.float32))

        torch.nn.init.xavier_uniform_(self.adjacency)
        torch.nn.init.xavier_uniform_(self.gcn_kernel)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        h = state
        #adjacency = self.a * self.adj_mul + self.adj_add
        #if step[0]<10:
        msg = torch.matmul(h, self.gcn_kernel.t())
        msg = torch.einsum('nm,bmf->bnf', self.adjacency.t(), msg)
        #else:
        #    msg = h

        r_in = self.dense_r_in(input)
        u_in = self.dense_u_in(input)
        c_in = self.dense_c_in(input)

        r_hid = self.dense_r_hid(msg)
        u_hid = self.dense_u_hid(msg)
        c_hid = self.dense_c_hid(msg)

        r = torch.sigmoid(r_in + r_hid)
        u = torch.sigmoid(u_in + u_hid)
        c = torch.tanh(c_in + r * c_hid)

        h = u * h + (1-u) * c
        
        #step = step + 1

        return h, h

class AAGC_LSTM(jit.ScriptModule):
    '''
    implements the AAGC_LSTM layer. Note that the input is expected to have the form (seq_len, batch, nodes, features)
    '''
    def __init__(self, *cell_args, **cell_kwargs):
        super(AAGC_LSTM, self).__init__()
        self.cell = AAGC_LSTM_cell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

class ReverseAAGC_LSTM(jit.ScriptModule):
    def __init__(self, *cell_args, **cell_kwargs):
        super(ReverseAAGC_LSTM, self).__init__()
        self.cell = AAGC_LSTM_cell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        #inputs = reverse(input.unbind(0))
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)-1, -1, -1):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state

class BiAAGC_LSTM(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, *cell_args, **cell_kwargs):
        super(BiAAGC_LSTM, self).__init__()
        self.directions = nn.ModuleList([
            AAGC_LSTM(*cell_args, **cell_kwargs),
            ReverseAAGC_LSTM(*cell_args, **cell_kwargs),
        ])

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        input = torch.transpose(input, 0, 1)
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [torch.transpose(out, 0, 1)]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states

class A3GC_LSTM(jit.ScriptModule):
    '''
    implements the AAGC_LSTM layer. Note that the input is expected to have the form (seq_len, batch, nodes, features)
    '''
    def __init__(self, *cell_args, **cell_kwargs):
        super(A3GC_LSTM, self).__init__()
        self.cell = A3GC_LSTM_cell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

class ReverseA3GC_LSTM(jit.ScriptModule):
    def __init__(self, *cell_args, **cell_kwargs):
        super(ReverseA3GC_LSTM, self).__init__()
        self.cell = A3GC_LSTM_cell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        #inputs = reverse(input.unbind(0))
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)-1, -1, -1):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state

class BiA3GC_LSTM(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, *cell_args, **cell_kwargs):
        super(BiA3GC_LSTM, self).__init__()
        self.directions = nn.ModuleList([
            A3GC_LSTM(*cell_args, **cell_kwargs),
            ReverseA3GC_LSTM(*cell_args, **cell_kwargs),
        ])

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        input = torch.transpose(input, 0, 1)
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [torch.transpose(out, 0, 1)]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states

class AGC_LSTM(jit.ScriptModule):
    '''
    implements the AAGC_LSTM layer. Note that the input is expected to have the form (seq_len, batch, nodes, features)
    '''
    def __init__(self, *cell_args, **cell_kwargs):
        super(AGC_LSTM, self).__init__()
        self.cell = AGC_LSTM_cell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

class ReverseAGC_LSTM(jit.ScriptModule):
    def __init__(self, *cell_args, **cell_kwargs):
        super(ReverseAGC_LSTM, self).__init__()
        self.cell = AGC_LSTM_cell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        #inputs = reverse(input.unbind(0))
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)-1, -1, -1):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state

class BiAGC_LSTM(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, *cell_args, **cell_kwargs):
        super(BiAGC_LSTM, self).__init__()
        self.directions = nn.ModuleList([
            AGC_LSTM(*cell_args, **cell_kwargs),
            ReverseAGC_LSTM(*cell_args, **cell_kwargs),
        ])

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        input = torch.transpose(input, 0, 1)
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [torch.transpose(out, 0, 1)]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states

class G_GRU(jit.ScriptModule):
    '''
    implements the AAGC_LSTM layer. Note that the input is expected to have the form (seq_len, batch, nodes, features)
    '''
    def __init__(self, *cell_args, **cell_kwargs):
        super(G_GRU, self).__init__()
        self.cell = G_GRU_cell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

class ReverseG_GRU(jit.ScriptModule):
    def __init__(self, *cell_args, **cell_kwargs):
        super(ReverseG_GRU, self).__init__()
        self.cell = G_GRU_cell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        #inputs = reverse(input.unbind(0))
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)-1, -1, -1):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state

class BiG_GRU(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, *cell_args, **cell_kwargs):
        super(BiG_GRU, self).__init__()
        self.directions = nn.ModuleList([
            G_GRU(*cell_args, **cell_kwargs),
            ReverseG_GRU(*cell_args, **cell_kwargs),
        ])

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        input = torch.transpose(input, 0, 1)
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tensor], [])
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [torch.transpose(out, 0, 1)]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


class AAGC_net(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, units_in, units_out, units_hidden, adjacency_matrix, linear_dropout=0.2, dropout=0.3, recurrent_dropout=0.3):
        super(AAGC_net, self).__init__()
        self.units_hidden = units_hidden
        self.linear_in = AAGC(units_in, units_hidden, adjacency_matrix, activation_fn='linear', dropout=linear_dropout)
        self.rnn1 = BiAAGC_LSTM(units_hidden, units_hidden, adjacency_matrix, activation_fn='tanh', dropout=dropout, recurrent_dropout=dropout)
        self.rnn2 = BiAAGC_LSTM(units_hidden*2, units_hidden, adjacency_matrix, activation_fn='tanh', dropout=dropout, recurrent_dropout=dropout)
        self.linear_out = AAGC(units_hidden*2, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0)

    def forward(self, x, h=None):
        if h is None:
            h_zeros = torch.zeros(x.size(0), 15, self.units_hidden,
                                  dtype=x.dtype, device=x.device)
            c_zeros = torch.zeros(x.size(0), 15, self.units_hidden,
                                  dtype=x.dtype, device=x.device)
            h = [(h_zeros, c_zeros), (h_zeros.clone(), c_zeros.clone())]
        x = self.linear_in(x)
        x = torch.relu(x)
        x, h = self.rnn1(x, h)
        x, h = self.rnn2(x, h)
        x = self.linear_out(x)
        return x, h

class A3GC_net(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, units_in, units_out, units_hidden, adjacency_matrix, linear_dropout=0.2, dropout=0.3, recurrent_dropout=0.3):
        super(A3GC_net, self).__init__()
        self.units_hidden = units_hidden
        self.linear_in = AAGC(units_in, units_hidden, adjacency_matrix, activation_fn='linear', dropout=linear_dropout)
        self.rnn1 = BiA3GC_LSTM(units_hidden, units_hidden, adjacency_matrix, activation_fn='tanh', dropout=dropout, recurrent_dropout=dropout)
        self.rnn2 = BiA3GC_LSTM(units_hidden*2, units_hidden, adjacency_matrix, activation_fn='tanh', dropout=dropout, recurrent_dropout=dropout)
        self.linear_out = AAGC(units_hidden*2, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0)

    def forward(self, x, h=None):
        if h is None:
            h_zeros = torch.zeros(x.size(0), 15, self.units_hidden,
                                  dtype=x.dtype, device=x.device)
            c_zeros = torch.zeros(x.size(0), 15, self.units_hidden,
                                  dtype=x.dtype, device=x.device)
            h = [(h_zeros, c_zeros), (h_zeros.clone(), c_zeros.clone())]
        x = self.linear_in(x)
        x = torch.relu(x)
        x, h = self.rnn1(x, h)
        x, h = self.rnn2(x, h)
        x = self.linear_out(x)
        return x, h

class AGC_net(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, units_in, units_out, units_hidden, adjacency_matrix, linear_dropout=0.2, dropout=0.3, recurrent_dropout=0.3):
        super(AGC_net, self).__init__()
        self.units_hidden = units_hidden
        self.linear_in = AAGC(units_in, units_hidden, adjacency_matrix, activation_fn='linear', dropout=linear_dropout)
        self.rnn1 = BiAGC_LSTM(units_hidden, units_hidden, adjacency_matrix, activation_fn='tanh', dropout=dropout, recurrent_dropout=dropout)
        self.rnn2 = BiAGC_LSTM(units_hidden*2, units_hidden, adjacency_matrix, activation_fn='tanh', dropout=dropout, recurrent_dropout=dropout)
        self.linear_out = AAGC(units_hidden*2, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0)

    def forward(self, x, h=None):
        if h is None:
            h_zeros = torch.zeros(x.size(0), 15, self.units_hidden,
                                  dtype=x.dtype, device=x.device)
            c_zeros = torch.zeros(x.size(0), 15, self.units_hidden,
                                  dtype=x.dtype, device=x.device)
            h = [(h_zeros, c_zeros), (h_zeros.clone(), c_zeros.clone())]
        x = self.linear_in(x)
        x = torch.relu(x)
        x, h = self.rnn1(x, h)
        x, h = self.rnn2(x, h)
        x = self.linear_out(x)
        return x, h

class G_GRU_net(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, units_in, units_out, units_hidden, adjacency_matrix, linear_dropout=0.2, dropout=0.3, recurrent_dropout=0.3):
        super(G_GRU_net, self).__init__()
        self.units_hidden = units_hidden
        self.linear_in = AAGC(units_in, units_hidden, adjacency_matrix, activation_fn='linear', dropout=linear_dropout)
        self.rnn1 = BiG_GRU(units_hidden, units_hidden, adjacency_matrix, activation_fn='tanh', dropout=dropout, recurrent_dropout=dropout)
        self.rnn2 = BiG_GRU(units_hidden*2, units_hidden, adjacency_matrix, activation_fn='tanh', dropout=dropout, recurrent_dropout=dropout)
        self.linear_out = AAGC(units_hidden*2, units_out, adjacency_matrix, activation_fn='linear', dropout=0.0)

    def forward(self, x, h=None):
        if h is None:
            h_zeros = torch.zeros(x.size(0), 15, self.units_hidden,
                                  dtype=x.dtype, device=x.device)
            h = [h_zeros, h_zeros.clone()]
        x = self.linear_in(x)
        x = torch.relu(x)
        x, h = self.rnn1(x, h)
        x, h = self.rnn2(x, h)
        x = self.linear_out(x)
        return x, h

class PoseNet(torch.nn.Module):
    r"""
    Whole pipeline for pose estimation using A2GC.
    """
    def __init__(self, input_size=12, rotsize=9, adjacency=None, device=torch.device("cpu"), n_hidden=256):
        super().__init__()
        
        self.rotsize = rotsize
        self.adjacency = adjacency

        self.pose_net = AAGC_net(input_size, rotsize, n_hidden, self.adjacency)

        self.m = art.ParametricModel(paths.male_smpl_file, device=device)

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R

        # variable
        self.rnn_state = None
        self.imu = None
        self.reset()

    def _reduced_glb_6d_to_full_local_mat(self, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose

    def _reduced_glb_to_full_local_mat(self, glb_reduced_pose, filter=None):
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose
        
    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        

    def forward(self, imu, rnn_state=None):
        global_reduced_pose, rnn_state = self.pose_net.forward(imu, rnn_state)
        return global_reduced_pose, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu, rnn_state=None):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, _ = self.forward(imu, rnn_state) 

        if self.rotsize == 6:
            pose = self._reduced_glb_6d_to_full_local_mat(global_reduced_pose)
        elif self.rotsize == 9:
            pose = self._reduced_glb_to_full_local_mat(global_reduced_pose.view(-1, 15, 3, 3))
        else: 
            pose = global_reduced_pose
        return pose, None


class PoseNet3(torch.nn.Module):
    r"""
    Whole pipeline for pose estimation using A3GC.
    """
    def __init__(self, input_size=12, rotsize=9, adjacency=None, device=torch.device("cpu"), n_hidden=256):
        super().__init__()
        
        self.rotsize = rotsize
        self.adjacency = adjacency

        self.pose_net = A3GC_net(input_size, rotsize, n_hidden, self.adjacency)

        self.m = art.ParametricModel(paths.male_smpl_file, device=device)

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R

        # variable
        self.rnn_state = None
        self.imu = None
        self.reset()

    def _reduced_glb_6d_to_full_local_mat(self, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose

    def _reduced_glb_to_full_local_mat(self, glb_reduced_pose, filter=None):
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose
        
    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        

    def forward(self, imu, rnn_state=None):
        global_reduced_pose, rnn_state = self.pose_net.forward(imu, rnn_state)
        return global_reduced_pose, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu, rnn_state=None):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, _ = self.forward(imu, rnn_state) 

        if self.rotsize == 6:
            pose = self._reduced_glb_6d_to_full_local_mat(global_reduced_pose)
        elif self.rotsize == 9:
            pose = self._reduced_glb_to_full_local_mat(global_reduced_pose.view(-1, 15, 3, 3))
        else: 
            pose = global_reduced_pose
        return pose, None


class PoseNet_AGC(torch.nn.Module):
    r"""
    Whole pipeline for pose estimation using AGC for the recurent_layers (i.e. A3GC without adjacency adaptivity).
    """
    def __init__(self, input_size=12, rotsize=9, adjacency=None, device=torch.device("cpu"), n_hidden=256):
        super().__init__()
        
        self.rotsize = rotsize
        self.adjacency = adjacency

        self.pose_net = AGC_net(input_size, rotsize, n_hidden, self.adjacency)

        self.m = art.ParametricModel(paths.male_smpl_file, device=device)

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R

        # variable
        self.rnn_state = None
        self.imu = None
        self.reset()

    def _reduced_glb_6d_to_full_local_mat(self, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose

    def _reduced_glb_to_full_local_mat(self, glb_reduced_pose, filter=None):
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose
        
    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        

    def forward(self, imu, rnn_state=None):
        global_reduced_pose, rnn_state = self.pose_net.forward(imu, rnn_state)
        return global_reduced_pose, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu, rnn_state=None):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, _ = self.forward(imu, rnn_state) 

        if self.rotsize == 6:
            pose = self._reduced_glb_6d_to_full_local_mat(global_reduced_pose)
        elif self.rotsize == 9:
            pose = self._reduced_glb_to_full_local_mat(global_reduced_pose.view(-1, 15, 3, 3))
        else: 
            pose = global_reduced_pose
        return pose, None

class PoseNet_GGRU(torch.nn.Module):
    r"""
    Whole pipeline for pose estimation using G-GRU.
    """
    def __init__(self, input_size=12, rotsize=9, adjacency=None, device=torch.device("cpu"), n_hidden=256):
        super().__init__()
        
        self.rotsize = rotsize
        self.adjacency = adjacency

        self.pose_net = G_GRU_net(input_size, rotsize, n_hidden, self.adjacency)

        self.m = art.ParametricModel(paths.male_smpl_file, device=device)

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R

        # variable
        self.rnn_state = None
        self.imu = None
        self.reset()

    def _reduced_glb_6d_to_full_local_mat(self, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose

    def _reduced_glb_to_full_local_mat(self, glb_reduced_pose, filter=None):
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose
        
    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        

    def forward(self, imu, rnn_state=None):
        global_reduced_pose, rnn_state = self.pose_net.forward(imu, rnn_state)
        return global_reduced_pose, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu, rnn_state=None):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, _ = self.forward(imu, rnn_state) 

        if self.rotsize == 6:
            pose = self._reduced_glb_6d_to_full_local_mat(global_reduced_pose)
        elif self.rotsize == 9:
            pose = self._reduced_glb_to_full_local_mat(global_reduced_pose.view(-1, 15, 3, 3))
        else: 
            pose = global_reduced_pose
        return pose, None

class PoseNetTP(torch.nn.Module):
    r"""
    Whole pipeline for pose estimation using Transpose.
    """
    def __init__(self, input_size=12, n_output=9, adjacency=None, device=torch.device("cpu"), n_hidden=256):
        super().__init__()
        
        self.n_output = n_output

        self.pose_net = DIP(input_size, n_output, n_hidden)

        self.m = art.ParametricModel(paths.male_smpl_file, device=device)

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R


    def _reduced_glb_6d_to_full_local_mat(self, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose

    def _reduced_glb_to_full_local_mat(self, glb_reduced_pose, filter=None):
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose
        

    def forward(self, imu, rnn_state=None):
        global_reduced_pose, rnn_state = self.pose_net.forward(imu, rnn_state)
        return global_reduced_pose, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu, rnn_state=None):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, _ = self.forward(imu, rnn_state) 

        if self.n_output == 90:
            pose = self._reduced_glb_6d_to_full_local_mat(global_reduced_pose)
        elif self.n_output == 135:
            pose = self._reduced_glb_to_full_local_mat(global_reduced_pose.view(-1, 15, 3, 3))
        else: 
            pose = global_reduced_pose
        return pose, None

class PoseNetDIP(torch.nn.Module):
    r"""
    Whole pipeline for pose estimation using DIP.
    """
    def __init__(self, input_size=12, rotsize=9, adjacency=None, device=torch.device("cpu"), n_hidden=512):
        super().__init__()
        
        self.n_output = 15*rotsize

        self.pose_net = DIP(60, self.n_output, n_hidden)

        self.m = art.ParametricModel(paths.male_smpl_file, device=device)

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R


    def _reduced_glb_6d_to_full_local_mat(self, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose

    def _reduced_glb_to_full_local_mat(self, glb_reduced_pose, filter=None):
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        return pose
        

    def forward(self, imu, rnn_state=None):
        global_reduced_pose, rnn_state = self.pose_net.forward(imu, rnn_state)
        return global_reduced_pose, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu, rnn_state=None):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, _ = self.forward(imu, rnn_state) 

        if self.n_output == 90:
            pose = self._reduced_glb_6d_to_full_local_mat(global_reduced_pose)
        elif self.n_output == 135:
            pose = self._reduced_glb_to_full_local_mat(global_reduced_pose.view(-1, 15, 3, 3))
        else: 
            pose = global_reduced_pose
        return pose, None

class pose_loss(torch.nn.Module):
    def __init__(self, loss_weight=None):
        self.loss_weight = loss_weight

    def forward(self, pred, targ):
        smpl_loss = torch.square(targ-pred)
        if self.loss_weight is not None:
            smpl_loss = smpl_loss * self.loss_weight
        smpl_loss = torch.sum(smpl_loss, -1, keepdim=False) # sum of frame
        smpl_loss = torch.mean(smpl_loss) # mean of batch
        return smpl_loss
