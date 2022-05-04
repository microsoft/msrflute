# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch as T
from torch import Tensor
from typing import List, Tuple

from core.model import BaseModel
from utils import softmax, to_device

class GRU2(T.nn.Module):
    def __init__(self, input_size, hidden_size, input_bias, hidden_bias):
        super(GRU2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_ih = T.nn.Linear(input_size, 3 * hidden_size, input_bias)
        self.w_hh = T.nn.Linear(hidden_size, 3 * hidden_size, hidden_bias)
        
    def _forward_cell(self, input : Tensor, hidden : Tensor) -> Tensor:
        g_i = self.w_ih(input)
        g_h = self.w_hh(hidden)
        i_r, i_i, i_n = g_i.chunk(3, 1)
        h_r, h_i, h_n = g_h.chunk(3, 1)
        reset_gate = T.sigmoid(i_r + h_r)
        input_gate = T.sigmoid(i_i + h_i)
        new_gate   = T.tanh(i_n + reset_gate * h_n)
        hy         = new_gate + input_gate * (hidden - new_gate)
        return hy
    
    def forward(self, input : Tensor) -> Tuple[Tensor, Tensor]:
        hiddens : List[Tensor] = [to_device(T.zeros((input.shape[0], self.hidden_size)))]
        for step in range(input.shape[1]):
            hidden = self._forward_cell(input[:, step], hiddens[-1])
            hiddens.append(hidden)
            
        return T.stack(hiddens, dim=1), hiddens[-1]
    

class Embedding(T.nn.Module):
    def __init__(self, vocab_size, embedding_size): 
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.table = T.nn.Parameter(T.zeros((vocab_size, embedding_size)))
        self.unembedding_bias = T.nn.Parameter(T.zeros(vocab_size))
        delta = (3 / self.table.shape[1]) ** 0.5
        T.nn.init.uniform_(self.table, -delta, delta)

    def forward(self, input : Tensor, embed : bool) -> Tensor:
        if embed:
            output = T.nn.functional.embedding(input, self.table)
        else:
            output = input @ self.table.t() + self.unembedding_bias
        return output
    

class GRU(BaseModel): #DLM_2_0
    def __init__(self, model_config, OOV_correct=False, dropout=0.0, topK_results=1, wantLogits=False, **kwargs):
        super(GRU, self).__init__()
        self.vocab_size = model_config['vocab_size']
        self.embedding_size = model_config['embed_dim']
        self.hidden_size = model_config['hidden_dim']
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        self.rnn = GRU2(self.embedding_size, self.hidden_size, True, True)
        self.squeeze = T.nn.Linear(self.hidden_size, self.embedding_size, bias=False)
        self.OOV_correct = OOV_correct
        self.topK_results = topK_results
        self.dropout=dropout
        self.wantLogits=wantLogits
        if self.dropout>0.0:
            self.drop_layer = T.nn.Dropout(p=self.dropout)

    def forward(self, input : T.Tensor) -> Tuple[Tensor, Tensor]:
        input = input['x'] if isinstance(input, dict) else input
        input = to_device(input)
        embedding = self.embedding(input, True)
        hiddens, state = self.rnn(embedding)
        if self.dropout>0.0:
            hiddens= self.drop_layer(hiddens)
        output = self.embedding(self.squeeze(hiddens), False)
        return output, state


    def loss(self, input : T.Tensor) -> T.Tensor:
        input = input['x'] if isinstance(input, dict) else input
        input = to_device(input)
        non_pad_mask = input >= 0
        input = input * non_pad_mask.long()
        non_pad_mask = non_pad_mask.view(-1)

        # Run the forward pass
        output, _ = self.forward(input[:, :-1])

        # Estimate the targets
        targets = input.view(-1)[non_pad_mask]
        preds   = output.view(-1, self.vocab_size)[non_pad_mask]

        # Estimate the loss
        return T.nn.functional.cross_entropy(preds, targets)


    def inference(self, input):
        input = input['x'] if isinstance(input, dict) else input
        input = to_device(input)
        non_pad_mask = input >= 0
        input = input * non_pad_mask.long()
        non_pad_mask = non_pad_mask.view(-1)
        output, _ = self.forward(input[:, :-1])

        # Apply mask to input/output
        targets = input.view(-1)[non_pad_mask]
        preds = output.view(-1, self.vocab_size)[non_pad_mask]

        # accuracy
        probs_topK, preds_topK = T.topk(preds, self.topK_results, sorted=True, dim=1)
        probs, preds = probs_topK[:,0], preds_topK[:,0]
        if self.OOV_correct:
            acc = preds.eq(targets).float().mean()
        else:
            valid = preds != 0  # reject oov predictions even if they're correct.
            acc = (preds.eq(targets) * valid).float().mean()

        if self.wantLogits:
            if 1:
                output=  {'probabilities': softmax(probs_topK.cpu().detach().numpy(), axis=1),
                               'predictions': preds_topK.cpu().detach().numpy(),
                               'labels': targets.cpu().detach().numpy()}
            else:
                output = {'probabilities': probs_topK.cpu().detach().numpy(),
                              'predictions': preds_topK.cpu().detach().numpy(),
                              'labels': targets.cpu().detach().numpy()}

        return {'output':output, 'acc': acc.item(), 'batch_size': input.shape[0]}



