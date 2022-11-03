import torch
from torch import nn
from torch.nn import functional as F
from core.model import BaseModel

''' 
    The CNN_DropOut model is taken from FedML repository. For more information regarding this model, 
    please refer to https://github.com/FedML-AI/FedML/blob/master/python/fedml/model/nlp/rnn.py.

'''

class nlp_rnn_fedshakespeare(nn.Module):
    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(nlp_rnn_fedshakespeare, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        # output = self.fc(final_hidden_state)
        # For fed_shakespeare
        output = self.fc(lstm_out[:, :])
        output = torch.transpose(output, 1, 2)
        return output

class RNN(BaseModel):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()
        self.net = nlp_rnn_fedshakespeare()

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, target = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(x)
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
        return criterion(output, target.long())

    def inference(self, input):
        '''Performs forward step and computes metrics'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, target = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(x)
        n_samples = x.shape[0]
        
        pred = torch.argmax(output, dim=1)
        mask = (target != 0)
        accuracy = torch.sum((pred[mask] == target[mask]).float()).item()
        accuracy = accuracy/mask.sum()

        return {'output':output, 'acc': accuracy, 'batch_size': n_samples} 


        