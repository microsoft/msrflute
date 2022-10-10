import torch
import torch.nn as nn
import numpy as np

npratio = 4

''' 
    The FedNewsRec model is taken from FedNewsRec-EMNLP-Findings-2020 repository and ported to PyTorch
    framework to be compatible with FLUTE (https://github.com/simra/FedNewsRec#fednewsrec-emnlp-findings-2020). 
    For more information regarding this model, please refer to https://github.com/taoqi98/FedNewsRec.
'''
class AttentivePooling(nn.Module):
    def __init__(self, dim1: int, dim2: int):
        super(AttentivePooling, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

        self.dropout = nn.Dropout(0.2)
        self.dense  = nn.Linear(dim2, 200)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim=1)
       

    def forward(self, x):
        user_vecs = self.dropout(x)
        user_att = self.tanh(self.dense(user_vecs))
        user_att = self.dense2(user_att).squeeze(2)
        user_att = self.softmax(user_att)
        result = torch.einsum('ijk,ij->ik', user_vecs, user_att)        
        return result

    def fromTensorFlow(self, tfmodel):
        keras_weights = tfmodel.layers[1].get_weights()
        # print(keras_weights)
        self.dense.weight.data = torch.tensor(keras_weights[0]).transpose(0,1).cuda()
        self.dense.bias.data = torch.tensor(keras_weights[1]).cuda()

        keras_weights = tfmodel.layers[2].get_weights()
        # print(keras_weights)
        self.dense2.weight.data = torch.tensor(keras_weights[0]).transpose(0,1).cuda()
        self.dense2.bias.data = torch.tensor(keras_weights[1]).cuda()

class Attention(nn.Module):
 
    def __init__(self, input_dim, nb_head, size_per_head, **kwargs):
        super(Attention, self).__init__(**kwargs)
        #self.input_shape = input_shape
        self.input_dim = input_dim
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        #self.WQ = nn.Linear(self.input_shape[0][-1], self.output_dim, bias=False)
        #self.WK = nn.Linear(self.input_shape[1][-1], self.output_dim, bias=False)
        #self.WV = nn.Linear(self.input_shape[2][-1], self.output_dim, bias=False)
        self.WQ = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.WK = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.WV = nn.Linear(self.input_dim, self.output_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.WQ.weight, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform_(self.WK.weight, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform_(self.WV.weight, gain=np.sqrt(2))
        
    def fromTensorFlow(self, tf, criteria = lambda l: l.name.startswith('attention')):
        for l in tf.layers:
            print(l.name, l.output_shape)
            if criteria(l):
                weights = l.get_weights()
                self.WQ.weight.data = torch.tensor(weights[0].transpose()).cuda()
                self.WK.weight.data = torch.tensor(weights[1].transpose()).cuda()
                self.WV.weight.data = torch.tensor(weights[2].transpose()).cuda()
                

 
    def forward(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        
        Q_seq = self.WQ(Q_seq)
        Q_seq = torch.reshape(Q_seq, (-1, Q_seq.shape[1], self.nb_head, self.size_per_head))
        #Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        Q_seq = torch.transpose(Q_seq, 1, 2)
        K_seq = self.WK(K_seq)
        K_seq = torch.reshape(K_seq, (-1, K_seq.shape[1], self.nb_head, self.size_per_head))
        K_seq = torch.transpose(K_seq, 1, 2)
        V_seq = self.WV(V_seq)
        V_seq = torch.reshape(V_seq, (-1, V_seq.shape[1], self.nb_head, self.size_per_head))
        V_seq = torch.transpose(V_seq, 1, 2)
        
        #print('pt shapes')
        #print(Q_seq.shape, K_seq.shape)
        #A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = torch.einsum('ijkl,ijml->ijkm', Q_seq, K_seq) / self.size_per_head**0.5
        # A = K.permute_dimensions(A, (0,3,2,1))
        # A = self.Mask(A, V_len, 'add')
        # A = K.permute_dimensions(A, (0,3,2,1))
        A = torch.softmax(A, dim=-1)
        #输出并mask
        #O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = torch.einsum('ijkl,ijlm->ijkm', A, V_seq)
        #O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = torch.transpose(O_seq, 1,2)
        #O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = torch.reshape(O_seq, (-1, O_seq.shape[1], self.output_dim))
        #O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 





class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.permute(*self.dims)

class SwapTrailingAxes(nn.Module):
    def __init__(self):
        super(SwapTrailingAxes, self).__init__()
        
    def forward(self, x):        
        return x.transpose(-2, -1)

class DocEncoder(nn.Module):
    def __init__(self):        
        super(DocEncoder,self).__init__()
        self.phase1 = nn.Sequential(
            nn.Dropout(0.2),
            # TODO: why we need the SwapTrailingAxes here?
            SwapTrailingAxes(),            
            nn.Conv1d(300, 400, 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            # TODO: seems here we swap the dimension back. why?
            SwapTrailingAxes()
        )

        #self.attention = nn.MultiheadAttention(400, 20, batch_first=True)
        self.attention = Attention(400, 20, 20)
        # Pytorch MultiheadAttention has in_proj_weight of size (3*embed_dim, embed_dim)
        # Thus, we need to scale the xavier by sqrt(2)
        #torch.nn.init.xavier_uniform_(self.attention.in_proj_weight, gain=np.sqrt(2))
        self.phase2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            AttentivePooling(30,400)
        )

    def fromTensorFlow(self, tfDoc):
        print('td')
        for l in self.phase1:
            if 'conv' in l._get_name().lower():
                print('conv shape:',l.weight.data.shape, l.bias.data.shape)
            #print('\t',[p[0] for p in l.named_parameters()])
        
                for lt in tfDoc.layers:
                    print(lt.name, lt.output_shape)
                    if 'conv' in lt.name.lower():
                        weights = lt.get_weights()
                        l.weight.data = torch.tensor(weights[0]).transpose(0,2).cuda()
                        l.bias.data = torch.tensor(weights[1]).cuda()
                        #print(len(l.get_weights()), [p.shape for p in l.get_weights()])
                        break
                break

        #for lt in tfDoc.layers:
        #    print('tf2')
        #    print(lt.name, lt.output_shape)
        #    if 'attention' in lt.name:
        # TODO: we should just pass the specific layer
        self.attention.fromTensorFlow(tfDoc)

        print('phase2')
        for l in self.phase2:
            if 'attentive' in l._get_name().lower():
                for lt in tfDoc.layers:
                    print(lt.name)
                    if 'model' in lt.name.lower():
                        print('copying attentive pooling')
                        l.fromTensorFlow(lt)

        

    
    def forward(self, x):
        # print(x.shape)
        l_cnnt = self.phase1(x)
        # print('doc_encoder:phase1',l_cnnt.shape)
        l_cnnt = self.attention([l_cnnt]*3)
        # print('doc_encoder:attention', l_cnnt.shape)
        result = self.phase2(l_cnnt)
        # print('doc_encoder:phase2', result.shape)
        return result


class VecTail(nn.Module):
    def __init__(self, n):
        super(VecTail, self).__init__()
        self.n = n

    def forward(self, x):
        return x[:,-self.n:,:]

class UserEncoder(nn.Module):
    def __init__(self):        
        super(UserEncoder,self).__init__()
        # news_vecs_input = Input(shape=(50,400), dtype='float32')
        #self.dropout1 = nn.Dropout(0.2)
        #self.tail = VecTail(15)
        #self.gru = nn.GRU(400, 400)
        #self.attention = nn.MultiheadAttention(400, 20)
        #self.pool = AttentivePooling(50, 400)
        #self.attention2 = nn.MultiheadAttention(400, 20, batch_first=True)
        self.attention2 = Attention(400, 20, 20)
        #torch.nn.init.xavier_uniform_(self.attention2.in_proj_weight, gain=np.sqrt(2))
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = AttentivePooling(50, 400)
        self.tail2 = VecTail(20)
        #TODO: what is batch_first?
        self.gru2 = nn.GRU(400,400, bidirectional=False, batch_first=True)
        self.pool3 = AttentivePooling(2, 400)

    def forward(self, news_vecs_input):    
        #news_vecs =self.dropout1(news_vecs_input)
        #gru_input = self.tail(news_vecs)
        #vec1 = self.gru(gru_input)
        #vecs2 = self.attention(*[news_vecs]*3)
        #vec2 = self.pool(vecs2)
        # print('news_vecs_input', news_vecs_input.shape)
        user_vecs2 = self.attention2([news_vecs_input]*3)
        user_vecs2 = self.dropout2(user_vecs2)
        user_vec2 = self.pool2(user_vecs2)
        # print('pool2_user_vec2', user_vec2.shape)
        #user_vec2 = keras.layers.Reshape((1,400))(user_vec2)
        #user_vec2 = user_vec2.unsqueeze(1)

        user_vecs1 = self.tail2(news_vecs_input)
        # print('tail2_user_vecs1', user_vecs1.shape)
        self.gru2.flatten_parameters()
        user_vec1, _u_hidden = self.gru2(user_vecs1)
        # print('gru2_user_vec1', user_vec1.shape)
        # TODO: does this flatten the second dimension? print out the shape to check
        user_vec1 = user_vec1[:, -1, :]
        #user_vec1 = keras.layers.Reshape((1,400))(user_vec1)
        #user_vec1 = user_vec1.unsqueeze(1)
        
        user_vecs = torch.stack([user_vec1, user_vec2], dim=1) #keras.layers.Concatenate(axis=-2)([user_vec1,user_vec2])
        # print(user_vecs.shape)
        vec = self.pool3(user_vecs)
        # print(vec.shape)
        return vec

    def fromTensorFlow(self, tfU):
        for l in tfU.layers:
            print(l.name, l.output_shape)
            if l.name == 'model_1':
                self.pool2.fromTensorFlow(l)
            elif l.name == 'model_2':
                self.pool3.fromTensorFlow(l)
            elif l.name=='gru_1':                              
                print(len(l.get_weights()), [p.shape for p in l.get_weights()])
                weights = l.get_weights()
                for p in self.gru2.named_parameters():
                    s1 = p[1].data.shape
                    if p[0] == 'weight_ih_l0':                        
                        p[1].data = torch.tensor(weights[0]).transpose(0,1).contiguous().cuda()
                    elif p[0] == 'weight_hh_l0':
                        p[1].data = torch.tensor(weights[1]).transpose(0,1).contiguous().cuda()
                    elif p[0] == 'bias_ih_l0':
                        p[1].data = torch.tensor(weights[2]).cuda()
                    elif p[0] == 'bias_hh_l0':
                        p[1].data = torch.zeros(p[1].data.shape).cuda()
                    print(p[0], s1, p[1].shape)
        self.attention2.fromTensorFlow(tfU)
        # TODO: GRU
        
            


class TimeDistributed(nn.Module):    
    def __init__(self, module): #, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        # self.batch_first = batch_first

    def forward(self, x):
        # print('TimeDist_x',x.size())
        if len(x.size()) <= 2:
            return self.module(x)

        output = torch.tensor([]).cuda(x.get_device())
        for i in range(x.size(1)):
          output_t = self.module(x[:, i, :, :])
          output_t  = output_t.unsqueeze(1)
          output = torch.cat((output, output_t ), 1)
          # print('TimeDist_output', output.size())
        return output
        # # Squash samples and timesteps into a single axis
        # x_reshape = x.contiguous().view(x.size(0), -1, x.size(-1))  # (samples * timesteps, input_size)
        #print('TimeDist_x_reshape',x_reshape.shape)
        # y = self.module(x_reshape)
        # print('TimeDist_y', y.shape)
        # # We have to reshape Y
        # if self.batch_first:
        #     y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        # else:
        #    y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        # print('TimeDist_y_reshape',y.size())
        #return y

class FedNewsRec(nn.Module):
    def __init__(self, title_word_embedding_matrix):
        super(FedNewsRec, self).__init__()
        self.doc_encoder = DocEncoder() 
        self.user_encoder = UserEncoder()
        self.title_word_embedding_layer = nn.Embedding.from_pretrained(torch.tensor(title_word_embedding_matrix, dtype=torch.float), freeze=True)
    
        # click_title = Input(shape=(50,30),dtype='int32')
        # can_title = Input(shape=(1+npratio,30),dtype='int32')
    
        self.softmax = nn.Softmax(dim=1)
        self.click_td = TimeDistributed(self.doc_encoder) #, batch_first=True)
        self.can_td = TimeDistributed(self.doc_encoder) #, batch_first=True)
        
    def forward(self, click_title, can_title):
        click_word_vecs = self.title_word_embedding_layer(click_title)
        # print('click', click_word_vecs.shape, click_word_vecs.type)
        can_word_vecs = self.title_word_embedding_layer(can_title)
        # print('can', can_word_vecs.shape, can_word_vecs.type)
        click_vecs = self.click_td(click_word_vecs)
        # print('click_vecs (None, 50, 400)', click_vecs.shape)
        can_vecs = self.can_td(can_word_vecs)
        # print('can_vecs (None, 5, 400)', can_vecs.shape)
    
        user_vec = self.user_encoder(click_vecs)        
        # print('user_vec (None, 400)', user_vec.shape)
        # TODO verify
        scores = torch.einsum('ijk,ik->ij',  can_vecs, user_vec)
        #if verbose:            
        #    print('model scores:', scores.detach().cpu().numpy())
        # print('scores  (None, 5)', scores.shape)
        #logits = self.softmax(scores)     
        # pytorch crossentropyloss function accepts unnormalized scores.
        logits = scores
        # print('logits  (None, 5)', logits.shape)
        
        #news_word_vecs = self.title_word_embedding_layer(news_input)
        #news_vec = self.doc_encoder(news_word_vecs)
        
        # print('user_vec', user_vec.shape)
        # print('news_vec', news_vec.shape)        
        return logits, user_vec #, news_vec

    def news_encoder(self, news_title):
        news_word_vecs = self.title_word_embedding_layer(news_title)
        news_vec = self.doc_encoder(news_word_vecs)
        return news_vec
