import os
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from nltk.tokenize import word_tokenize

from core.model import BaseModel
from experiments.fednewsrec.utils import ndcg_score, mrr_score
from experiments.fednewsrec.fednewsrec_model import FedNewsRec

''' 
    The FedNewsRec model is taken from FedNewsRec-EMNLP-Findings-2020 repository and ported to PyTorch
    framework to be compatible with FLUTE (https://github.com/simra/FedNewsRec#fednewsrec-emnlp-findings-2020). 
    For more information regarding this model, please refer to https://github.com/taoqi98/FedNewsRec.
'''

class FEDNEWS(BaseModel):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()

        root_data_path = model_config['embbeding_path']
        embedding_path = model_config['embbeding_path']

        news,news_index,category_dict,subcategory_dict,word_dict = self.read_news(root_data_path,['train','val'])
        title_word_embedding_matrix, _ = self.load_matrix(embedding_path,word_dict)
        self.net = FedNewsRec(title_word_embedding_matrix)

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''

        if not self.net.training:
            return torch.tensor(0) # Not using the loss during evaluation
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        (click, sample), label = input['x'], input['y']
        click = click.to(device)
        sample = sample.to(device)
        label = label.to(device)
        criterion = CrossEntropyLoss()
        output, _ = self.net.forward(click, sample)
        return criterion(output, label)

    def inference(self, input):
        '''Performs forward step and computes metrics'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        (nv_hist, nv_imp), labels = input['x'], input['y']
        nv_hist = nv_hist.to(device)
        nv_imp = nv_imp.to(device)

        nv = self.net.news_encoder(nv_imp).detach().cpu().numpy()  # news vector?
        nv_hist = self.net.news_encoder(nv_hist)
        uv = self.net.user_encoder(nv_hist.unsqueeze(0)).detach().cpu().numpy()[0] # user vector?

        score = np.dot(nv,uv)
        auc = roc_auc_score(labels,score)
        mrr = mrr_score(labels,score)
        acc = ndcg_score(labels,score,k=1)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)

        return {'output':None, 'acc': acc, 'batch_size': 1, \
                'auc': {'value':auc,'higher_is_better': True},
                'mrr': {'value':mrr,'higher_is_better': True},
                'ndcg5': {'value':ndcg5,'higher_is_better': True},
                'ndcg10': {'value':ndcg10,'higher_is_better': True}} 

    def read_news(self, root_data_path, modes):
        news={}
        category=[]
        subcategory=[]
        news_index={}
        index=1
        word_dict={}
        word_index=1
        
        for mode in modes:
            with open(os.path.join(root_data_path,mode,'news.tsv'), encoding="utf8") as f:
                lines = f.readlines()
            for line in lines:
                splited = line.strip('\n').split('\t')
                doc_id,vert,subvert,title= splited[0:4]
                if doc_id in news_index:
                    continue
                news_index[doc_id]=index
                index+=1
                category.append(vert)
                subcategory.append(subvert)
                title = title.lower()
                title=word_tokenize(title)
                news[doc_id]=[vert,subvert,title]
                for word in title:
                    word = word.lower()
                    if not(word in word_dict):
                        word_dict[word]=word_index
                        word_index+=1
        category=list(set(category))
        subcategory=list(set(subcategory))
        category_dict={}
        index=1
        for c in category:
            category_dict[c]=index
            index+=1
        subcategory_dict={}
        index=1
        for c in subcategory:
            subcategory_dict[c]=index
            index+=1
        return news,news_index,category_dict,subcategory_dict,word_dict
    
    def load_matrix(self, embedding_path,word_dict):
        embedding_matrix = np.zeros((len(word_dict)+1,300))
        have_word=[]
        with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
            while True:
                l=f.readline()
                if len(l)==0:
                    break
                l=l.split()
                word = l[0].decode()
                if word in word_dict:
                    index = word_dict[word]
                    tp = [float(x) for x in l[1:]]
                    embedding_matrix[index]=np.array(tp)
                    have_word.append(word)
        return embedding_matrix,have_word
            