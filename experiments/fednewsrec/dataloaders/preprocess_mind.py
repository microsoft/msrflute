from nltk.tokenize import word_tokenize
import random
import os
import numpy as np
import torch

MAX_SENTENCE = 30
MAX_ALL = 50
npratio = 4

''' 
    Preprocessing steps for MIND dataset were taken from the FedNewsRec-EMNLP-Findings-2020
    repository, for more information please refer to https://github.com/taoqi98/FedNewsRec/blob/master/code/preprecoess.py.
'''

class MIND:
    def __init__(self, root_data_path, embedding_path) :

        # Preprocessing
        news,news_index,category_dict,subcategory_dict,word_dict = read_news(root_data_path,['train','val'])
        news_title,news_vert,news_subvert=get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict)
        title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)
        train_session, train_uid_click, train_uid_table = read_clickhistory(root_data_path,'train')
        test_session, test_uid_click,test_uid_table = read_clickhistory(root_data_path,'val')
        train_user = parse_user(train_session,news_index)
        test_user = parse_user(test_session,news_index)
        train_sess, train_user_id, train_label, train_user_id_sample = get_train_input(train_session,train_uid_click,news_index)
        test_impressions, test_userids = get_test_input(test_session,news_index)
        get_user_data = GetUserDataFunc(news_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id)
        
        # Return datasets
        print("Preparing train datasets ...")
        train_dict = {'users': [], 'num_samples': [], 'user_data': dict(), 'user_data_label': dict()}

        for uidx in range(50000):
            uid = train_uid_table[uidx]
            click, sample, label = get_user_data(uid)
            user = str(uidx) # uid
            train_dict['users'].append(user)
            train_dict['num_samples'].append(len(click))
            train_dict['user_data'][user] = {'x': click, 'y': sample}
            train_dict['user_data_label'][user] = label

        print("Preparing test datasets ...")
        test_dict = {'users': [], 'num_samples': [], 'user_data': dict(), 'user_data_label': dict()}
        doc_cache = []

        for j in range(len(news_title)):
            doc_cache.append(torch.from_numpy(np.array([news_title[j]])))

        for i in range(len(test_impressions)):
            docids = test_impressions[i]['docs']
            labels = test_impressions[i]['labels']
            nv_hist = [doc_cache[j] for j in test_user['click'][i]]
            nv_imp = [doc_cache[j] for j in docids]
            user = str(i)
            test_dict['users'].append(user)
            test_dict['num_samples'].append(len(nv_imp))
            test_dict['user_data'][user] = {'x':nv_hist, 'y':nv_imp}
            test_dict['user_data_label'][user] = labels
        self.trainset = train_dict
        self.testset = test_dict

def GetUserDataFunc(news_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id):
    def _get_user_data(uid):
        click = []
        sample = []
        label = []
        for sid in train_user_id_sample[uid]:
            click.append(train_user['click'][train_user_id[sid]])
            sample.append(train_sess[sid])
            label.append(train_label[sid])
        click = np.array(click)
        sample = np.array(sample)
        label = np.array(label)
        click = news_title[click]
        sample = news_title[sample]        
        return click,sample,label
    return _get_user_data

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)

def read_news(root_data_path,modes):
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

def get_doc_input(news,news_index,category,subcategory,word_dict):
    news_num=len(news)+1
    news_title=np.zeros((news_num,MAX_SENTENCE),dtype='int32')
    news_vert=np.zeros((news_num,),dtype='int32')
    news_subvert=np.zeros((news_num,),dtype='int32')
    for key in news:    
        vert,subvert,title=news[key]
        doc_index=news_index[key]
        news_vert[doc_index]=category[vert]
        news_subvert[doc_index]=subcategory[subvert]
        for word_id in range(min(MAX_SENTENCE,len(title))):
            news_title[doc_index,word_id]=word_dict[title[word_id].lower()]
        
    return news_title,news_vert,news_subvert


def load_matrix(embedding_path,word_dict):
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


def read_clickhistory(root_data_path,mode):
    
    lines = []
    userids = {}
    uid_table = {}
    with open(os.path.join(root_data_path,mode,'behaviors.tsv')) as f:
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,_,click,imp = lines[i].strip().split('\t')
        true_click = click.split()
        assert not '' in true_click
        if not uid in userids:
            uid_table[len(userids)] = uid
            userids[uid] = []
        userids[uid].append(i)
        imp = imp.split()
        pos = []
        neg = []
        for beh in imp:
            nid, label = beh.split('-')
            if label == '0':
                neg.append(nid)
            else:
                pos.append(nid)
        sessions.append([true_click,pos,neg])
    return sessions,userids,uid_table

def parse_user(session,news_index):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_ALL),dtype='int32'),}
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg =session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) >MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click=[0]*(MAX_ALL-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user

def get_train_input(session,uid_click_talbe,news_index):
    inv_table = {}
    user_id_session = {}

    for uid in uid_click_talbe:
        user_id_session[uid] = []
        for v in uid_click_talbe[uid]:
            inv_table[v] = uid
    
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)                
            user_id_session[inv_table[sess_id]].append(len(sess_pos)-1)
            
    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        #index = np.random.randint(1+npratio)
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    
    return sess_all, user_id, label, user_id_session

def get_test_input(session,news_index):
    
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    
    return Impressions, userid,



