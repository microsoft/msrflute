# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import logging
import json

from utils import print_rank
from core.dataset import BaseDataset
from experiments.nlg_gru.utils.utility import *

class Dataset(BaseDataset):
    """
    Map a text source to the target text
    """
    
    def __init__(self, data, min_num_words=2, max_num_words=25, test_only=False, user_idx=0, vocab_dict=None, preencoded=False, **kwargs):

        self.utt_list = list()
        self.test_only = test_only
        self.max_num_words = max_num_words
        self.min_num_words = min_num_words
        self.preencoded = preencoded

        # Load the vocab
        self.vocab = load_vocab(kwargs['args']['vocab_dict']) if 'args' in kwargs else load_vocab(vocab_dict)
        self.vocab_size = len(self.vocab)

        # reading the jsonl for a specific user_idx
        self.load_data(data, user_idx)

    def __len__(self):
        """Return the length of the elements in the list."""
        return len(self.utt_list)


    def __getitem__(self, idx):
        """Find the index in the available data"""

        if self.preencoded:
            batch = np.array([self.utt_list[idx]['src_text']], dtype=np.int32)
        else:
            # case_backoff_batch tries to find the best capitalisation that will allow the word to be in vocabulary
            batch = case_backoff_batch([self.utt_list[idx]['src_text']], self.vocab.term_to_idx)
            batch = to_indices(self.vocab, batch)

        return  batch, self.user

    def load_data(self, orig_strct, user_idx):

        if isinstance(orig_strct, str):
            print('Loading json-file: ', orig_strct)
            with open(orig_strct, 'r') as fid:
                orig_strct = json.load(fid)


        self.user_list  = orig_strct['users']
        self.num_samples = orig_strct['num_samples']
        self.user_data  = orig_strct['user_data'] 
        self.user = 'test_only' if self.test_only else self.user_list[user_idx]

        if user_idx != -1:
            self.process_x(self.user_data)

    def process_x(self, user_data):
        print_rank('Processing data-structure: {} Utterances expected'.format(sum(self.num_samples)), loglevel=logging.DEBUG)
        for user in self.user_list:
            for e in user_data[user]['x']:
                utt={}
                utt['src_text'] = e if type(e) is list else e.split()
                utt['duration'] = len(e)
                if utt['duration']<= self.min_num_words:
                    continue

                if utt['duration'] > self.max_num_words:
                    utt['src_text'] = utt['src_text'][:self.max_num_words]
                    utt['duration'] = self.max_num_words
                utt["loss_weight"] = 1.0
                self.utt_list.append(utt)
