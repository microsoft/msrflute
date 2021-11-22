# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from experiments.nlg_gru.dataloaders.text_dataset import TextDataset
from utils.data_utils import BatchSampler, DynamicBatchSampler

class TextDataLoader(DataLoader):
    """
    PyTorch dataloader for loading text data from
    text_dataset.
    """
    def __init__(self, mode, num_workers=0, **kwargs):

        args = kwargs['args']
        self.batch_size = args['batch_size']
        batch_sampler = None

        dataset = TextDataset(
                        data   = kwargs['data'],
                        test_only    = not mode=="train",
                        vocab_dict   = args['vocab_dict'],
                        user_idx     = kwargs['user_idx'], 
                        max_num_words= args['max_num_words'],
                        preencoded   = args.get('preencoded', False))
        
        if mode == 'train':
            
            sampler = DistributedSampler(dataset,num_replicas=1,rank=0)
            sampler.set_epoch(random.randint(0, 10**10))
            batch_sampler = DynamicBatchSampler(sampler,
                                            frames_threshold = args['max_num_words'],
                                            max_batch_size   = self.batch_size,
                                            unsorted_batch   = args['unsorted_batch'],
                                            fps=1)

        elif mode == 'val' or mode == 'test':
            sampler = BatchSampler(dataset, batch_size=self.batch_size, randomize=False, drop_last=False)
            super().__init__(dataset,
                             batch_sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=self.collate_fn,
                             pin_memory=args["pin_memory"])
            return

        if batch_sampler is None:
            super().__init__(dataset,
                             batch_size=self.batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=self.collate_fn,
                             drop_last=True)
        else:
            super().__init__(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             collate_fn=self.collate_fn,
                             pin_memory=args["pin_memory"])


    def create_loader(self):
        return self
        

    def collate_fn(self, batch):
        def pad_and_concat_feats(labels):
            batch_size = len(labels)
            max_len = max(len(l[0]) for l in labels)
            cat_labels = np.full((batch_size, max_len), -1)

            for e, l in enumerate(labels):
                cat_labels[e,:len(l[0])] = np.squeeze(l)
            return cat_labels


        src_seq, utt_ids = zip(*batch)
        x_len =  [len(s[0]) for s in src_seq]

        src_seq = pad_and_concat_feats(src_seq)
        packed  = {
                    'x': torch.from_numpy(src_seq).long(),
                    'x_len': x_len,
                    'utt_ids' : utt_ids,
                    'total_frames' : sum(x_len),
                    'total_frames_with_padding' : np.prod(src_seq.shape),
                    'loss_weight' : None
                }
        return packed
    