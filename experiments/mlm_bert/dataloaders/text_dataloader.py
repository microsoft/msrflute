# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers.data.data_collator import default_data_collator, DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from experiments.mlm_bert.dataloaders.text_dataset import TextDataset
import torch

class TextDataLoader(DataLoader):
    """
    PyTorch dataloader for loading text data from
    text_dataset.
    """
    def __init__(self, mode, data, num_workers=0,  **kwargs):

        args = kwargs['args']
        task = args['task']
        user_idx = kwargs['user_idx']
        mlm_probability = args['mlm_probability']
        self.batch_size = args['batch_size']
        self.mode = mode
        self.num_workers = num_workers
        self.utt_ids = None
        max_samples_per_user = args.get('max_samples_per_user', -1)
        min_words_per_utt = args.get('min_words_per_utt', 5)
        tokenizer_kwargs = {
                            "cache_dir": args['cache_dir'],
                            "use_fast": args['tokenizer_type_fast'],
                            "use_auth_token":  None
                        }                     
        
        if 'tokenizer_name' in args:
            tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name'], **tokenizer_kwargs)
        elif 'model_name_or_path' in args:
            tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'], **tokenizer_kwargs)
        else:
            raise ValueError("You are instantiating a new tokenizer from scratch. This is not supported by this script.")

        print("Tokenizer is: ",tokenizer)
        
        dataset = TextDataset(
                                data,
                                args= args,
                                test_only = self.mode is not 'train',
                                tokenizer= tokenizer,
                                user_idx=user_idx,
                                max_samples_per_user=max_samples_per_user,
                                min_words_per_utt=min_words_per_utt,
                              )
        self.utt_ids = dataset.user

        try:
            data_collator = DataCollatorForLanguageModeling(
                                                    tokenizer=tokenizer,
                                                    mlm= task=='mlm',
                                                    mlm_probability=mlm_probability,)
        except:

            print('There is an issue with the DataCollator .. Falling back to default_data_collator')
            data_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)

        if self.mode == 'train':
            train_sampler = RandomSampler(dataset)
            super(TextDataLoader, self).__init__(
                                            dataset,
                                            batch_size=self.batch_size,
                                            sampler=train_sampler,
                                            collate_fn=data_collator,
                                            drop_last=False,
                                            num_workers=self.num_workers,
                                            pin_memory=True,
                                            )
                                            
        elif self.mode == 'val' or self.mode == 'test':
            eval_sampler = SequentialSampler(dataset)
            super(TextDataLoader, self).__init__(
                                            dataset,
                                            sampler=eval_sampler,
                                            batch_size= self.batch_size,
                                            collate_fn=data_collator,
                                            drop_last=False,
                                            num_workers=self.num_workers,
                                            pin_memory=True,
                                            )

        else:
            raise Exception("Sorry, there is something wrong with the 'mode'-parameter ")

    def create_loader(self):
        return self

    def get_user(self):
        return self.utt_ids
    