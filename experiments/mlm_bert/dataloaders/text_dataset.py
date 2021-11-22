# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.utils.data import Dataset
from utils import print_rank
import logging
import json
import itertools

class TextDataset(Dataset):
    """
    Map a text source to the target text
    """
    def __init__(self, data, args, tokenizer, test_only=False, user_idx=None, max_samples_per_user=-1, min_words_per_utt=5):
        self.utt_list = list()
        self.test_only= test_only
        self.padding = args.get('padding', True)
        self.max_seq_length= args['max_seq_length']
        self.max_samples_per_user = max_samples_per_user
        self.min_num_words = min_words_per_utt
        self.tokenizer = tokenizer
        self.process_line_by_line=args.get('process_line_by_line', False)
        self.user = None


        if self.max_seq_length is None:
            self.max_seq_length = self.tokenizer.model_max_length
            if self.max_seq_length > 512:
                print_rank(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 512 instead. You can change that default value by passing --max_seq_length xxx.", loglevel=logging.DEBUG
                )
                self.max_seq_length = 512
        else:
            if self.max_seq_length > self.tokenizer.model_max_length:
                print_rank(
                    f"The max_seq_length passed ({self.max_seq_length}) is larger than the maximum length for the"
                    f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}.", loglevel=logging.DEBUG
                )
            self.max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

        self.read_data(data, user_idx)

        if not self.process_line_by_line:
            self.post_process_list()


    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        # Find the index in the available data
        if self.process_line_by_line:
            tokenized_text = LineByLineTextDataset(
                                tokenizer=self.tokenizer,
                                input_lines=self.utt_list[idx]['src_text'],
                                line_by_line=True,
                                truncation=True,
                                max_length=self.max_seq_length,
                                padding="max_length")

            self.utt_list[idx]['duration']= len(tokenized_text['input_ids'])
            return tokenized_text
        else:
            return self.utt_list[idx]


    def read_data(self, orig_strct, user_idx):
        """ Reads the data for a specific user (unless it's for val/testing) and returns a 
        list of embeddings and targets."""

        if isinstance(orig_strct, str):
            print('Loading json-file: ', orig_strct)
            with open(orig_strct, 'r') as fid:
                orig_strct = json.load(fid)

        self.user_list  = orig_strct['users']
        self.num_samples= orig_strct['num_samples']
        self.user_data  = orig_strct['user_data']

        if self.test_only:
            self.user = 'test_only'
            self.process_x(self.user_data)
        else:
            self.user = self.user_list[user_idx]
            self.process_x(self.user_data[self.user])


    def process_x(self, raw_x_batch):

        if self.test_only:
            for i, user in enumerate(self.user_list):
                counter=self.process_user(user, raw_x_batch[user])
                self.num_samples[i] = counter # Update userdata counter "num_samples[user]" after truncation
        else:
            counter = self.process_user(self.user, raw_x_batch)
            self.num_samples[self.user_list.index(self.user)] = counter # Update userdata counter "num_samples[user]" after truncation

        if len(self.utt_list) == 0:
            self.utt_list = [{'src_text': 'N/A', 'duration': 0, 'loss_weight': 1.0}]

        print_rank('Processing json-structure for User: {} Utterances Processed: {}'.format(self.user, len(self.utt_list)), loglevel=logging.INFO)


    def process_user(self, user, user_data):
        counter=0
        for line in user_data:
            for e in line:
                if len(e.split()) < self.min_num_words:
                    continue
                if self.max_samples_per_user > -1 and counter >= self.max_samples_per_user:
                    print_rank('Max allowed size per user is reached for user: {},  N: {} utts,  Utt_list Len: {}' \
                               .format(user, counter, len(self.utt_list)), loglevel=logging.DEBUG)
                    return counter
                counter += 1

                utt = {}
                utt['src_text'] = e
                utt['duration'] = len(e.split())
                utt['loss_weight'] = 1.0
                self.utt_list.append(utt)
        return counter


    def post_process_list(self):

        # Use only the text part of the dataset
        input_lines=[line['src_text'] for line in self.utt_list]

        # Process all lines of text
        print_rank('Tokenizing {} Utterances'.format(len(input_lines)), loglevel=logging.DEBUG)
        self.utt_list= LineByLineTextDataset(self.tokenizer, input_lines) #this one has return_special_tokens_mask as True
        
        def group_texts(examples):
            """"Main data processing function that will concatenate all texts
            from our dataset and generate chunks of max_seq_length."""
            
            print_rank('Concatenating Frames in Sequences of {} samples'.format(self.max_seq_length), loglevel=logging.DEBUG)

            if self.padding: # Padding last frame

                total_length = sum([len(k) for k in examples['input_ids']])
                print_rank('Found {} samples Before Concatenation'.format(total_length), loglevel=logging.DEBUG)
                padN= self.max_seq_length - (total_length % self.max_seq_length)
                print_rank('Padding last frame with {} samples'.format(padN), loglevel=logging.DEBUG)
                print_rank('keys {}'.format(examples.keys()), loglevel=logging.DEBUG)
                examples['input_ids'].append([self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*padN) 
                examples['attention_mask'].append([0]*padN)
                
                if 'special_tokens_mask' in examples.keys():
                    examples['special_tokens_mask'].append([1]*padN)

                if 'token_type_ids' in examples.keys():
                    examples['token_type_ids'].append([0]*padN)

       
            # Concatenate all input.
            concatenated_examples = {k: list(itertools.chain.from_iterable(examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            print_rank('Concatenated in {} Samples'.format(total_length), loglevel=logging.DEBUG)
            total_length = (total_length // self.max_seq_length) * self.max_seq_length
            print_rank('Concatenated in {} Frames'.format(total_length // self.max_seq_length), loglevel=logging.DEBUG)

            # Split by chunks of max_len
            self.utt_list=[]
            for i in range(0, total_length, self.max_seq_length):
                utt={}
                for k, t in concatenated_examples.items():
                    utt[k]= t[i : i + self.max_seq_length]
                self.utt_list.append(utt)
                print_rank('Utterance Len is: {}'.format(len(utt['input_ids'])),loglevel=logging.DEBUG)
                
        # Process list of text
        group_texts(self.utt_list) 

        total_length = len(self.utt_list)
        print_rank('Finished Reshaping in Sequences of {} Frames'.format(total_length), loglevel=logging.INFO)

        # Update userdata after truncation
        if not self.test_only:
            self.num_samples[self.user_list.index(self.user)] = total_length

        # Not used anywhere but necessary when the dataset is initiated
        if total_length == 0:
            self.utt_list = [{"input_ids": [0, 2], "special_tokens_mask": [1, 1], "attention_mask": [0, 0]}]

def LineByLineTextDataset(tokenizer, input_lines, truncation=True, max_length=512, padding = False, line_by_line=False):

    if input_lines==['N/A']:
        batch_encoding = {"input_ids": [[0, 2]], "special_tokens_mask": [[1, 1]], "attention_mask": [[0, 0]]}
    else:
        lines = [line for line in input_lines if (len(line) > 0 and not line.isspace())]
        print_rank ('padding is : ' + str(padding),loglevel=logging.DEBUG)
        print_rank ('max_length is : ' + str(max_length),loglevel=logging.DEBUG)
        batch_encoding = tokenizer(lines, truncation=truncation, max_length=max_length, padding = padding, return_special_tokens_mask=True,)
    if line_by_line:
        batch_encoding["input_ids"] = batch_encoding["input_ids"][0]
        batch_encoding["special_tokens_mask"] = batch_encoding["special_tokens_mask"][0]
        batch_encoding["attention_mask"] = batch_encoding["attention_mask"][0]

    return batch_encoding