# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import time
from argparse import ArgumentParser

import numpy as np
from collections import namedtuple
from tqdm import tqdm

TR_UPPER = {ord('i'): 'İ'}
TR_LOWER = {ord('I'): 'ı'}

Vocab = namedtuple('Vocab', ['idx_to_term', 'term_to_idx'])


def load_vocab(url):
    """Load a vocabulary file.

    url -- string -- url to the txt file

    returns -- Vocab(idx_to_term=list, term_to_idx=dict)
    """
    term_to_idx = {}
    idx_to_term = []
    with open(url, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word = line.strip()
            idx_to_term.append(word)
            term_to_idx[word] = i
    return Vocab(idx_to_term, term_to_idx)


def to_indices(vocab, batch, ndim=2, oov_idx=0, pad_idx=-1):
        """Convert a nested list of strings to a np.array of integers.
        
        vocab -- Vocab -- the vocabulary of the model
        
        batch -- [..[str]..] -- multidimensional batch

        ndim -- int -- number of dimensions in batch

        oov_idx -- int or None -- if specified, replace missing terms by
                   the given index, otherwise raise an error

        pad_idx -- int or None -- if specified, pad short last-dimension
                   as specified, otherwise raise an error

        raises -- ValueError -- if pad is required but pad_idx not specified
               -- KeyError -- if oov is required but oov_idx not specified

        returns -- np.array(int) -- term indices
        """
        #print_rank(f'to_indices: batch len: {len(batch)} ndim: {ndim}')
        if ndim == 1:
            return np.array(
                [(vocab.term_to_idx[term] if oov_idx is None else 
                        vocab.term_to_idx.get(term, oov_idx)) 
                            for term in batch],  dtype=np.int32)

        if ndim == 2:
            # note: in most circumstances there is only one example in the batch
            # as a result, padding is never applied. We rely on collate_fn to properly
            # apply padding.
            length = max(len(row) for row in batch)
            if pad_idx is None and min(len(row) for row in batch) != length:
                raise ValueError('Padding required, but no pad_idx provided')
            pad = length * [pad_idx]

            result = np.array(
                [[(vocab.term_to_idx[term] if oov_idx is None else
                        vocab.term_to_idx.get(term, oov_idx))
                            for term in row] + pad[len(row):]
                                for row in batch], dtype=np.int32)
            #print_rank(f'to_indices result: {result.shape}')
            return result

        # Flatten to a 2D batch, then recurse & reshape up (this ensures
        # padding is handled correctly)
        shape = [len(batch)]
        for _ in range(2, ndim):
            shape.append(len(batch[0]))
            batch = [item for sub_batch in batch for item in sub_batch]
        shape.append(-1)
        return to_indices(vocab, batch, ndim=2, oov_idx=oov_idx, pad_idx=pad_idx).reshape(*shape)

def case_backoff_batch(batch, vocab):
    """Perform capitalization backoff on words both to lower & initial-upper case variants.

    batch -- list(list(string)) -- batch of sentences of words, to back off

    vocab -- set(string) -- vocabulary to consider

    returns -- list(list(string)) -- backed-off batch
    """

    def _variants(word):
        yield word
        yield word.translate(TR_LOWER).lower()
        yield word.lower()
        if len(word) > 1:
            yield word[0].translate(TR_UPPER).capitalize() + word[1:]
        yield word.capitalize()

    return [[next((variant for variant in _variants(word) if variant in vocab),
                  word)  # will become OOV
             for word in sentence]
            for sentence in batch]


def encode_data(data_dict, vocab):
    '''Encode data that is in the format expected by FLUTE
    
    Parameters
    ----------
    data_dict: dict
        Dictionary where keys consist of usernames and values give
        the data for that user, specified by another dictionary with
        keys :code:`x` (features) and, optionally, :code:`y` (labels).
    vocab:

    Returns
    -------
    dict
        Dictionary in the same format as the input one, but now the
        data in the :code:`x` field is given by tokens (i.e., integers),
        instead of strings.
    '''
    new_dict = {}
    for key, value in tqdm(data_dict.items()):
        user_data = [s.split() for s in value['x']]
        processed_data = case_backoff_batch(user_data, vocab.term_to_idx)
        encoded_data = [[vocab.term_to_idx.get(term, 0) for term in row] for row in processed_data]
        new_dict[key] = {'x': encoded_data}

    return new_dict


if __name__ == '__main__':
    parser = ArgumentParser(description='Encodes data')
    parser.add_argument('data_path', type=str, help='Path to data')
    parser.add_argument('vocab_path', type=str, help='Path to vocabulary')
    args = parser.parse_args()

    if not os.path.isfile(args.data_path):
        raise ValueError('data file does not exist')
    if not os.path.isfile(args.vocab_path):
        raise ValueError('vocabulary file does not exist')
    if args.data_path[-5:] != '.json':
        raise ValueError('argument must be a valid json file')

    # Load vocabulary
    print('Loading vocabulary...')
    vocab = load_vocab(args.vocab_path)

    # Load and encode data
    print('Loading data... ', end='', flush=True)
    start_time = time.time()
    with open(args.data_path, 'r') as input_file:
        all_data = json.load(input_file)
    print(f'Finished in {time.time() - start_time:.2f}s')

    print('Converting data...')
    converted_user_data = encode_data(all_data['user_data'], vocab)
    
    # For debug purposes
    for k, v in converted_user_data.items():
        print(f'USER: {k}\nDATA: {v}')
        break

    # Save encoded data to disk
    print('Saving encoded data to disk...')
    all_data['user_data'] = converted_user_data
    with open(f'{args.data_path[:-5]}-encoded.json', 'w') as output_file:
        json.dump(all_data, output_file)