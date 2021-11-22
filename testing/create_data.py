# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import argparse
from collections import OrderedDict
from itertools import islice

val_file = r"./mockup/data/val/val_data.json"
test_file = r"./mockup/data/test/test_data.json"
train_file = r"./mockup/data/train/train_data.json"

def main():

    exp = parse_args()

    # Remove vocab if already exists
    try:
        os.remove("./mockup/models/vocab_reddit.vocab")
    except:
        print("Vocab file not found")

    # Building vocab
    os.system("echo Building vocab")
    os.system("python build_vocab.py --data-dir ./mockup/data/train --target-dir mockup/models")

    # Preprocessing data
    os.system("echo Preprocessing data")

    min = -25
    max = 0
    for iteration in range(3):

        min = min + 25
        max = max + 25

        if iteration == 0:
            file = val_file
        elif iteration == 1:
            file = test_file
        elif iteration == 2:
            file = train_file

        with open(file, 'r') as f:
            json_file = json.load(f)
            
        users_list = list()
        num_samples = json_file['num_samples']
        user_data = json_file['user_data']

        # Truncate user_data to only 25 elements per file
        user_data = OrderedDict(islice(user_data.items(), min, max))
        user_data = dict(user_data)

        # Give format to user_data and create users_list
        if exp == "nlg":
            for users in user_data:
                listToStr = ''
                users_list.append(users)
                for i, sentences in enumerate(user_data[users]['x']):
                    for j, pieces in enumerate(sentences):
                        listToStr = ' '.join([elem for elem in pieces])
                        user_data[users]['x'][i][j] = listToStr

                    full_sentence = ' '.join([elem for elem in sentences])
                    full_sentence = full_sentence.replace('<PAD>', '').replace('<EOS>', '').replace('<BOS>', '').strip()
                    user_data[users]['x'][i] = full_sentence
                    user_data[users].pop('y',None)

        elif exp == "mlm":

            user_data_aux = dict()
            for users in user_data:
                listToStr = ''
                users_list.append(users)
                for i, sentences in enumerate(user_data[users]['x']):
                    for j, pieces in enumerate(sentences):
                        listToStr = ' '.join([elem for elem in pieces])
                        listToStr = listToStr.replace('<PAD>', '').replace('<EOS>', '').replace('<BOS>', '').strip()
                        user_data[users]['x'][i][j] = listToStr
                    user_data[users].pop('y',None)
                user_data_aux[users] = user_data[users]['x']
            user_data = user_data_aux
        
        # Adjust number of samples
        new_dict = {'users':users_list ,'num_samples':num_samples[min:max], 'user_data':user_data}
        f = open(file,'w')
        json.dump(new_dict,f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--Exp", help="Experiment name (nlg/mlm)")
    args = parser.parse_args()
    exp = args.Exp

    if exp != "mlm" and exp!="nlg":
        raise ValueError ("Invalid experiment name, please try once again with mlm/nlg")
    else:
        return exp

if __name__ == '__main__':
    main()
