import random
import torch
from torch.utils.data import Dataset
import numpy as np


from utils.transform import Random_CMR
import pickle

import pandas as pd
import ast  # Importing ast for safe evaluation of strings

import os

def load_data(root, sess_key='session_id'):
    print('Loading Data ...')

    # Initialize variables to ensure they exist for the return statement
    sess_map = None
    train_data = None
    valid_data = None
    train_full_data = None
    test_data = None
    n_item = None

    try:
        # Check if necessary files exist before attempting to load them
        sess_map_path = os.path.join(root, 'sess_map.pickle')
        train_path = os.path.join(root, 'train.pickle')
        valid_path = os.path.join(root, 'valid.pickle')
        train_full_path = os.path.join(root, 'train_full.pickle')
        test_path = os.path.join(root, 'test.pickle')
        n_item_path = os.path.join(root, 'n_item.pickle')

        # Load session map
        print('Loading session map...')
        with open(sess_map_path, 'rb') as f:
            sess_map = pickle.load(f)
        print('Successfully loaded session map!')

        # Load train data
        print('Loading train data...')
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        print('Successfully loaded train data!')

        # Load validation data
        print('Loading validation data...')
        with open(valid_path, 'rb') as f:
            valid_data = pickle.load(f)
        print('Successfully loaded validation data!')

        # Load full training data
        print('Loading full training data...')
        with open(train_full_path, 'rb') as f:
            train_full_data = pickle.load(f)
        print('Successfully loaded full training data!')

        # Load test data
        print('Loading test data...')
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        print('Successfully loaded test data!')

        # Load number of items
        print('Loading number of items...')
        with open(n_item_path, 'rb') as f:
            n_item = pickle.load(f)
        print('Successfully loaded number of items!')

    except FileNotFoundError as e:
        print(f'Dataset does not exist! Reformatting the Data ...')
        print(e)

        # Load the data from CSV if files are missing
        data_path = os.path.join(root, 'seq_new.csv')
        print(f'Reading data from {data_path}...')
        
        all_seqs = pd.read_csv(data_path)
        all_seqs['ItemId'] = all_seqs['ItemId'].apply(eval)
        all_seqs['not_skipped'] = all_seqs['not_skipped'].apply(eval)

        sess_map = all_seqs.drop(columns='ItemId').reset_index(drop=True)

        train_full_data = reformat_data(all_seqs, 'train_valid')
        print('Train Full data ... Done!')
        train_data = reformat_data(all_seqs, 'train')
        print('Train data ... Done!')
        valid_data = reformat_data(all_seqs, 'valid')
        print('Valid data ... Done!')
        test_data = reformat_data(all_seqs, 'test')
        print('Test data ... Done!')

        itemmap_path = os.path.join(root, 'item_map.csv')
        item_map = pd.read_csv(itemmap_path)
        n_item = item_map.shape[0] + 1

        # Save the loaded data for future use
        print('Saving loaded data as pickle files...')
        with open(sess_map_path, 'wb') as f:
            pickle.dump(sess_map, f, pickle.HIGHEST_PROTOCOL)
        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
        with open(valid_path, 'wb') as f:
            pickle.dump(valid_data, f, pickle.HIGHEST_PROTOCOL)
        with open(train_full_path, 'wb') as f:
            pickle.dump(train_full_data, f, pickle.HIGHEST_PROTOCOL)
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
        with open(n_item_path, 'wb') as f:
            pickle.dump(n_item, f, pickle.HIGHEST_PROTOCOL)
        print('Data saving completed!')

    except Exception as e:
        print(f'An error occurred: {e}')

    # Return the loaded or reformatted data
    return train_data, valid_data, train_full_data, test_data, n_item, sess_map


def df2sess(all_seqs, data_type, sess_key='session_id', item_key=['ItemId', 'not_skipped']):
    user_seqs = all_seqs[all_seqs['train'] == data_type].reset_index(drop=True)

    # Context Info
    shuffle_info = user_seqs[['session_id', 'shuffle_session']]

    user_seqs.index = user_seqs[sess_key]
    user_seqs = user_seqs[item_key].to_dict()
    return user_seqs, shuffle_info

def process_sess(user_seqs, shuffle_info):
    shuffle_info.set_index('session_id', inplace=True)
    shuffle_info = shuffle_info['shuffle_session'].to_dict()

    ids = []
    pos = []
    seqs = []
    listen = []

    user_ids = list(user_seqs['ItemId'].keys())
    for i in range(len(user_ids)):
        user_id = user_ids[i]
        seq = user_seqs['ItemId'][user_id]
        not_skipped = user_seqs['not_skipped'][user_id]
        
        if shuffle_info[user_id] == 'nonshuffle':
            for j in range(1, len(seq)):
                if not_skipped[-j] == False: # remain only not_skipped == True
                    continue
                target = seq[-j]
                pos += [target]
                seqs += [seq[:-j]]
                listen += [not_skipped[:-j]]
                ids += [user_id]

        else:
            for j in range(1, len(seq)):
                if not_skipped[-j] == False: # remain only not_skipped == True
                    continue
                if np.sum(not_skipped[:-j]) == 0:
                    continue
                else:
                    target = seq[-j]
                    pos += [target]
                    seqs += [list(np.array(seq[:-j])[not_skipped[:-j]])]
                    listen += [[True] * np.sum(not_skipped[:-j])]
                    ids += [user_id]
    
    return (seqs, pos, ids, listen)

def reformat_data(all_seqs, data_type, sess_key='session_id', item_key=['ItemId', 'not_skipped']):
    user_seqs, shuffle_info = df2sess(all_seqs, data_type, sess_key, item_key)
    (seqs, pos, ids, listen) = process_sess(user_seqs, shuffle_info)

    return (seqs, pos, ids, listen)

class BaseData(Dataset):
    def __init__(self, data, shuffle_idx, context_idx, hybrid_idx, ranshu=False):
        self.data = data
        self.shuffle_idx = shuffle_idx
        self.context_idx = context_idx
        self.hybrid_idx = hybrid_idx
        self.ranshu = ranshu

    def __getitem__(self, index):
        if self.ranshu:
            orig_sess = self.data[0][index][:]
            x = list(enumerate(orig_sess))
            random.shuffle(x)
            orig_index, sess_item = zip(*x)
            orig_index = list(orig_index)
            sess_item = list(sess_item)
        else:
            sess_item = self.data[0][index]
            orig_index = list(range(len(self.data[0][index])))
            orig_sess = self.data[0][index][:]
        sess_target = self.data[1][index]
        sess_id = self.data[2][index]
        sess_listen = self.data[3][index]
        sess_shuffle = self.shuffle_idx[index]
        sess_context = self.context_idx[index]
        sess_hybrid = self.hybrid_idx[index]

        return sess_item, sess_target, sess_id, sess_listen, sess_shuffle, sess_context, sess_hybrid, orig_index, orig_sess

    def __len__(self):
        return len(self.data[2])


class collate_fn(object):
    def __init__(self, args, n_items, transform, transition_adj=None, maxlen=19, device=None, train=False):
        self.args = args
        self.n_items = n_items
        self.maxlen = args.maxlen
        self.augmentation = True
        self.transform = transform
        self.transition_adj = transition_adj
        self.maxlen = maxlen
        self.device = device
        self.train = train

        if self.augmentation:
            self.aug = Random_CMR(transition_r=self.args.prob, reorder_r=self.args.reorder_r, maxlen=self.maxlen,
                                  transition_dict=self.transition_adj, device=self.device)

    def __call__(self, batch):
        if self.augmentation:
            return self.padded_batch_w_aug(batch)
        else:
            return self.padded_batch(batch)

    def padded_batch(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        batch_dict = {}

        batch_dict['sess'] = []
        batch_dict['lens'] = [len(x[0]) for x in batch]
        batch_dict['right_padded_sesss'] = torch.zeros(len(batch), self.maxlen).long()
        batch_dict['orig_sess'] = torch.zeros(len(batch), self.maxlen).long()
        batch_dict['position_labels'] = torch.zeros(len(batch), self.maxlen).long() - 1
        batch_dict['listen'] = torch.ones(len(batch), self.maxlen).long() * 2
        batch_dict['labels'] = []
        batch_dict['ids'] = []
        batch_dict['shuffle'] = []
        batch_dict['context'] = []
        batch_dict['hybrid'] = []

        for i, (sess, label, id, listen, shu, ctxt, hyd, orig_index, orig_sess) in enumerate(batch):
            batch_dict['sess'].append(sess)
            batch_dict['labels'].append(label)
            batch_dict['right_padded_sesss'][i, :batch_dict['lens'][i]] = torch.LongTensor(sess)
            batch_dict['orig_sess'][i, :batch_dict['lens'][i]] = torch.LongTensor(orig_sess)
            batch_dict['position_labels'][i, :batch_dict['lens'][i]] = torch.LongTensor(orig_index)
            batch_dict['listen'][i, :batch_dict['lens'][i]] = torch.LongTensor(listen)
            batch_dict['ids'].append(id)
            batch_dict['shuffle'].append(shu)
            batch_dict['context'].append(ctxt)
            batch_dict['hybrid'].append(hyd)
        batch_dict['labels'] = torch.tensor(batch_dict['labels']).long()
        batch_dict['shuffle'] = torch.tensor(batch_dict['shuffle']).long()
        batch_dict['context'] = torch.tensor(batch_dict['context']).long()
        batch_dict['hybrid'] = torch.tensor(batch_dict['hybrid']).long()

        if self.transform:
            batch_dict = self.transform(batch_dict)

        return batch_dict
    
    def padded_batch_w_aug(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        batch_dict = {}

        batch_dict['sess'] = []
        batch_dict['lens'] = [len(x[0]) for x in batch]
        batch_dict['right_padded_sesss'] = torch.zeros(len(batch), self.maxlen).long()
        batch_dict['orig_sess'] = torch.zeros(len(batch), self.maxlen).long()
        batch_dict['aug1'] = torch.zeros(len(batch), self.maxlen).long()
        batch_dict['aug_len1'] = []
        batch_dict['position_labels'] = torch.zeros(len(batch), self.maxlen).long() - 1
        batch_dict['aug2'] = torch.zeros(len(batch), self.maxlen).long()
        batch_dict['aug_len2'] = []
        batch_dict['position_labels2'] = torch.zeros(len(batch), self.maxlen).long() - 1
        batch_dict['listen'] = torch.ones(len(batch), self.maxlen).long() * 2
        batch_dict['labels'] = []
        batch_dict['ids'] = []
        batch_dict['shuffle'] = []
        batch_dict['context'] = []
        batch_dict['hybrid'] = []

        for i, (sess, label, id, listen, shu, ctxt, hyd, orig_index, orig_sess) in enumerate(batch):
            if self.train:
                if shu == 0: # nonshuffle
                    aug1, aug1_pos = self.aug(orig_sess, shuffle='nonshuffle')

                elif shu != 0: # shuffle
                    aug1, aug1_pos = self.aug(orig_sess, shuffle='shuffle')                
                    
                aug_len1 = len(aug1)
                batch_dict['aug_len1'].append(aug_len1)
                batch_dict['aug1'][i, :aug_len1] = torch.LongTensor(aug1)
                batch_dict['position_labels'][i, :aug_len1] = torch.LongTensor(aug1_pos)
            
            batch_dict['sess'].append(sess)
            batch_dict['labels'].append(label)
            batch_dict['right_padded_sesss'][i, :batch_dict['lens'][i]] = torch.LongTensor(sess)
            batch_dict['orig_sess'][i, :batch_dict['lens'][i]] = torch.LongTensor(orig_sess)        
            batch_dict['listen'][i, :batch_dict['lens'][i]] = torch.LongTensor(listen)
            batch_dict['ids'].append(id)
            batch_dict['shuffle'].append(shu)
            batch_dict['context'].append(ctxt)
            batch_dict['hybrid'].append(hyd)

        batch_dict['labels'] = torch.tensor(batch_dict['labels']).long()
        batch_dict['shuffle'] = torch.tensor(batch_dict['shuffle']).long()
        batch_dict['context'] = torch.tensor(batch_dict['context']).long()
        batch_dict['hybrid'] = torch.tensor(batch_dict['hybrid']).long()

        return batch_dict
