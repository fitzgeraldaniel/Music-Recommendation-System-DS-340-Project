import random
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import pickle
import argparse
from utils.transform import Random_CMR



def load_data(root, load_full=False, sess_key='session_id'):
    # Initialize variables
    sess_map, train_data, valid_data, train_full_data, test_data, n_item = None, None, None, None, None, None

    try:
        # Paths to dataset files
        sess_map_path = os.path.join(root, 'sess_map.pickle')
        train_path = os.path.join(root, 'train.pickle')
        valid_path = os.path.join(root, 'valid.pickle')
        n_item_path = os.path.join(root, 'n_item.pickle')

        # Load session map
        with open(sess_map_path, 'rb') as f:
            sess_map = pickle.load(f)

        # Load train data
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)

        # Load validation data
        with open(valid_path, 'rb') as f:
            valid_data = pickle.load(f)

        # Load number of items
        with open(n_item_path, 'rb') as f:
            n_item = pickle.load(f)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        # Handle missing files here if needed

    # Return the variables without printing their contents
    return train_data, valid_data, n_item, sess_map
    
    


class BaseData(Dataset):
    def __init__(self, data, shuffle_idx, context_idx, hybrid_idx, ranshu=False):
        self.data = data
        self.shuffle_idx = shuffle_idx
        self.context_idx = context_idx
        self.hybrid_idx = hybrid_idx
        self.ranshu = ranshu

    def __getitem__(self, index):
        # Remove any print statements or logging here
        sess_item = self.data[0][index]
        sess_target = self.data[1][index]
        sess_id = self.data[2][index]
        sess_listen = self.data[3][index]
        sess_shuffle = self.shuffle_idx[index]
        sess_context = self.context_idx[index]
        sess_hybrid = self.hybrid_idx[index]

        return sess_item, sess_target, sess_id, sess_listen, sess_shuffle, sess_context, sess_hybrid

    def __len__(self):
        return len(self.data[2])


class CollateFn:
    def __init__(self, args, n_items, transform=None, transition_adj=None, maxlen=19, device=None, train=False):
        self.args = args
        self.n_items = n_items
        self.maxlen = getattr(args, 'maxlen', maxlen)
        self.augmentation = hasattr(args, 'prob') and hasattr(args, 'reorder_r')
        self.transform = transform
        self.transition_adj = transition_adj
        self.device = device
        self.train = train

        # Initialize augmentation only if required
        if self.augmentation:
            self.aug = Random_CMR(
                transition_r=getattr(args, 'prob', 1.0),
                reorder_r=getattr(args, 'reorder_r', 0.5),
                maxlen=self.maxlen,
                transition_dict=self.transition_adj,
                device=self.device
            )

    def __call__(self, batch):
        if self.augmentation:
            return self.padded_batch_w_aug(batch)
        else:
            return self.padded_batch(batch)

    def padded_batch_w_aug(self, batch):
        for i, item in enumerate(batch):
            if not isinstance(item, (tuple, list)) or not isinstance(item[0], (list, torch.Tensor)):
                raise TypeError(f"Unexpected batch structure at index {i}: {item}")

        batch.sort(key=lambda x: len(x[0]), reverse=True)

        batch_dict = {
            'sess': [],
            'lens': [len(x[0]) for x in batch],
            'right_padded_sesss': torch.zeros(len(batch), self.maxlen).long(),
            'orig_sess': torch.zeros(len(batch), self.maxlen).long(),
            'aug1': torch.zeros(len(batch), self.maxlen).long(),
            'aug_len1': [],
            'position_labels': torch.zeros(len(batch), self.maxlen).long() - 1,
            'listen': torch.ones(len(batch), self.maxlen).long() * 2,
            'labels': [],
            'ids': [],
            'shuffle': [],
            'context': [],
            'hybrid': [],
        }

        for i, (sess, label, id, listen, shu, ctxt, hyd, orig_index, orig_sess) in enumerate(batch):
            if self.train:
                if shu == 0:  # nonshuffle
                    aug1, aug1_pos = self.aug(orig_sess, shuffle='nonshuffle')
                elif shu != 0:  # shuffle
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

    def padded_batch(self, batch):
        # Logic for padded batch without augmentation goes here.
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        batch_dict = {
            'sess': [],
            'lens': [len(x[0]) for x in batch],
            'right_padded_sesss': torch.zeros(len(batch), self.maxlen).long(),
            'orig_sess': torch.zeros(len(batch), self.maxlen).long(),
            'listen': torch.ones(len(batch), self.maxlen).long() * 2,
            'labels': [],
            'ids': [],
            'shuffle': [],
            'context': [],
            'hybrid': [],
        }

        for i, (sess, label, id, listen, shu, ctxt, hyd, orig_index, orig_sess) in enumerate(batch):
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
