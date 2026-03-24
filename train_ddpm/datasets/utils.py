import os
import os.path
import hashlib
import errno
import torch
from torch.utils.data import Dataset
from torch.utils.model_zoo import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
import glob as glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class KMFlowTensorDataset(Dataset):
    def __init__(self, data_path,
                 train_ratio=1, test=False,
                 stat_path=None,
                 max_cache_len=4000,
                 ):
        np.random.seed(1)
        self.all_data = np.load(data_path) 
        print('Data set shape: ', self.all_data.shape)
        idxs = np.arange(self.all_data.shape[0])
        num_of_training_seeds = int(train_ratio*len(idxs)) 
        self.train_idx_lst = idxs[:num_of_training_seeds]
        print(f'Train {len(self.train_idx_lst)} sequences')
        self.test_idx_lst = idxs[num_of_training_seeds:]
        print(f'Test {len(self.test_idx_lst)} sequences.')
        self.time_step_lst = np.arange(self.all_data.shape[1]-2)

        if not test:
            self.idx_lst = self.train_idx_lst[:]
        else:
            self.idx_lst = self.test_idx_lst[:]


        self.cache = {}
        self.max_cache_len = max_cache_len

        if stat_path is not None:
            self.stat_path = stat_path
            loaded_stat = np.load(stat_path)
            if isinstance(loaded_stat, np.lib.npyio.NpzFile):
                # Convert to a plain dict so DataLoader workers can pickle this dataset on Windows.
                self.stat = {k: loaded_stat[k] for k in loaded_stat.files}
                loaded_stat.close()
            else:
                self.stat = loaded_stat.item() if hasattr(loaded_stat, "item") else loaded_stat
        else:
            self.stat = {}
            self.prepare_data()

    def __len__(self):
        return len(self.idx_lst) * len(self.time_step_lst)

    def prepare_data(self):
        # load all training data and calculate their statistics
        self.stat['mean'] = np.mean(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        self.stat['scale'] = np.std(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        data_mean = self.stat['mean']
        data_scale = self.stat['scale']
        print(f'Data statistics, mean: {data_mean}, scale: {data_scale}')


    def preprocess_data(self, data):
        # normalize data

        s = data.shape[-1]

        data = (data - self.stat['mean']) / (self.stat['scale'])
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        # save data statistics to out_dir
        np.savez(out_dir, mean=self.stat['mean'], scale=self.stat['scale'])

    def __getitem__(self, idx):
        seed = self.idx_lst[idx // len(self.time_step_lst)]
        frame_idx = idx % len(self.time_step_lst)
        id = idx

        if id in self.cache.keys():
            return self.cache[id]
        else:
            frame0 = self.preprocess_data(self.all_data[seed, frame_idx  ])
            frame1 = self.preprocess_data(self.all_data[seed, frame_idx+1])
            frame2 = self.preprocess_data(self.all_data[seed, frame_idx+2])

            frame = np.concatenate((frame0[None, ...], frame1[None, ...], frame2[None, ...]), axis=0)

            self.cache[id] = frame

            if len(self.cache) > self.max_cache_len:
                random_key = np.random.choice(list(self.cache.keys()))
                self.cache.pop(random_key)
            return frame






