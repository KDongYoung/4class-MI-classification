from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from collections import Counter

class DataGenerator:
    def __init__(self, args, id):
        print("[Load Data]")
        if args.mode == 'train':
            self.train_loader = self.__data_loader(args, id, 'train')
            self.valid_loader = self.__data_loader(args, id, 'val')
            self.test_loader = self.__data_loader(args, id, 'test')
            print(f"train size: {self.train_loader.dataset.X.shape}")
            print(f"val size: {self.valid_loader.dataset.X.shape}")
            mini_batch_shape = list(self.train_loader.dataset.X.shape)
            mini_batch_shape[0] = args.batch_size
            print("")
        else:
            self.test_loader = self.__data_loader(args, id, 'test')
            print("")

    # get weights (use the number of samples)
    def make_weights_for_balanced_classes(self,dataset):  # y값 class에 따라
        counts = Counter()
        classes = []
        for y in dataset:
            y = int(y[1]) # class에 접근
            counts[y] += 1 # count each class samples
            classes.append(y) 
        n_classes = len(counts)

        weight_per_class = {}
        for y in counts: # the key of counts
            weight_per_class[y] = 1 / (counts[y] * n_classes)

        weights = torch.zeros(len(dataset))
        for i, y in enumerate(classes):
            weights[i] = weight_per_class[int(y)]

        return weights

    def __data_loader(self, args, id, phase):
        dataset = EEGDataset(args.data_root, id, phase, args.generate)
        
        if phase=="train":
            weights=self.make_weights_for_balanced_classes(dataset)

            # sampler=torch.utils.data.WeightedRandomSampler(weights, replacement=True, num_samples=args.batch_size*2)
            sampler=torch.utils.data.WeightedRandomSampler(weights, replacement=True, num_samples=args.batch_size*5)
            return DataLoader(dataset, batch_size=args.batch_size, sampler= sampler)
                            #   drop_last=True if phase == 'train' else False)
        else:
            return DataLoader(dataset, batch_size=args.batch_size, shuffle= False)
            
"""
Make a EEG dataset
X: EEG data
Y: 4 class
"""
class EEGDataset(Dataset):
    def __init__(self, data_root, subj_id, phase, generate):
        self.data_root = data_root
        self.subj_id = subj_id
        self.phase = phase
        self.generate=generate
        self.load_data()

    def load_data(self):
        if self.phase != 'test':
            if self.generate and self.phase=='train':
                self.X1 = np.load(f"{self.data_root}/{self.phase}/S{self.subj_id:02}_X.npy")
                self.y1 = np.load(f"{self.data_root}/{self.phase}/S{self.subj_id:02}_y.npy")
                self.X2 = np.load(f"{self.data_root}/{self.phase}/Generate_1010_3/S{self.subj_id:02}_X.npy")
                self.y2 = np.load(f"{self.data_root}/{self.phase}/Generate_1010_3/S{self.subj_id:02}_Y.npy")

                self.X=np.concatenate((self.X1, self.X2))
                self.y=np.concatenate((self.y1, self.y2))
            else:
                self.X = np.load(f"{self.data_root}/{self.phase}/S{self.subj_id:02}_X.npy")
                self.y = np.load(f"{self.data_root}/{self.phase}/S{self.subj_id:02}_y.npy")

        else:
            self.X = np.load(f"{self.data_root}/{self.phase}/S{self.subj_id:02}_X.npy")
    
    def __len__(self):
        if self.phase != 'test':
            return len(self.y)
        else:
            return len(self.X)

    def __getitem__(self, idx):
        if self.phase != 'test':
            X = self.X[idx].astype('float32')  # for only eeg
            y = self.y[idx].astype('int64') # X 1개 (segment 1개)에 따라 y 1개
            X=np.expand_dims(X,axis=0) # (1, channel, time) batch 형태로
            return X, y, self.subj_id
        else:
            X = self.X[idx].astype('float32')  # for only eeg
            X=np.expand_dims(X,axis=0) # (1, channel, time) batch 형태로
            return X, self.subj_id