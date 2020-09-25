import os
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset v2. Requires a text file listing paths to audio files.
    Others -> 'unknown' class.
    """

    def __init__(self, paths, db, transform=None):
        """paths[Dict]: expects 
           - 'classes': path to txt containing list of class labels
           - 'filelist': path to txt file containing list of WAV
           - 'dataset_dir': path up to dir in which WAV files are stored
        """
        
        with open(paths['classes'], 'r') as clist:
            classes = [s.rstrip() for s in clist.readlines()]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        print(class_to_idx)
        
        data = []
        data_as_map = defaultdict(list)
        with open(paths['filelist'], 'r') as flist:
            for line in flist:
                class_dir, wav_file = line.split('/')
                target = class_to_idx[class_dir] if class_dir in classes else 0
                data.append((line, target))
                data_as_map[target].append(line)

        self.classes = classes
        self.data = data
        self.data_as_map = data_as_map
        self.transform = transform
        self.dataset_root = paths['dataset_dir']
        self.db = db

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Reads audio file here and returns (samples, target)"""
        
        path, target = self.data[index]
        data = self.__loaditem(path, target)

        # pysyft expects 2 values when unpacking a batch
        return data["input"], data["target"]

    def __loaditem(self, path, target):
        samples = self.db[path]

        data = {"input": samples, "target": target}
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
            # DEBUG: only sample from 0,1,2,3,4
            #if item[1] in [5,6,7,8,9]:
            #    weight[idx] = 0.
        return weight
    
    def get_num_of_classes(self):
        return len(self.classes)
    
    def enumerate_by_class(self):
        """Returns generator that outputs all records in class *i* each iteration *i*."""
        
        for class_label_idx in self.data_as_map.keys():
            yield [self.__loaditem(p, class_label_idx) for p in self.data_as_map[class_label_idx]]