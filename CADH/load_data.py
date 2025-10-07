import torch
import scipy.io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np

class uadDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index], index

    def __len__(self):
        count = len(self.images)
        #print(len(self.images),len(self.labels))
        assert len(self.images) == len(self.labels)
        return count

def get_source_data(base_path , domain_name,batch_size=64): # Update with your actual data path
    
    path = base_path + domain_name + '.mat'
    data = sio.loadmat(path)
    data_tensor = torch.from_numpy(data['deepfea'])
    label_tensor = torch.from_numpy(data['label'])
    label_tensor = label_tensor.T  

    source_data = uadDataset(data_tensor, label_tensor)
    #print(type(data_tensor))
    source_loader = DataLoader(source_data, shuffle=True, num_workers=4, batch_size=batch_size)

    classes = torch.unique(label_tensor)
    n_class = classes.size(0)

    dim_fea = data_tensor.size(1)

    return source_loader, n_class, dim_fea  

def get_target_data(base_path, domain_name,batch_size=64):
    path = base_path + domain_name + '.mat'

    data = sio.loadmat(path)
    data_tensor = torch.from_numpy(data['deepfea'])
    label_tensor = torch.from_numpy(data['label'])
    label_tensor = label_tensor.T  

    train_data, test_data, train_label, test_label = train_test_split(data_tensor,
                                                                      label_tensor,
                                                                      test_size=0.1,
                                                                      random_state=42)

    imgs = {'train': train_data, 'query': test_data}
    labels = {'train': train_label, 'query': test_label}

    dataset = {x: uadDataset(images=imgs[x], labels=labels[x])
               for x in ['train', 'query']}

    shuffle = {'train': True, 'query': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['train', 'query']}

    return dataloader

#get_loader_target_unlabeled('D:\\python_project\\untitled10\\uad\\data\\office-31\\', 'amazon_fc7')



