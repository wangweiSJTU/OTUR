import torch.utils.data as data
import torch,pdb
import h5py
import random

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.label = hf.get("label")

    def __getitem__(self, index):
        r=random.randint(0,self.label.shape[0]-1)
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.label[r,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]
