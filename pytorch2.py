import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np

train_images = np.random.randint(256, size=(20, 32, 32, 3))
train_labels = np.random.randint(2, size=(20, 1))

print(train_images.shape, train_labels.shape)
#batch, 너비,높이, 채널수
class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.x_data = self.x_data.permute(0, 3, 1, 2) #이미지개수,채널수, 너비,높이로 순서변경
        self.y_data = torch.LongTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_data = TensorData(train_images, train_labels)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
print(train_data[0][0].size())

class Mydataset(Dataset):

    def __init__(self, x_data, y_data, transform = None):

        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class ToTesor:
    def __call__(self, sample):
        inputs