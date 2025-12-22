import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch

class MOSEI_dataset(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(MOSEI_dataset, self).__init__()
        # 兼容 Inforeg 的路径结构，假设数据在 Inforeg/data/MOSEI 下
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        dataset = pickle.load(open(dataset_path, 'rb'))

        # 提取特征
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
        self.data = data
        self.n_modalities = 3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        Y = self.labels[index]
        
        # 将连续值标签转换为三分类（和 mosei 项目一致）
        threshold = 0.5
        if -3 <= Y <= -threshold:
            target = 0
        elif -threshold < Y < threshold:
            target = 1
        else:
            target = 2
            
        # 返回格式尽量向 Inforeg 靠拢
        # Inforeg AVDataset 返回: spectrogram, image_n, label, av_file
        # 这里返回: text, audio, vision, target
        return self.text[index], self.audio[index], self.vision[index], target

