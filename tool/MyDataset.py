import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, data_guids,data_texts,data_imgs, data_labels=None,config=None):
        self.config = config
        self.data_guids = data_guids
        self.data_texts = data_texts
        self.data_imgs = data_imgs
        self.data_labels = data_labels
        
    def __getitem__(self, i):
        if self.config.mode=="train":
            return (self.data_guids[i],self.data_texts[i],self.data_imgs[i],self.data_labels[i])
        else:
            return (self.data_guids[i],self.data_texts[i],self.data_imgs[i])
        
    def __len__(self):
        return len(self.data_guids)
    
def collate_fn(batch):
    # print(len(batch[0]))
    is_train = True if len(batch[0])==4 else False
    guids = [b[0] for b in batch]
    texts = [torch.LongTensor(b[1]) for b in batch]
    imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch])
    if is_train:
        labels = torch.IntTensor([b[3] for b in batch])
    else:
        labels = None
    ## ......对于文本需要进行pad处理......
    texts_mask = [torch.ones_like(text) for text in texts]

    padded_texts = pad_sequence(texts,batch_first=True,padding_value=0)
    padded_text_mask = pad_sequence(texts_mask,batch_first=True,padding_value=0)

    if is_train:
        return (guids,padded_texts,padded_text_mask,imgs,labels)
    else:
        return (guids,padded_texts,padded_text_mask,imgs)