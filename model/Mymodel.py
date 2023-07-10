import torch
import torch.nn as nn

import random

from model.BertModel import TextModel
from model.ResNet50 import resnet

class myModel(nn.Module):
    def __init__(self,config):
        super(myModel, self).__init__()
        self.text_model = TextModel(config)
        self.img_model = resnet([3,4,6,3],config)


        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(config.middle_hidden*2,config.out_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.out_hidden,config.num_classes),
            nn.Softmax(dim=1)
        )

        # attention
        # 输入 （seq_len, batch_size, middle_hidden）
        self.text_img_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden,
            num_heads=8, 
            dropout=0.5,
        )
        self.img_text_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden,
            num_heads=8, 
            dropout=0.5,
        )

        # 全连接分类器
        # 输入 （batch，middle_hidden+middle_hidden）
        self.text_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(config.middle_hidden*2, config.out_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.out_hidden, config.num_classes),
            nn.Softmax(dim=1)
        )
        self.img_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(config.middle_hidden*2, config.out_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.out_hidden, config.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, text_mask,text,img):
        # print(text[0])
        text_hidden_state, text_feature = self.text_model(text, text_mask)

        img_hidden_state, img_feature = self.img_model(img)

        text_hidden_state = text_hidden_state.permute(1, 0, 2)
        img_hidden_state = img_hidden_state.permute(1, 0, 2)

        text_img_attention_out, _ = self.img_text_attention(img_hidden_state, \
            text_hidden_state, text_hidden_state)
        text_img_attention_out = torch.mean(text_img_attention_out, dim=0).squeeze(0)
        img_text_attention_out, _ = self.text_img_attention(text_hidden_state, \
            img_hidden_state, img_hidden_state)
        img_text_attention_out = torch.mean(img_text_attention_out, dim=0).squeeze(0)

        text_prob_vec = self.text_classifier(torch.cat([text_feature, img_text_attention_out], dim=1))
        img_prob_vec = self.img_classifier(torch.cat([img_feature, text_img_attention_out], dim=1))
        # (batch,num_classes=3)

        # prob = self.classifier(torch.cat([img_feature, text_feature], dim = 1))

        # prob = torch.softmax(prob,dim=1)

        # w = random.random()

        prob = torch.softmax((text_prob_vec+img_prob_vec),dim=1)
        
        # pred_labels = torch.argmax(prob, dim=1)
        # (batch,)
        
        return prob
        