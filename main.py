import os
import torch
import argparse

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import sampler

from Config.config import config

from tool.DataLoader import load_date
from tool.MyDataset import MyDataset
from tool.MyDataset import collate_fn
from tool.Train import trainer
from model.Mymodel import myModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='train',help='train or test the model')
    parser.add_argument('--lr',default=3e-3,help='learning_rate',type=float)
    parser.add_argument('--epoch',default=10,help='epochs of training',type=int)

    parser.add_argument('--text_only', action='store_true', help='predict with only text')
    parser.add_argument('--img_only', action='store_true', help='predict with only img')
    args = parser.parse_args()

    config.mode = args.mode
    config.learning_rate = args.lr
    config.epoch = args.epoch

    if args.img_only:
        config.only = 'img'
    elif args.text_only:
        config.only = 'text'
    else:
        config.only = None
    print(config.only)
    return


def pre_work():
    if config.mode == 'train':
        # print('touch')
        train_guids,train_texts,train_imgs,train_labels = load_date(config.train_data_path,config)
        # print(train_guids.shape)
        train_dataset = MyDataset(train_guids,train_texts,train_imgs,train_labels,config)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=2,
            sampler=sampler.RandomSampler(range(config.TRAIN_NUM))
        )
        valid_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=2,
            sampler=sampler.RandomSampler(range(config.TRAIN_NUM,config.TRAIN_NUM+config.VALID_NUM))
        )
        return train_dataloader, valid_dataloader

    elif config.mode == 'test':
        test_guids,test_texts,test_imgs = load_date(config.test_data_path,config)
        test_dataset = MyDataset(test_guids,test_texts,test_imgs,config=config)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config.BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=2
        )
        return test_dataloader
    

def train(train_dataloader,valid_dataloader):
    model = myModel(config)
    config.loss_fc = torch.nn.CrossEntropyLoss()
    # config.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    Trainer = trainer(config=config, model=model)

    train_losses = []
    alllosses = []
    vallosses = []
    acces = []
    for epoch in range(config.epoch):
        print("epoch : %d" % (epoch))
        train_loss, losses, val_losses, acc = Trainer.train(train_dataloader=train_dataloader,valid_dataloader=valid_dataloader)
        train_losses.append(train_loss)
        alllosses.extend(losses)
        vallosses.extend(val_losses)
        acces.extend(acc)
        # val_loss,acc = trainer.valid(valid_dataloader)
        # print("valid:")
        # print('valid_acc : %.4f'%acc)
        print()
    
    # torch.save(model,"./saveModel/dev_model.pt")
    config.load_model_path = os.path.join(config.root_path,'/saveModel/dev_model.pt')
    train_losses = np.array(train_losses)
    alllosses = np.array(alllosses)
    np.savetxt('./DataLog/loss_pre_epoch.csv', train_losses, delimiter=',', fmt='%.3f')
    np.savetxt('./DataLog/alllosses.csv', alllosses, delimiter=',', fmt='%.3f')
    np.savetxt('./DataLog/vallosses.csv', vallosses, delimiter=',', fmt='%.3f')
    np.savetxt('./DataLog/acces.csv', acces, delimiter=',', fmt='%.2f')
    return

def test(dataloader):
    if config.load_model_path is None:
        print("you should train model before test!")
        exit(0)
    model = torch.load(config.load_model_path)
    config.loss_fc = torch.nn.CrossEntropyLoss()
    # config.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    Trainer = trainer(config=config, model = model)
    if config.only is not None:
        print("%s only valid"%(config.only))
        _,acc = Trainer.valid(dataloader)
        print("%s only acc : %.2f"%(config.only,acc))
    else:
        pred_guids,pred_labels = Trainer.predict(dataloader)
        test_with_labels = pd.DataFrame()
        test_with_labels['guid'] = pred_guids
        pred_tags = []
        for x in pred_labels:
            if x==0:
                pred_tags.append('neutral')
            elif x==1:
                pred_tags.append('negative')
            elif x==2:
                pred_tags.append('positive')
        test_with_labels['tag']=pred_tags
        test_with_labels.to_csv(config.output_test_path,index=False)
    return


if __name__ == "__main__":
    print("parsing args.........")
    print()
    parse_args()
    print("parsing args finished")
    print()
    if config.only is None:
        if config.mode=='train':
            print("loading data ......")
            print()
            train_dataloader,valid_dataloader=pre_work()
            print("loading data finished")
            print()
            print("start training ......")
            print()
            train(train_dataloader=train_dataloader,valid_dataloader=valid_dataloader)
            print("training finished")
            print()
        elif config.mode=='test':
            print("loading data ......")
            print()
            test_dataloader=pre_work()
            print("loading data finished")
            print()
            print("start testing ......")
            print()
            test(dataloader=test_dataloader)
            print("testing finished")
            print()
    else: # 消融实验
        config.mode='train'
        print("loading data ......")
        print()
        _,valid_dataloader=pre_work()
        print("loading data finished")
        print()
        print("strat testing ......")
        print()
        test(dataloader=valid_dataloader)
        print("testing finished")
        print()