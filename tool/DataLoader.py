from torchvision import transforms
from transformers import AutoTokenizer

from PIL import Image
import numpy as np
import pandas as pd
import chardet
import re

def load_date(path,config):
    train_raw = pd.read_csv(path)
    train_raw_top3 = train_raw[:3]
    texts = []
    imgs = []
        
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    img_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for guid in train_raw['guid']:
        ## ......文本部分......

        ## ...对于消融过程中设置text_raw=''...
        if config.only == 'img':
            text_raw = ''
        else:
            ## ...从文件中读取txt...
            text_file = open("./data/data/text/"+str(guid)+".txt",'rb')
            text_byte = text_file.read()
            encode = chardet.detect(text_byte)
            try:
                text_raw = text_byte.decode(encode['encoding'])
            except:
                text_raw= text_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
            text_raw = text_raw.strip('\n').strip('\r').strip(' ').strip()
            # print("raw:",text_raw)
        
            ## ...删除txt中的链接和一些@xxx的无用信息...
            text_raw = re.sub(r'^(https:\S+)','',text_raw)
            text_raw = re.sub(r'^(http:\S+)','',text_raw)
            text_raw = re.sub(r'[a-zA-Z]+://[^\s]*','',text_raw)
            text_raw = re.sub('@\w+\s?','',text_raw)
            # print(text_raw)
        
            ## ...去除一些无用字符。例如#，@，$等...
            text_raw = text_raw.replace("RT ",'').replace('#','').replace('@','').replace('&','').replace('$','')
            # print(text_raw)

        ## ...添加特殊的token [CLS],[SEP],并利用tokenizer将文本数据转化为向量数据...
        tokens = tokenizer.tokenize('[CLS]'+text_raw+'[SEP]')
        to_ids = tokenizer.convert_tokens_to_ids(tokens)
        to_ids = np.array(to_ids)
        texts.append(to_ids)
        
        ## ......图像部分......

        ## ...对于消融过程中设置利用Image模块生成（3，224，224）的全0图片...
        if config.only == "text":
            img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
        else:
            ## ...读取图像...
            img = Image.open("./data/data/img/"+str(guid)+".jpg")
            img.load()

        ## ...图像处理（reshape，normalize）...
        img_pro = img_transform(img)
        img_pro = np.array(img_pro)
        # print(img_pro)
        # print(img_pro.shape)
        imgs.append(img_pro)

    ## ...对label进行处理...
    def label2id(label):
        if label=='neutral':
            return 0
        elif label=='negative':
            return 1
        elif label=='positive':
            return 2
    

    train_raw['text'] = texts
    train_raw['img'] = imgs

    # imgs = np.array(imgs)
    # texts = np.array(texts)
    guids = np.array(train_raw['guid'])
    
    if config.mode == "train":
        train_raw.to_csv("./dev_data_log/train_txt_img_label.csv",index=False)
        labels = train_raw['tag']
        labels_idx = [label2id(label) for label in labels]
        labels_idx = np.array(labels_idx)
        return guids,texts,imgs,labels_idx
     
    elif config.mode == 'test':
        return guids,texts,imgs