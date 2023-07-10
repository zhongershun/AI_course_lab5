import torch
import os
class config:
    #目录信息
    root_path = os.getcwd()
    train_data_path = os.path.join(root_path, 'data/train.txt')
    test_data_path = os.path.join(root_path, 'data/test_without_label.txt')
    output_test_path = os.path.join(root_path, 'data/test_with_label.txt')
    load_model_path = os.path.join(root_path,'./saveModel/dev_model.pt')

    # 数据集信息
    BATCH_SIZE = 16
    TRAIN_NUM = 3200
    VALID_NUM = 800

    mode = "train"
    epoch = 10
    learning_rate = 3e-5
    weight_decay = 0
    
    middle_hidden = 64
    out_hidden = 128
    num_classes = 3
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    loss_fc = None
    optimizer = None
    
    # 消融用时输入img或只输入text
    only = None

    # BERT相关
    bert_learning_rate = 5e-6
    bert_dropout = 0.2
    fixed_text_model_params = False

    # ResNet相关
    fixed_image_model_params = True
    resnet_learning_rate = 5e-6
    resnet_dropout = 0.2
    img_hidden_seq = 64
    