import torch
import torch.nn as nn
from transformers import AutoModel

class TextModel(nn.Module):

    def __init__(self,config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-cased')

        self.trans = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, config.middle_hidden),
            nn.ReLU(inplace=True)
        ) 

        # 是否进行fine-tune
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, bert_inputs, masks):
        # assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_out = self.bert(input_ids=bert_inputs,attention_mask=masks)
        # print(bert_out)
        hidden_state = bert_out[0]
        pooler_out = bert_out[1]
        
        return self.trans(hidden_state),self.trans(pooler_out)
    
# if __name__=='__main__':
#     model = TextModel()

#     input = torch.randn(3, 37).long()
#     masks = torch.randn(3, 37).long()
#     print(input)
#     print(masks)
#     hidden_state,pooler_out = model(input,masks)
#     print(hidden_state)
#     print(pooler_out)

