import torch
from sklearn.metrics import accuracy_score

class trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = config.device
        self.model = model.to(self.device)

        self.loss_fc = config.loss_fc
        bert_params = set(self.model.text_model.bert.parameters())
        resnet_params = set(self.model.img_model.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - resnet_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.config.bert_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.config.bert_learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.img_model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.config.resnet_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.img_model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.config.resnet_learning_rate, 'weight_decay': 0.0},
            {'params': other_params,
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
        ]

        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate)

    def train(self, train_dataloader,valid_dataloader):
        self.model.train()
        losses = []
        acces = []
        val_losses = []
        best_acc = 0
        for i,(guids,text_masks,texts,imgs,labels) in enumerate(train_dataloader):
            text_masks, texts, imgs, labels = text_masks.to(self.device), texts.to(self.device), imgs.to(self.device), labels.to(self.device)
            # print(text_masks[0])
            # print(imgs[0])
            pred_prob = self.model(texts, text_masks, imgs)
            pred = torch.argmax(pred_prob,dim=1)
            loss = self.loss_fc(pred_prob,labels.long())
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i%10 == 0:
                val_loss,acc = self.valid(valid_dataloader)
                print('Iteration %d / %d, loss = %.6f' % (i, 80, loss.item()))
                print('valid_acc : %.4f'%acc)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.model,"./saveModel/dev_model.pt")
                acces.append(acc)
                val_losses.append(val_loss)
                # print(pred)
        train_loss = round(sum(losses)/len(losses), 5)
        return train_loss, losses, val_losses, acces
    
    def valid(self, valid_dataloader):
        self.model.eval()
        val_loss = 0
        total_acc = 0
        with torch.no_grad():
            for i,(guids,text_masks,texts,imgs,labels) in enumerate(valid_dataloader):
                # total_acc = 0
                text_masks, texts, imgs, labels = text_masks.to(self.device), texts.to(self.device), imgs.to(self.device), labels.to(self.device)
                pred_prob = self.model(texts, text_masks, imgs)
                loss = self.loss_fc(pred_prob,labels.long())
                pred = torch.argmax(pred_prob,dim=1)
                labels_np = labels.cpu().numpy()
                pred_np = pred.cpu().numpy()
                acc = accuracy_score(labels_np,pred_np)
                # print(pred)
                total_acc += acc
                val_loss += loss.item()
            self.model.train()
        return val_loss/len(valid_dataloader),total_acc/len(valid_dataloader)
    
    def predict(self,test_dataloader):
        self.model.eval()
        pred_guids = []
        pred_labels = []
        with torch.no_grad():

            for i,(guids,text_masks,texts,imgs) in enumerate(test_dataloader):
                text_masks, texts, imgs = text_masks.to(self.device), texts.to(self.device), imgs.to(self.device)
                pred_prob = self.model(texts, text_masks, imgs)
                pred = torch.argmax(pred_prob,dim=1)
                pred_guids.extend(guids)
                pred_labels.extend(pred.tolist())
        return pred_guids,pred_labels
            