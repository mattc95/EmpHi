import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import \
    BertForSequenceClassification, BertConfig
from transformers import AdamW
from prior_dataset import get_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# hyp
EPOCH_NUM = 50
BATCH_SIZE = 16
LR = 5*1e-4

# data
dataset = get_data('./intent_prediction/prior_data')
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset['valid'], batch_size=4*BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset['test'], batch_size=4*BATCH_SIZE, shuffle=False)

# posterior_bert = torch.load('./intent_prediction/paras_1.pkl').cuda()

def train_step(model, optimizer, train_loader, num):

    batch_num = 0
    item_num = 0
    total_loss = 0
    correct_num = 0
    model.train()
    for batch in train_loader:

        for key, value in batch.items():
            batch[key] = value.cuda()

        optimizer.zero_grad()
        output = model(input_ids=batch['input_ids'], \
        token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['intent'])

        loss = output.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.logits.argmax(dim=-1)
        correct_num += (pred == batch['intent']).sum().item()

        batch_num += 1
        item_num += pred.size(0)

    loss = total_loss / batch_num
    acc = correct_num / item_num
    print('Train %d \t ACC %f \t Loss %f' % (num, acc, loss))


def eval_step(model, eval_loader, valid=True):

    batch_num = 0
    item_num = 0
    total_loss = 0
    correct_num = 0
    prior_vocab = {key: 0 for key in range(9)}
    ref_vocab = {key: 0 for key in range(9)}
    model.eval()
    for batch in eval_loader:

        for key, value in batch.items():
            batch[key] = value.cuda()

        output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], \
        attention_mask=batch['attention_mask'], labels=batch['intent'])

        loss = output.loss
        total_loss += loss.item()

        pred = output.logits.argmax(dim=-1)
        correct_num += (pred == batch['intent']).sum().item()

        for i in range(pred.size(0)):
            prior_vocab[pred[i].item()] += 1
            ref_vocab[batch['intent'][i].item()] += 1 

        batch_num += 1
        item_num += pred.size(0)

    loss = total_loss / batch_num
    acc = correct_num / item_num


    """
        Dataset KL Divergence
    """

    if valid is True:
        print('Valid \t ACC %f \t Loss %f' % (acc, loss))
    else:
        print('Test \t ACC %f \t Loss %f' % (acc, loss))
    
    print(prior_vocab)
    print(ref_vocab)

    return acc

def main():

    print('Start !')

    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 9
    model = BertForSequenceClassification(config).cuda()
    optimizer = AdamW(model.parameters(), lr=LR)
    for param in model.base_model.parameters():
        param.requires_grad = False
    best = -1
    for i in range(EPOCH_NUM):
        train_step(model, optimizer, train_loader, i+1)
        result = eval_step(model, valid_loader, valid=True)
        if result > best:
            best = result
            torch.save(model, './intent_prediction/prior_paras_2.pkl')
            print('New Best')

    print('Load best model')
    model = torch.load('./intent_prediction/prior_paras_2.pkl').cuda()
    _ = eval_step(model, test_loader, valid=False)


if __name__ == '__main__':

    main()
