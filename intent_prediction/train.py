import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import \
    BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import AdamW
from .dataset import get_data


"""
    Train a BERT intent classifier by Hugging Transformer
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# hyp
EPOCH_NUM = 50
BATCH_SIZE = 16
LR = 1e-5

# data
dataset = get_data('./data')
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset['valid'], batch_size=4*BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset['test'], batch_size=4*BATCH_SIZE, shuffle=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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
        output = model(**batch)
        loss = output.loss
        total_loss += loss.item()
        loss.backward()

        preds = output.logits.argmax(dim=-1)
        correct_num += (preds == batch['labels']).sum().item()
        optimizer.step()

        batch_num += 1
        item_num += batch['labels'].size(0)

    loss = total_loss / batch_num
    acc = correct_num / item_num
    print('Train %d \t ACC %f \t Loss %f' % (num, acc, loss))


def eval_step(model, eval_loader, valid=True):

    batch_num = 0
    item_num = 0
    total_loss = 0
    correct_num = 0
    vocab_num = {key: 0 for key in range(9)}
    vocab_str = {key: list() for key in range(9)}
    model.eval()
    for batch in eval_loader:

        for key, value in batch.items():
            batch[key] = value.cuda()

        output = model(**batch)
        loss = output.loss
        total_loss += loss.item()

        preds = output.logits.argmax(dim=-1)

        if valid is False:
            for i in range(preds.shape[0]):
                vocab_num[preds[i].item()] += 1
                vocab_str[preds[i].item()].append(tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True))


        correct_num += (preds == batch['labels']).sum().item()
        batch_num += 1
        item_num += batch['labels'].size(0)

    loss = total_loss / batch_num
    acc = correct_num / item_num
    if valid is True:
        print('Valid \t ACC %f \t Loss %f' % (acc, loss))
    else:
        print('Test \t ACC %f \t Loss %f' % (acc, loss))
        print(vocab_num)
        for key, value in vocab_str.items():
            print(key)
            for v in value[: 10]:
                print(v)

    return acc

def main():

    print('Start !')

    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 9
    model = BertForSequenceClassification(config).cuda()
    optimizer = AdamW(model.parameters(), lr=LR)
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    best = -1
    # for i in range(EPOCH_NUM):
    #     train_step(model, optimizer, train_loader, i+1)
    #     result = eval_step(model, valid_loader, valid=True)
    #     if result > best:
    #         best = result
    #         torch.save(model, './paras_1.pkl')
    #         print('New Best')

    print('Load best model')
    model = torch.load('./paras.pkl').cuda()
    _ = eval_step(model, test_loader, valid=False)


if __name__ == '__main__':

    main()
