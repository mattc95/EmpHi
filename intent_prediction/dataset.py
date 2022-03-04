import torch
from transformers import BertTokenizer

"""
    Preprocess the Empathetic Intents dataset
"""


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_data(path):

    dataset = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for type in ['train', 'valid', 'test']:

        data_path = path + '/' + type + '.txt'
        text = []
        label = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                label.append(int(line[0]))
                text.append(line[1])
        encodings = tokenizer(text, truncation=True, padding=True)
        dataset[type] = IntentDataset(encodings, label)

    return dataset
