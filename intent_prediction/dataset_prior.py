import torch
from transformers import BertTokenizer

class PriorDataset(torch.utils.data.Dataset):
    def __init__(self, context_encodings, response_encodings, intent):
        self.context_encodings = context_encodings
        self.response_encodings = response_encodings
        self.intent = intent

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.context_encodings.items()}
        for key, val in self.response_encodings.items():
            item['res_' + key] = torch.tensor(val[idx]) 
        item['intent'] = torch.tensor(self.intent[idx])

        return item

    def __len__(self):
        return len(self.intent)


def get_data(path):

    dataset = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for type in ['train', 'valid', 'test']:

        data_path = path + '/' + type + '.txt'
        intent = []
        context = []
        response = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                intent.append(int(line[0]))
                context.append(line[1])
                response.append(line[3])

        context_encodings = tokenizer(context, truncation=True, padding=True)
        response_encodings = tokenizer(response, truncation=True, padding=True)

        dataset[type] = PriorDataset(context_encodings, response_encodings, intent)

    return dataset


if __name__ == '__main__':

    dataset = get_data('./intent_prediction/prior_data')
    print(dataset['test'][1])
    print(len(dataset['train']))