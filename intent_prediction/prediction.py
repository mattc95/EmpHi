import os
import torch
import torch.nn.functional as F
import numpy
from transformers import BertTokenizer


def intent_prediction(model, tokenizer, text):

    """
        Predict intent using BERT intent classifier
    """
    model.eval()
    encodings = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    encodings = {key: value.cuda() for key, value in encodings.items()}
    output = model(**encodings)
    prob = F.softmax(output.logits, dim=-1)
    return prob.argmax(dim=-1), prob

    