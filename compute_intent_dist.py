import os
import torch
import math
from transformers import BertTokenizer
from intent_prediction.prediction import intent_prediction

emotions = ['afraid',
          'angry',
          'annoyed',
          'anticipating',
          'anxious',
          'apprehensive',
          'ashamed',
          'caring',
          'confident',
          'content',
          'devastated',
          'disappointed',
          'disgusted',
          'embarrassed',
          'excited',
          'faithful',
          'furious',
          'grateful',
          'guilty',
          'hopeful',
          'impressed',
          'jealous',
          'joyful',
          'lonely',
          'nostalgic',
          'prepared',
          'proud',
          'sad',
          'sentimental',
          'surprised',
          'terrified',
          'trusting',
        ]
gen_emotion_intent_num = {key: [0]*9 for key in emotions}
ref_emotion_intent_num = {key: [0]*9 for key in emotions}
cur_emotion = None

intent_bert = torch.load('./intent_prediction/paras.pkl').cuda()
intent_bert.eval()

intent_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num = 0
with open('./new_output.txt') as f:
    for line in f:
        line = line.strip()
        x = line.split('Emotion: ')
        # if (num > 500):
        #     break
        if len(x) > 1:
            num += 1
            print(num)
            cur_emotion = x[1]
            continue
        y = line.split('Generated: ')
        if len(y) > 1:
            generated = y[1]
            intent, _ = intent_prediction(intent_bert, intent_tokenizer, generated)
            gen_emotion_intent_num[cur_emotion][intent] += 1
            continue
        z = line.split('Reference: ')
        if len(z) > 1:
            reference = z[1]
            intent, _ = intent_prediction(intent_bert, intent_tokenizer, reference)
            ref_emotion_intent_num[cur_emotion][intent] += 1           

for key, value in ref_emotion_intent_num.items():
    kl = 0
    num = sum(value)
    for i in range(9):
        pr = value[i] / num
        value[i] = pr
        pg = gen_emotion_intent_num[key][i] / num 
        gen_emotion_intent_num[key][i] = pg
        kl += pr * math.log((pr+1e-10) / (pg + 1e-10))
    gen_emotion_intent_num[key].append(kl)    

with open('./intent_prediction/emotion_intent/ours_output_0.00.txt', 'w') as f:
    for key, value in gen_emotion_intent_num.items():
        string = key
        for i in value:
            string = string + ' ' + str(i)
        string += '\n'
        f.write(string)

# with open('./intent_prediction/emotion_intent/train_reference.txt', 'w') as f:
#     for key, value in ref_emotion_intent_num.items():
#         string = key
#         for i in value:
#             string = string + ' ' + str(i)
#         string += '\n'
#         f.write(string)
 
