import json

def readData(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        data = json.load(f)
        for i in range(9):
            intent = data[str(i)]
            print(intent['1gram'])
            print(len(intent['1gram']))
            print(intent['2gram'])
            print(len(intent['2gram']))
            print(intent['3gram'])
            print(len(intent['3gram']))

if __name__ == '__main__':
    readData('./intent_prediction/intent_keywords.json')