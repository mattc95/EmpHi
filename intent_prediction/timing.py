        if self.model.training is True:

            for i in range(batch.batchsize):
                intent = posterior_intent[i].item()
                prob = posterior_prob[i].item()

                self.prior_data[intent].append(
                    (intent, batch.observations[i]['full_text'], batch.observations[i]['text'], batch.labels[i], prob)
                )

        else:
            train_data = list()
            valid_data = list()
            test_data = list()
            
            small = 1e9
            for i in range(9):
                small = min(small, len(self.prior_data[i]))

            print(small)
            
            for i in range(9):
                
                self.prior_data[i].sort(reverse=True, key=lambda x: x[-1])
                random.shuffle(self.prior_data[i])
                train_data.extend(self.prior_data[i][: 1000])
                valid_data.extend(self.prior_data[i][1000: 1100])
                test_data.extend(self.prior_data[i][1100: 1200])

            random.shuffle(train_data)
            random.shuffle(valid_data)
            random.shuffle(test_data)

            with open('./intent_prediction/prior_data/train.txt', 'w') as f:
                for (intent, full_text, text, label, _) in train_data:
                    f.write(str(intent) + '\t' + full_text + '\t' + text + '\t' + label + '\n')

            with open('./intent_prediction/prior_data/valid.txt', 'w') as f:
                for (intent, full_text, text, label, _) in valid_data:
                    f.write(str(intent) + '\t' + full_text + '\t' + text + '\t' + label + '\n')

            with open('./intent_prediction/prior_data/test.txt', 'w') as f:
                for (intent, full_text, text, label, _) in test_data:
                    f.write(str(intent) + '\t' + full_text + '\t' + text + '\t' + label + '\n')









        if self.model.training is True:
            for idx, exs in enumerate(batch.observations):
                emotion = exs['emotion_label']
                self.emotion_data[emotion].append((emotion, exs['full_text']))
                self.emotion_intent_prior[emotion][posterior_intent[idx].item()] += 1
        else:
            
            train_data = []
            valid_data = []
            test_data = []
            data_num = {key: len(value) for key, value in self.emotion_data.items()}
            print(data_num)

            for i in range(32):
                num = len(self.emotion_data[i])
                random.shuffle(self.emotion_data[i])
                train_data.extend(self.emotion_data[i][: int(0.8*num)])
                valid_data.extend(self.emotion_data[i][int(0.8*num): int(0.9*num)])
                test_data.extend(self.emotion_data[i][int(0.9*num): ])

            random.shuffle(train_data)
            random.shuffle(valid_data)
            random.shuffle(test_data)

            with open('./intent_prediction/emotion_data/train.txt', 'w') as f:
                for (emotion, text) in train_data:
                    f.write(str(emotion) + '\t' + text + '\n')

            with open('./intent_prediction/emotion_data/valid.txt', 'w') as f:
                for (emotion, text) in valid_data:
                    f.write(str(emotion) + '\t' + text + '\n')

            with open('./intent_prediction/emotion_data/test.txt', 'w') as f:
                for (emotion, text) in test_data:
                    f.write(str(emotion) + '\t' + text + '\n')   

            with open('./intent_prediction/emotion_data/emotion_intent_prior.txt', 'w') as f:
                
                for key, value in self.emotion_intent_prior.items():
                    s = str(key)
                    num = 0
                    for i in range(9):
                        num += value[i]
                    for i in range(9):
                        s = s + '\t' + str(value[i]/num)
                    s = s + '\n'
                    f.write(s)
                