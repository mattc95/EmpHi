import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

intent_groups = 9

emotion = ['afraid',
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

intents = [
          'agreeing',
          'acknowledging',
          'encouraging',
          'consoling',
          'sympathizing',
          'suggesting',
          'questioning',
          'wishing',
          'neutral']

truth_emotion_intent_dist = {key: [] for key in emotion}
ours_emotion_intent_dist = {key: [] for key in emotion}
ours_kl = []
baseline_emotion_intent_dist = {key: [] for key in emotion}
baseline_kl = []

with open('./intent_prediction/emotion_intent/ours_output_0.00.txt') as f, \
        open('./intent_prediction/emotion_intent/moel.txt') as g, \
        open('./intent_prediction/emotion_intent/reference.txt') as k:

    for line in k:
        line = line.split()
        for x in line[1: ]:  
            truth_emotion_intent_dist[line[0]].append(float(x))

    for line in f:
        line = line.split()
        ours_kl.append(float(line[-1]))
        for x in line[1: -1]:
            ours_emotion_intent_dist[line[0]].append(float(x))

    for line in g:
        line = line.split()
        baseline_kl.append(float(line[-1]))
        for x in line[1: -1]:
            baseline_emotion_intent_dist[line[0]].append(float(x))
    
    for i, emotion in enumerate(emotion):

        truth_prob = truth_emotion_intent_dist[emotion]
        ours_prob = ours_emotion_intent_dist[emotion]
        baseline_prob = baseline_emotion_intent_dist[emotion]

        fig, ax = plt.subplots()

        index = np.arange(intent_groups)
        bar_width = 0.35

        opacity = 0.4

        rects1 = ax.bar(index, truth_prob, bar_width,
                        alpha=opacity, color='b',
                        label='Human')

        rects2 = ax.bar(index + bar_width, ours_prob, bar_width,
                        alpha=opacity, color='r',
                        label='EmpHi')

        ax.set_xlabel('Intent Index \n KL-divergence is %f' % (ours_kl[i]))
        ax.set_ylabel('Prob')
        ax.set_title('Emotion: %s ' % (emotion))
        ax.set_xticks(index + bar_width/2)
        ax.set_xticklabels(range(9))
        ax.legend()

        fig.tight_layout()
        plt.show()

        plt.savefig('./intent_prediction/intent_vision_ours_0.00/%s.png' % (emotion))