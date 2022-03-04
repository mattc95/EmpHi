import os
import numpy
from parlai.core.metrics import InterDistinctMetric

dist_1 = None
dist_2 = None
with open('./multitask_transformer_output.txt') as f:
    for line in f:
        if len(line.split('Generated:')) < 2:
            continue

        x, text = line.split('Generated:')
        
        if dist_1 is None:
            dist_1 = InterDistinctMetric.compute(text, ngram=1)
            dist_2 = InterDistinctMetric.compute(text, ngram=2)
        else:
            dist_1 = dist_1 + InterDistinctMetric.compute(text, ngram=1)
            dist_2 = dist_2 + InterDistinctMetric.compute(text, ngram=2)

    print(dist_1.value())
    print(dist_2.value())