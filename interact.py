import random
import sys
sys.path.append('..')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import warnings
warnings.filterwarnings('ignore')
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive import Interactive

if __name__ == '__main__':

    # call it with particular args
    Interactive.main(

        model='agents.gru:GRUAgent',
        model_file='paras/gru_7/model',

        num_beams=5,
        mmi=True,
        num_layers=3,
        embedding_size=300,

    )