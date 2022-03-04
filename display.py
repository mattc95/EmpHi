import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import warnings
warnings.filterwarnings('ignore')
from parlai.scripts.display_model import DisplayModel


if __name__ == '__main__':


    DisplayModel.main(
        task='empathetic_dialogues',
        model='agents.intent_cvae_agent:DiscreteIntentCopyGRUAgent',
        model_file='paras/ex_im_dy_0.5_0.5_1_0_0_2/model',

        dict_lower=True,
        alpha=0.5,
        gamma=0.5,
        tau=1,
        dropout=0,
        exposure=0,
        implicit=True,
        implicit_dynamic=True,
        explicit=True,
        prior_rate=0.5,

        num_layers=2,

        datatype='test',

        embedding_size=300,

        num_examples=100,

    )