import os
import sys
from parlai.scripts.eval_model import eval_model
from parlai.scripts.eval_model import setup_args as parlai_setupargs
import warnings

warnings.filterwarnings('ignore')

sys.path.append('..')

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def setup_args():
    parser = parlai_setupargs()
    parser.set_defaults(

        task='empathetic_dialogues',
        datatype='test',
        train_experiencer_only=True,
        

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

        num_layers=2,

        embedding_type='glove',
        embedding_size=300,

        batchsize=16,
        skip_generation=False,
        rank_candidates=False,
        metrics='ppl',
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    eval_model(opt)
