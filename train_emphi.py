import os
import sys
import torch
from parlai.scripts.train_model import TrainLoop
from parlai.scripts.train_model import setup_args as train_setupargs
import warnings
warnings.filterwarnings('ignore')
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.set_num_threads(5)

def setup_args():
    """
    Defaults for baseline model.
    """
    parser = train_setupargs()

    parser.set_defaults(

        task='empathetic_dialogues',
        train_experiencer_only=True,

        model='agents.emphi:EmpHi',
        model_file='paras/emphi_ex_im_dy_0.5_0.5_1_0/model',

        dict_lower=True,
        alpha=0.5,
        gamma=0.5,
        tau=1,
        dropout=0,
        implicit=True,
        implicit_dynamic=True,
        explicit=True,

        num_layers=2,
        embedding_type='glove',
        embedding_size=300,


        batchsize=16,
        skip_generation=True,
        metrics='ppl',
        learningrate=1e-4,
        optimizer='adam',
        num_epochs=50,
        warmup_updates=100,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_every_n_epochs=0.2,
        validation_patience=3,
    )
    return parser


if __name__ == '__main__':

    parser = setup_args()
    opt = parser.parse_args()
    path = opt['model_file'].split('/model')[0]
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    TrainLoop(opt).train()



