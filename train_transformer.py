import os
import sys

from parlai.scripts.train_model import TrainLoop
from parlai.scripts.train_model import setup_args as train_setupargs
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import warnings

warnings.filterwarnings('ignore')

def setup_args():
    """
    Defaults for baseline model.
    """
    parser = train_setupargs()

    parser.set_defaults(

        task='empathetic_dialogues',
        train_experiencer_only='True',

        model='agents.transformer:TransAgent',
        model_file='paras/transformer_2_10_256/model',

        n_segments=0,
        n_heads=2, n_layers=10, n_positions=256, text_truncate=256,
        label_truncate=64, ffn_size=256, embedding_size=300,
        dict_lower=True,
        learn_positional_embeddings=True,

        embedding_type='glove',


        multitask=False,
        alpha=0.5,
        batchsize=16,
        skip_generation=True,
        rank_candidates=False,
        metrics='ppl,bleu',
        learningrate=1e-4,
        optimizer='adam',
        num_epochs=50,
        warmup_updates=1000,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_every_n_epochs=0.25,
        validation_patience=3,
    )
    return parser


if __name__ == '__main__':

    # print(TrainModel.help(model='transformer/generator'))
    parser = setup_args()
    opt = parser.parse_args()
    path = opt['model_file'].split('/model')[0]
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    TrainLoop(opt).train()



