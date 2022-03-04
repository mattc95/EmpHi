import os
import sys
from parlai.scripts.eval_model import eval_model
from parlai.scripts.eval_model import setup_args as parlai_setupargs
import warnings

warnings.filterwarnings('ignore')

sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def setup_args():
    parser = parlai_setupargs()
    parser.set_defaults(

        task='empathetic_dialogues',
        datatype='test',
        train_experiencer_only=True,
        
        model='agents.transformer:TransAgent',
        model_file='paras/transformer_2_10_256/model',

        n_segments=0,
        n_heads=2, n_layers=10, n_positions=256, text_truncate=256,
        label_truncate=64, ffn_size=256, embedding_size=300,
        dict_lower=True,
        learn_positional_embeddings=True,

        inference='beam',
        beam_size=5,
        embedding_type='glove',

        multitask=False,

        batchsize=32,
        skip_generation=False,
        rank_candidates=False,
        metrics='ppl',
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    eval_model(opt)

