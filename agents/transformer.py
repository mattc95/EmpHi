import os
import sys
sys.path.append('..')
import random
import copy
import nltk
import pickle
import numpy as np
import torch
import heapq
import torch.nn.functional as F
import parlai.utils.logging as logging
from parlai.utils.torch import (
    neginf,
    total_parameters,
    trainable_parameters,
    PipelineHelper,
)
from typing import Optional
from parlai.core.metrics import (
    Metric,
    SumMetric,
    AverageMetric,
    BleuMetric,
    FairseqBleuMetric,
    InterDistinctMetric,
    normalize_answer,
)
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.core.torch_agent import Batch, Output
from parlai.core.torch_generator_agent import PPLMetric
from parlai.agents.transformer.transformer import TransformerGeneratorAgent, TransformerGeneratorModel
from parlai.core.params import ParlaiParser
from nltk.translate import bleu_score
from .modules.model import ConditionalTransformer, MultiTaskTransformer
from .history import EmpathyHistory


class TransAgent(TransformerGeneratorAgent):
    """
    Implementation of the Selection Agent.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('Contrastive Agent')

        agent.add_argument(
            '--multitask',
            type=bool,
            default='False',
            help='Whether use emotion classification',
        )

        agent.add_argument(
            '--alpha',
            type=float,
            default=0.1,
            help='The coefficient of KL divergence',
        )

        super(TransAgent, cls).add_cmdline_args(parser, partial_opt)

        return agent

    def __init__(self, opt: Opt, shared=None):

        self.task = str.lower(opt['task'].split(':')[-1])

        SEED = 46
        random.seed(SEED)
        np.random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        torch.random.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For hyper-parameters
        self.multitask = opt.get('multitask', False)
        self.alpha = opt.get('alpha', 0.1)
        self.map = {
        'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
        'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14,
        'disappointed': 15, 'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
        'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27,
        'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31
        }
        self.rev_map = {value: key for key, value in self.map.items()}

        super().__init__(opt, shared)

        if shared is None:
            self.emo_criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.output = open('./multitask_transformer_output.txt', 'w')
        else:
            self.emo_criterion = shared['emo_criterion']
            self.output = shared['output']

        self.id = 'TransAgent'

        self.reset()

    def build_model(self, states=None):
        """
        Build and return model.
        """

        self.dict.add_additional_special_tokens(['<QRY>'])

        if self.multitask is True:
            model = MultiTaskTransformer(self.opt, self.dict)
        else:
            model = TransformerGeneratorModel(self.opt, self.dict)

        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    @classmethod
    def history_class(cls):
        """
        Override from the TorchAgent
        Return the history class that this agent expects to use.

        """
        return EmpathyHistory

    def observe(self, observation):

        """
        Override from TorchAgent:
            1. Update history and persona according to the observation
            2. Vectorize text and persona
        :param observation:
        :return: obs including vectorization
        """
        obs = Message(observation)
        obs = super().observe(obs)
        if 'emotion' in obs:
            obs['emotion_label'] =  self.map[obs['emotion']]
        self.observation = obs
        return obs

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """

        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs['full_text'] = history_string
            if history_string:
                obs['text_vec'], obs['text_segment'] = history.get_history_vec()
                obs['full_text_vec'], _ = history.get_history_vec()

            obs['text_vec'].insert(0, self.dict.tok2ind['<QRY>'])
            obs['text_segment'].insert(0, 3)

        # check truncation
        if obs.get('text_vec') is not None:
            truncate_left = not self.history_reversed
            truncated_text_vec = self._check_truncate(
                obs['text_vec'], truncate, truncate_left
            )
            truncated_text_segment = self._check_truncate(
                obs['text_segment'], truncate, truncate_left
            )

            obs.force_set('text_vec', torch.LongTensor(truncated_text_vec))
            obs.force_set('text_segment', torch.LongTensor(truncated_text_segment))

        return obs

    def batchify(self, obs_batch, sort=False):

        batch = super().batchify(obs_batch, sort=False)

        if batch.batchsize == 0:
            return batch

        device = batch.text_vec.device
        exs = batch.observations

        segment = None
        if any(ex.get('text_segment') is not None for ex in exs):
            segment = [ex.get('text_segment', self.EMPTY) for ex in exs]
            segment, _ = self._pad_tensor(segment)

        emotion_label = None
        if any(ex.get('emotion_label') is not None for ex in exs):
            emotion_label = [ex.get('emotion_label', self.EMPTY) for ex in exs]
            emotion_label = torch.LongTensor(emotion_label).to(device)

        batch = Batch(
            **dict(batch),
            segment=segment,
            emotion_label=emotion_label,
        )
        return batch

    def _dummy_batch(self, batchsize, maxlen):

        batch = super()._dummy_batch(batchsize, maxlen)
        segment = torch.zeros_like(batch.text_vec).to(batch.text_vec.device)
        batch = Batch(
            **dict(batch),
            segment=segment
        )
        return batch

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, encoder_states = model_output
        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        if self.multitask is True and 'emotion_label' in batch:
            emo_logits = self.model.classification(encoder_states[0][:, 0])
            emo_loss = self.emo_criterion(emo_logits, batch.emotion_label)
            emo_preds = emo_logits.argmax(dim=-1)
            emo_correct = (emo_preds == batch.emotion_label)
            self.record_local_metric('emo_loss', AverageMetric.many(emo_loss))
            self.record_local_metric('emo_correct', AverageMetric.many(emo_correct))
            total_loss = loss / target_tokens + self.alpha * emo_loss
            self.record_local_metric('loss', AverageMetric.many(total_loss))

        self.record_local_metric('nll', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token

        if self.multitask is True and 'emotion_label' in batch:
            loss += self.alpha * emo_loss.mean()

        if return_output:
            return (loss, model_output)
        else:
            return loss

    def translate(self, label):
        
        label = self.dict.parse(label.lower())
        label = self._v2t(label)
        return label

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
            preds, scores = zip(*beam_preds_scores)
            self._add_generation_metrics(batch, preds)

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            for beam in beams:
                beam_texts.append([])
                for tokens, score in beam.get_rescored_finished():
                    try:
                        beam_texts[-1].append((self._v2t(tokens), score.item()))
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

            self._add_bleu_samples(batch, beam_texts)


        text = [self._v2t(p) for p in preds] if preds is not None else None
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
            self._compute_nltk_bleu(batch, text)
        retval = Output(text, token_losses=token_losses)
        if not self.skip_generation:
            retval.beam_texts = beam_texts
        return retval

    def _add_generation_metrics(self, batch, preds):
        
        batchsize = batch.batchsize
        bleu = {key: list() for key in range(1, 5)}

        for i in range(batchsize):
            # print(batch.observations[i]['text'])
            reference = [batch.labels[i].lower()]
            text = self._v2t(preds[i])

            for k in range(1, 5):
                weights = [1 / k for _ in range(k)]
                if len(normalize_answer(text).split(" ")) == 1:
                    score = 0
                else:
                    score = bleu_score.sentence_bleu(
                        [normalize_answer(a).split(" ") for a in reference],
                        normalize_answer(text).split(" "),
                        smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method7,
                        weights=weights,
                    )

                bleu[k].append(score)

        for k in range(1, 5):
            self.record_local_metric(f'bleu_{k}', bleu[k])

        text = [self._v2t(p) for p in preds] if preds is not None else None
        for i, exs in enumerate(batch.observations):

            emoion = 'Emotion: ' + exs['emotion'] + '\n' 
            context = 'Context: ' + exs['full_text'].lower().replace('\n', ' [SEP] ') + '\n'
            generated = 'Generated: ' + text[i] + '\n'
            reference = 'Reference: ' + self.translate(batch.labels[i]) + '\n'

            self.output.write(emoion)
            self.output.write(context)
            self.output.write(generated)
            self.output.write(reference)
            self.output.write('\n')

    def _add_bleu_samples(self, batch, beam_preds):
        
        batchsize = batch.batchsize

        precision_bleu = {key: list() for key in range(1, 5)}
        recall_bleu = {key: list() for key in range(1, 5)}

        for i in range(batchsize):
            
            reference = [batch.labels[i].lower()]
            preds = beam_preds[i]
            
            # print('Context: ')
            # print(batch.observations[i]['full_text'])
            # print('Generated: ')
            # for j in range(sample_num):
            #     print(self.intents[sample_prior_intent[i][j].item()])
            #     print(texts[j])

            # print('\n')

            # print(reference)

            # print([normalize_answer(a).split(" ") for a in reference])

            for k in range(1, 5):
                weights = [1 / k for _ in range(k)]
                max_bleu = 0
                mean_bleu = []
                for text, _ in preds:
                    
                    # print(normalize_answer(text).split(" "))
                    if len(normalize_answer(text).split(" ")) == 1:
                        score = 0
                    else:
                        score = bleu_score.sentence_bleu(
                            [normalize_answer(a).split(" ") for a in reference],
                            normalize_answer(text).split(" "),
                            smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method7,
                            weights=weights,
                        )
                    
                    max_bleu = max(score, max_bleu)
                    mean_bleu.append(score)

                if len(mean_bleu) == 0:
                    precision_bleu[k].append(0)
                else:
                    precision_bleu[k].append(sum(mean_bleu)/len(mean_bleu))
                recall_bleu[k].append(max_bleu)

        for k in range(1, 5):
            self.record_local_metric(f'pre_bleu_{k}', precision_bleu[k])
            self.record_local_metric(f'rec_bleu_{k}', recall_bleu[k])



    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        try:
            model_dict = self.model.state_dict()
            new_model_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
            model_dict.update(new_model_dict)
            self.model.load_state_dict(model_dict)
        except RuntimeError as msg:
            msg_ = str(msg)
            if 'size mismatch' in msg_ and 'embedding' in msg_:
                if hasattr(self, 'special_toks') and len(self.special_toks) > 0:
                    state_dict = self._resize_token_embeddings(state_dict, msg_)
                    self.model.load_state_dict(state_dict)
                    self.resized_embeddings = True  # make note that we resized here
                else:
                    raise RuntimeError(
                        f'{msg_}\n'
                        '-----------------\n'
                        'Could not load the model due to a size mismatch in the '
                        'embeddings. A common reason for this is trying to load '
                        'a model trained with fp16 but loaded without fp16. Try '
                        'adding --fp16 true or --force-fp16-tokens true.'
                    )
            else:
                raise

    def report(self):
        """
        Override from TorchAgent
        Simplify the output
        :return:
        """
        report = super().report()
        new_report = dict()
        if 'lr' in report.keys():
            new_report['lr'] = report['lr']

        return new_report

    def share(self):

        shared = super().share()
        shared['emo_criterion'] = self.emo_criterion
        shared['output'] = self.output
        return shared

