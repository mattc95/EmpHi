import os
import json
import random
import copy
import nltk
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
import parlai.utils.logging as logging
from typing import Optional
from parlai.core.metrics import (
    AverageMetric,
    BleuMetric,
    InterDistinctMetric,
    normalize_answer,
)
from parlai.utils.misc import warn_once
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.torch_agent import Batch, Output
from parlai.core.torch_generator_agent import PPLMetric, TorchGeneratorAgent
from parlai.core.params import ParlaiParser
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AdamW
from nltk.translate import bleu_score
from intent_prediction.prediction import intent_prediction
from .modules.intent_cvae import IntentCVAEGRUModel
from .modules.gumbel_softmax import gumbel_softmax_sample

class EmpHi(TorchGeneratorAgent):
    """
    Implementation of EmpHi
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('EmpHi')

        agent.add_argument(
            '--alpha',
            type=float,
            default=0.5,
            help='The coefficient of intent kl-divergence',
        )

        agent.add_argument(
            '--gamma',
            type=float,
            default=0.5,
            help='The coefficient of emotion prediction loss',
        )

        agent.add_argument(
            '--tau',
            type=float,
            default=1,
            help='The coefficient of copy rate loss',
        )

        agent.add_argument(
            '--implicit',
            type=bool,
            default=True,
            help='Whether use implicit intent embedding',
        )

        agent.add_argument(
            '--implicit_dynamic',
            type=bool,
            default=True,
            help='Whether use the gate operation for dynamically using intent embedding',
        )

        agent.add_argument(
            '--explicit',
            type=bool,
            default=True,
            help='Whether use copy mechanism',
        )

        agent.add_argument(
            '--dropout',
            type=float,
            default=0,
            help='Dropout ratio of GRU',
        )


        agent.add_argument(
            '--num_layers',
            type=int,
            default=2,
            help='Layer num of gru',
        )

        agent.add_argument(
            '--embedding_size',
            type=int,
            default='300',
            help='Size of word embedding',
        )

        super(EmpHi, cls).add_cmdline_args(parser, partial_opt)

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
        self.alpha = opt.get('alpha', 0.1)
        self.gamma = opt.get('gamma', 0.1)
        self.tau = opt.get('tau', 0.5)
        self.implicit = opt.get('implicit', False)
        self.implicit_dynamic = opt.get('implicit_dynamic', False)
        self.explicit = opt.get('explicit', False)
        self.dropout = opt.get('dropout', False)

        self.emotions = ['afraid',
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

        self.emotion_vocab = {self.emotions[idx]: idx for idx in range(32)} 

        self.intents = [
          'agreeing',
          'acknowledging',
          'encouraging',
          'consoling',
          'sympathizing',
          'suggesting',
          'questioning',
          'wishing',
          'neutral']
        
        super().__init__(opt, shared)

        if shared is None:
            
            # Load BERT intent classifier
            self.recog_bert = torch.load('./intent_prediction/paras.pkl').cuda()
            self.recog_bert.eval()
            
            self.intent_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.intent_criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.emotion_criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.copy_criterion = torch.nn.BCELoss(reduction='none')

            # Load intent keywords
            with open('./intent_prediction/intent_keywords.json', 'r', encoding='utf-8') as f:
                intent_keywords = json.load(f)
            
            intent_1gram = list()
            for i in range(9):
                for key, value in intent_keywords[str(i)].items():

                    value = value[: 30]
                    if key == '1gram':
                        gram = [self.dict.txt2vec(x)[0] for x, _ in value]
                        intent_1gram.append(gram)            
        

            self.intent_keywords = torch.LongTensor(intent_1gram).to('cuda')

        else:

            self.recog_bert = shared['recog_bert']
        
            self.intent_keywords = shared['intent_keywords']
            self.intent_tokenizer = shared['intent_tokenizer']
            self.intent_criterion = shared['intent_criterion']
            self.emotion_criterion = shared['emotion_criterion']
            self.copy_criterion = shared['copy_criterion']

        self.id = 'EmpHi'

        self.reset()

    def build_model(self, states=None):
        """
        Build and return model.
        """

        model = IntentCVAEGRUModel(self.opt, self.dict)

        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def observe(self, observation):

        obs = Message(observation)
        obs = super().observe(obs)
        if 'emotion' in obs:
            obs['emotion_label'] =  self.emotion_vocab[obs['emotion']]
        self.observation = obs
        return obs

    def batchify(self, obs_batch, sort=False):

        """
            Creat a batch of data for training or evaluation
        """

        batch = super().batchify(obs_batch, sort=False)

        if batch.batchsize == 0:
            return batch

        bsz = batch.batchsize
        device = batch.text_vec.device
        exs = batch.observations
        if any(ex.get('emotion_label') is not None for ex in exs):
            emotion_label = [ex.get('emotion_label', self.EMPTY) for ex in exs]
            emotion_label = torch.LongTensor(emotion_label).to(device)

        recog_intent, _ = intent_prediction \
        (self.recog_bert, self.intent_tokenizer, batch.labels)

        copy_label = [list() for _ in exs]
        for i, _ in enumerate(exs):
            
            intent = recog_intent[i].item()
            for idx in batch.label_vec[i]:
                idx = idx.item()
                if idx in self.intent_keywords[intent]:
                    copy_label[i].append(1)
                else:
                    copy_label[i].append(0)
        copy_label = torch.FloatTensor(copy_label).to(device)

        batch = Batch(
            **dict(batch),
            emotion_label=emotion_label,
            copy_label=copy_label,
        )
        return batch

    def _encoder_input(self, batch):

        return (batch.text_vec, batch.text_lengths)

    def compute_loss(self, batch, return_output=False):

        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model = self.model

        batchsize = batch.batchsize
        encoder_states = model.encoder(*self._encoder_input(batch))
        
        """
            Intent prediction loss
        """
        recog_intent, recog_intent_prob = intent_prediction \
        (self.recog_bert, self.intent_tokenizer, batch.labels)
        
        state = encoder_states[1]
        prior_intent_logit = self.model.intent_prediction(state[-1])
        prior_intent_prob = F.softmax(prior_intent_logit, dim=-1)
        intent_loss = (recog_intent_prob * (torch.log(recog_intent_prob) - torch.log(prior_intent_prob))).sum(dim=-1)

        self.record_local_metric('intent_loss', AverageMetric.many(intent_loss))

        prior_top1 = (recog_intent == prior_intent_prob.argmax(dim=-1))
        prior_top2 = prior_intent_prob.topk(k=2, dim=-1, largest=True).indices
        prior_top3 = prior_intent_prob.topk(k=3, dim=-1, largest=True).indices

        prior_top2 = (recog_intent.unsqueeze(dim=-1).expand_as(prior_top2) == prior_top2).sum(dim=-1)
        prior_top3 = (recog_intent.unsqueeze(dim=-1).expand_as(prior_top3) == prior_top3).sum(dim=-1)

        self.record_local_metric('intent_top1', AverageMetric.many(prior_top1))
        self.record_local_metric('intent_top2', AverageMetric.many(prior_top2))
        self.record_local_metric('intent_top3', AverageMetric.many(prior_top3))

        """
            Emotion classification loss
        """        

        emotion_logit = self.model.emotion_prediction(state[-1])
        emotion_pred = F.softmax(emotion_logit, dim=-1)
        emotion_label = batch.emotion_label
        emotion_loss = self.emotion_criterion(emotion_logit, emotion_label)
        self.record_local_metric('emotion_loss', AverageMetric.many(emotion_loss))

        emotion_top1 = (emotion_label == emotion_pred.argmax(dim=-1))

        emotion_top2 = emotion_pred.topk(k=2, dim=-1, largest=True).indices
        emotion_top3 = emotion_pred.topk(k=3, dim=-1, largest=True).indices

        emotion_top2 = (emotion_label.unsqueeze(dim=-1).expand_as(emotion_top2) == emotion_top2).sum(dim=-1)
        emotion_top3 = (emotion_label.unsqueeze(dim=-1).expand_as(emotion_top3) == emotion_top3).sum(dim=-1)

        emotion_label = F.one_hot(emotion_label, num_classes=32)

        self.record_local_metric('emotion_top1', AverageMetric.many(emotion_top1))
        self.record_local_metric('emotion_top2', AverageMetric.many(emotion_top2))
        self.record_local_metric('emotion_top3', AverageMetric.many(emotion_top3))

        """
            Seq2seq training
        """
        scores, preds, copy_rates = model.decode_forced(emotion_pred, \
            recog_intent, self.intent_keywords, encoder_states, batch.label_vec)

        model_output = scores, preds, encoder_states

        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)

        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        self.record_local_metric('nll', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )

        total_loss = loss / target_tokens + self.alpha * intent_loss + self.gamma * emotion_loss
        loss = loss.sum() / target_tokens.sum() + self.alpha * intent_loss.mean() + self.gamma * emotion_loss.mean()


        """
            Copy rate loss
        """
        if self.explicit is True:
            copy_loss = self.copy_criterion(copy_rates.view(-1), batch.copy_label.view(-1))
            copy_loss = (copy_loss.view(copy_rates.shape) * notnull).sum(dim=-1)

            copy_preds = (copy_rates > 0.5)
            copy_correct = ((batch.copy_label == copy_preds) * notnull).sum(dim=-1)

            self.record_local_metric('copy_loss', AverageMetric.many(copy_loss, target_tokens))
            self.record_local_metric('copy_acc', AverageMetric.many(copy_correct, target_tokens))

            total_loss = total_loss + self.tau * copy_loss / target_tokens
            loss = loss + self.tau * copy_loss.sum() / target_tokens.sum()

        self.record_local_metric('loss', AverageMetric.many(total_loss))

        if return_output:
            return (loss, model_output)
        return loss

    def greedy_generate(self, batch):
        """
        Greedy decoding
        """

        encoder_states = self.model.encoder(*self._encoder_input(batch))

        """
            Emotion Prediction
        """
        state = encoder_states[1]
        
        emotion_logit = self.model.emotion_prediction(state[-1])
        emotion_pred = F.softmax(emotion_logit, dim=-1)

        """
            Intent prediction
        """
        recog_intent, recog_intent_prob = intent_prediction \
            (self.recog_bert, self.intent_tokenizer, batch.labels)

        prior_intent_logit = self.model.intent_prediction(state[-1])

        prior_intent = gumbel_softmax_sample(prior_intent_logit).argmax(dim=-1)

        # preds = self.model.decode_greedy \
        #   (emotion_pred, recog_intent, self.intent_keywords, encoder_states, 32)
        preds = self.model.decode_greedy \
            (emotion_pred, prior_intent, self.intent_keywords, encoder_states, 32)

        return preds


    def eval_step(self, batch):

        if batch.text_vec is None and batch.image is None:
            return

        self.model.eval()
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

            preds = self.greedy_generate(batch)
            self._add_generation_metrics(batch, preds)

        text = [self._v2t(p) for p in preds] if preds is not None else None

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
            self._compute_nltk_bleu(batch, text)
        retval = Output(text, token_losses=token_losses)

        return retval

    def _add_intent_metric(self, batch, preds):

        """
            Metric for intent accuracy of generated responses
        """

        preds = [self._v2t(p) for p in preds]

        recog_intents, _ = intent_prediction(self.recog_bert, self.intent_tokenizer, batch.labels)

        pred_intents, _ = intent_prediction(self.recog_bert, self.intent_tokenizer, preds)

        top1 = (recog_intents == pred_intents)

        self.record_local_metric('generated_acc', AverageMetric.many(top1))

    def _add_generation_metrics(self, batch, preds):

        """
            Add generation metrics including: BLEU, Intent ACC
        """
        if batch.text_vec is None or preds is None:
            return

        self._add_intent_metric(batch, preds)
        self._add_bleu_metrics_for_greedy(batch, preds)
        self._add_bleu_metrics_for_samples(batch)

        return
    
    def _add_bleu_metrics_for_greedy(self, batch, preds):

        """
            Calculate BLEU for one response 
        """
        
        batchsize = batch.batchsize
        bleu = {key: list() for key in range(1, 5)}

        for i in range(batchsize):
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

    def _add_bleu_metrics_for_samples(self, batch):
        
        """
            Calculate BLEU Recall and BLEU Precision for sample responses
        """

        batchsize = batch.batchsize
        sample_num = 5      
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        context, state, mask = encoder_states

        context = context.unsqueeze(dim=1).expand(-1, sample_num, -1, -1).contiguous().\
        view(batchsize*sample_num, context.shape[-2], context.shape[-1])        
        state = state.unsqueeze(dim=2).expand(-1, -1, sample_num, -1).contiguous().\
        view(state.shape[0], batchsize*sample_num, -1)
        mask = mask.unsqueeze(dim=1).expand(-1, sample_num, -1).contiguous().\
        view(batchsize*sample_num, -1)

        encoder_states = [context, state, mask]
        state = encoder_states[1]
        
        """
            Emotion Prediction
        """
        emotion_logit = self.model.emotion_prediction(state[-1])
        emotion_pred = F.softmax(emotion_logit, dim=-1)

        """
            Intent prediction
        """
        prior_intent_logit = self.model.intent_prediction(state[-1])

        prior_intent = gumbel_softmax_sample(prior_intent_logit).argmax(dim=-1)

        # preds = self.model.decode_greedy \
        #   (emotion_pred, recog_intent, self.intent_keywords, encoder_states)
        preds = self.model.decode_greedy \
            (emotion_pred, prior_intent, self.intent_keywords, encoder_states)

        sample_preds = preds.view(batchsize, sample_num, -1)

        precision_bleu = {key: list() for key in range(1, 5)}
        recall_bleu = {key: list() for key in range(1, 5)}

        for i in range(batchsize):
            
            reference = [batch.labels[i].lower()]
            preds = sample_preds[i]
            
            texts = [self._v2t(pred) for pred in preds]

            for k in range(1, 5):
                weights = [1 / k for _ in range(k)]
                max_bleu = 0
                mean_bleu = []
                for text in texts:
                    
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
        shared['recog_bert'] = self.recog_bert

        shared['intent_keywords'] = self.intent_keywords
        shared['intent_tokenizer'] = self.intent_tokenizer
        shared['intent_criterion'] = self.intent_criterion
        shared['emotion_criterion'] = self.emotion_criterion
        shared['copy_criterion'] = self.copy_criterion

        return shared

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        return None