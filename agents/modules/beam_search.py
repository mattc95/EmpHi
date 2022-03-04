import torch
import torch.nn as nn
import torch.nn.functional as F


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty

        if len(self) < self.num_beams or score > self.worst_score:
            # 可更新的情况：数量未饱和或超过最差得分
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 数量饱和需要删掉一个最差的
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)],
                    key=lambda x: x[0])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        相关样本是否已经完成生成。
        best_sum_logprobs是新的候选序列中的最高得分。
        """

        if len(self) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret


def beam_search(
        encoder_states,
        model,
        start_idx,
        end_idx,
        pad_idx,
        mmi=False,
        target=None,
        num_beams=10,
        max_length=64,
        length_penalty=1,
):
    context, state, mask = encoder_states
    bsz = context.size(0)
    embedding_size = context.size(-1)
    vocab_size = model.vocab_size
    device = context.device

    # 建立beam容器，每个样本一个
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty)
        for _ in range(bsz)
    ]
    done = [False for _ in range(bsz)]

    # 第一步自动填入bos_token
    input_ids = torch.full(
        (bsz * num_beams, ),
        start_idx,
        dtype=torch.long,
        device=device,
    )

    # 每个beam容器的得分，共batch_size*num_beams个
    beam_scores = torch.zeros((bsz, num_beams), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    beam_seq = input_ids.unsqueeze(dim=-1)
    beam_context = context.unsqueeze(dim=1).expand(-1, num_beams, -1, -1).contiguous().view(
        bsz*num_beams, -1, embedding_size
    )
    beam_state = state.unsqueeze(dim=2).expand(-1, -1, num_beams, -1).contiguous().view(
        -1, bsz*num_beams, embedding_size
    )
    beam_mask = mask.unsqueeze(dim=1).expand(-1, num_beams, -1).contiguous().view(
        bsz*num_beams, -1
    )

    # 当前长度设为1
    cur_len = 1

    while cur_len < max_length:
        # 将编码器得到的上下文向量和当前结果输入解码器，即图中1

        out, state = model.decoder(input_ids, None, None, beam_context, beam_state, beam_mask)
        # 输出矩阵维度为：(batch*num_beams)*cur_len*vocab_size

        # 取出最后一个时间步的各token概率，即当前条件概率
        # (batch*num_beams)*vocab_size
        logits, _ = model.output(out[:, -1, :], out[:, -1, :], None, None)
        scores = torch.log(F.softmax(logits, dim=-1))
        ###########################
        # 这里可以做一大堆操作减少重复 #
        ###########################
        # 计算序列条件概率的，因为取了log，所以直接相加即可。得到图中2矩阵
        # (batch_size * num_beams, vocab_size)

        next_scores = scores + beam_scores[:, None].expand_as(scores)
        # 为了提速，将结果重排成图中3的形状
        next_scores = next_scores.view(
            bsz, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)
        # 取出分数最高的token（图中黑点）和其对应得分
        # sorted=True，保证返回序列是有序的

        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        # 下一个时间步整个batch的beam列表
        # 列表中的每一个元素都是三元组
        # (分数, token_id, beam_id)
        next_batch_beam = []
        # 对每一个样本进行扩展
        for batch_idx in range(bsz):
            # 检查样本是否已经生成结束
            if done[batch_idx]:
                # 对于已经结束的句子，待添加的是pad token
                next_batch_beam.extend([(0, pad_idx, 0)] * num_beams)  # pad the batch
                continue
            # 当前样本下一个时间步的beam列表
            next_sent_beam = []
            # 对于还未结束的样本需要找到分数最高的num_beams个扩展
            # 注意，next_scores和next_tokens是对应的
            # 而且已经按照next_scores排好顺序
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and word IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size
                effective_beam_id = batch_idx * num_beams + beam_id
                # 如果出现了EOS token说明已经生成了完整句子
                if token_id.item() == end_idx:
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    # 往容器中添加这个序列
                    # print(beam_seq[effective_beam_id], beam_token_score.item())

                    generated_hyps[batch_idx].add(
                        beam_seq[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted word if it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                # 扩展num_beams个就够了
                if len(next_sent_beam) == num_beams:
                    break
            # 检查这个样本是否已经生成完了，有两种情况
            # 1. 已经记录过该样本结束
            # 2. 新的结果没有使结果改善
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len=cur_len
            )
            # 把当前样本的结果添加到batch结果的后面
            next_batch_beam.extend(next_sent_beam)
        # 如果全部样本都已经生成结束便可以直接退出了
        if all(done):
            break

        # 把三元组列表再还原成三个独立列表
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        # 准备下一时刻的解码器输入
        # 取出实际被扩展的beam
        input_ids = beam_tokens
        beam_state = state[:, beam_idx]
        # 在这些beam后面接上新生成的token
        beam_seq = beam_seq[beam_idx]
        beam_seq = torch.cat([beam_seq, beam_tokens.unsqueeze(1)], dim=-1)

        # 更新当前长度
        cur_len = cur_len + 1
        # end of length while

    # 将未结束的生成结果结束，并置入容器中
    for batch_idx in range(bsz):
        # 已经结束的样本不需处理
        if done[batch_idx]:
            continue

        # 把结果加入到generated_hyps容器
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            score = beam_scores[effective_beam_id].item()
            seq = beam_seq[effective_beam_id]
            generated_hyps[batch_idx].add(seq, score)

    sorted_res = {key: list() for key in range(bsz)}
    # for batch_idx in range(bsz):
    #     beams = generated_hyps[batch_idx].beams

    #     if mmi is True:
    #         beams = mmi_sort(beams, target[batch_idx], pad_idx)

    #     sorted_scores = sorted(
    #         [(s, idx) for idx, (s, _) in enumerate(beams)],
    #         key=lambda x: x[0])
    #     sorted_res.append(beams[sorted_scores[-1][1]][1][1:])

    for batch_idx in range(bsz):
        beams = generated_hyps[batch_idx].beams
        for score, hyp in beams:
            sorted_res[batch_idx].append(hyp[1: ])
        
    return sorted_res

def mmi_sort(beams, target, pad_idx, alpha=0.1):

    device = 'cuda'
    model = torch.load('./paras/reverse_gru.pkl')
    beam_num = len(beams)

    source = [hyp for (score, hyp) in beams]
    source = torch.nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=pad_idx).to(device)
    source_len = [hyp.shape[0] for (score, hyp) in beams]

    encoder_states = model.encoder(source, source_len)
    target = target.unsqueeze(0).expand(beam_num, -1).to(device)

    logit, _ = model.decode_forced(encoder_states, target)
    notnull = target.ne(pad_idx)
    score = torch.gather(input=F.softmax(logit, dim=-1), index=target.unsqueeze(-1), dim=-1).squeeze(-1)
    logprobs = torch.log(score)
    reverse_logprobs = (logprobs * notnull).sum(dim=1) / notnull.sum(dim=1)
    # print(beams)
    beams = [(score + alpha*reverse_logprobs[i].item(), hyp) for i, (score, hyp) in enumerate(beams)]
    # print(beams)
    return beams






