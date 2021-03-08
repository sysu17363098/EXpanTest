'''
https://github.com/plmsmile/NLP-Demos/blob/master/en-zh-translation/model.py
【翻译模型】
@author PLM
@date 2017-10-16
'''
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def my_log_softmax(x):
  '''只能处理3维的'''
  size = x.size()
  res = F.log_softmax(x.squeeze())
  res = res.view(size[0], size[1], -1)
  return res


class EncoderRNN(nn.Module):
  ''' 对句子进行编码 input-embeded-gru-output
  [s, batch_size] -- [s, b, h]，即[句子长度，句子个数] -- [句子长度，句子个数，编码维数]
  '''

  def __init__(self,
               input_size,
               hidden_size,
               n_layers=1,
               dropout_p=0.1,
               bidir=False,
               rnn_cell='GRU'):
    super(EncoderRNN, self).__init__()
    #self.vocab_size = vocab_size
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.bidir = bidir
    #self.embedding = nn.Embedding(vocab_size, hidden_size)
    if rnn_cell == 'GRU':
      self.rnn_cell = nn.GRU(input_size, hidden_size, n_layers,
                             dropout=dropout_p, bidirectional=bidir)
    elif rnn_cell == 'LSTM':
      self.rnn_cell = nn.LSTM(input_size, hidden_size, n_layers,
                              dropout=dropout_p, bidirectional=bidir)
    else:
      raise ValueError(f'Unknown rnn_cell type: {rnn_cell}')

  def forward(self, input_seqs, input_lengths, hidden=None):
    ''' 对输入的多个句子经过GRU计算出语义信息
    1. input_seqs > embeded
    2. embeded - packed > GRU > outputs - pad -output
    Args:
        input_seqs: [s, b]
        input_lengths: list[int]，每个batch句子的真实长度
    Returns:
        outputs: [s, b, h]
        hidden: [n_layer, b, h]
    '''
    # 一次运行，多个batch，多个序列
    # embedded = self.embedding(input_seqs)
    embedded = input_seqs

    outputs, hidden = self.rnn_cell(embedded, hidden)

    # 双向，两个outputs求和
    if self.bidir is True:
      outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
      h = torch.sum(hidden[0], dim=0, keepdim=True)
      o = torch.sum(hidden[1], dim=0, keepdim=True)
      # print(f"h.shape={h.shape}, o.shape={o.shape}")
      hidden = (h, o)

    return outputs, hidden


class Attn(nn.Module):
  '''计算对齐向量，只有general可以使用'''

  def __init__(self, score_type, hidden_size):
    '''
    Args:
        score_type: 计算score的方法，'dot', 'general', 'concat'
        hidden_size: Encoder和Decoder的hidden_size
    '''
    super(Attn, self).__init__()
    self.score_type = score_type
    self.hidden_size = hidden_size
    if score_type == 'general':
      self.attn = nn.Linear(hidden_size, hidden_size)
    elif score_type == 'concat':
      self.attn = nn.Linear(hidden_size * 2, hidden_size)
      self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

  def score(self, decoder_rnn_output, encoder_output):
    ''' 计算Decoder中yt与Encoder中hs的打分。算出所有得分，再softmax就可以算出对齐向量。
    下面均是单个batch
    Args:
        decoder_rnn_output: [1, h]，Decoder中顶层RNN的输出[1,h] < [1,b,h]
        encoder_output: [1, h]，Encoder最后的输出[1,h] < [s,b,h]>
    Returns:
        energy: 即Yt与Xs的得分
    '''
    # dot 需要两个1维的向量
    if self.score_type == 'dot':
      energy = decoder_rnn_output.squeeze(0).dot(encoder_output.squeeze(0))
    elif self.score_type == 'general':
      energy = self.attn(encoder_output)
      energy = decoder_rnn_output.squeeze(0).dot(energy.squeeze(0))
    elif self.score_type == 'concat':
      h_o = torch.cat((decoder_rnn_output, encoder_output), 1)
      energy = self.attn(h_o)
      energy = self.v.squeeze(0).dot(energy.squeeze(0))
    return energy

  def forward(self, rnn_outputs, encoder_outputs):
    '''ts个时刻，计算ts个与is的对齐向量，也是注意力权值
    Args:
        rnn_outputs: Decoder中GRU的输出[ts, b, h]
        encoder_outputs: Encoder的最后的输出, [is, b, h]
    Returns:
        attn_weights: Yt与所有Xs的注意力权值，[b, ts, is]
    '''
    target_seqlen = rnn_outputs.size()[0]
    input_seqlen = encoder_outputs.size()[0]
    batch_size = encoder_outputs.size()[1]

    rnn_outputs = rnn_outputs.transpose(0, 1)         # [b, ts, h]

    if self.score_type == 'general':
      # (b, h, is)
      encoder_outputs = encoder_outputs.transpose(0, 1)
      encoder_outputs = self.attn(encoder_outputs).transpose(1, 2)
      # [b,ts,is] <[b,ts,h]*[b,h,is]
      attn_energies = rnn_outputs.bmm(encoder_outputs)
      res = my_log_softmax(attn_energies)
      return res

    elif self.score_type == 'dot':
      encoder_outputs = encoder_outputs.transpose(0, 1) # [b, is, h]

      attn_energies = torch.bmm(rnn_outputs, encoder_outputs.transpose(1, 2))
    else:
      raise ValueError(f"Unknown score_type: {self.score_type}")

    return torch.softmax(attn_energies, dim=-1)


class AttnDecoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, score_method='dot', rnn_cell='GRU', n_layers=1, dropout_p=0.1):
    super(AttnDecoderRNN, self).__init__()
    self.score_method = score_method
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p

    # self.embedding = nn.Embedding(output_size, hidden_size)
    # self.embedding_dropout = nn.Dropout(dropout_p)
    if rnn_cell == 'GRU':
      self.rnn_cell = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout_p)
    elif rnn_cell == 'LSTM':
      self.rnn_cell = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p)
    else:
      raise ValueError(f'Unknown rnn_cell type: {rnn_cell}')
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

    # 选择attention
    if score_method != 'none':
      self.attn = Attn(score_method, hidden_size)

  def forward(self, input_seqs, last_hidden, encoder_outputs):
    '''
    一次输入(ts, b)，b个句子, ts=target_seq_len
    1. input > embedded
    2. embedded, last_hidden --GRU-- rnn_output, hidden
    3. rnn_output, encoder_outpus --Attn-- attn_weights
    4. attn_weights, encoder_outputs --相乘-- context
    5. rnn_output, context --变换,tanh,变换-- output
    Args:
        input_seqs: [ts, b, hidden_size]
        last_hidden: [n_layers, b, h]
        encoder_outputs: [is, b, h]
    Returns:
        output: 最终的输出，[ts, b, o]
        hidden: GRU的隐状态，[nl, b, h]
        attn_weights: 对齐向量，[b, ts, is]
    '''
    # batch_size = input_seqs.size()[1]
    # ts = input_seqs.size()[0]
    # ins = encoder_outputs.size()[0]
    ts, batch_size, hidden_size = input_seqs.shape
    embedded_start = time.time()

    # embedded = self.embedding(input_seqs)
    # embedded = embedded.view(ts, batch_size, self.hidden_size)
    embedded = input_seqs

    # (ts, b, h), (nl, b, h)
    rnn_output, hidden = self.rnn_cell(embedded, last_hidden)
    # [ts, b, is]

    attn_start = time.time()
    # 对齐向量 [b,ts,is]
    attn_weights = self.attn(rnn_output, encoder_outputs)

    attn_end = time.time()
    # 新的语义 [b,ts,h] < [b,ts,is] * [b,is,h].
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
    # [ts,b,h] <
    context = context.transpose(0, 1)
    # show_tensor("context", context)

    # 语义和输出 [ts, b, 2h] < [ts, b, h], [ts, b, h]
    output_context = torch.cat((rnn_output, context), 2)
    # [ts, b, h]
    output_context = self.concat(output_context)
    concat_output = F.tanh(output_context)

    # [ts, b, o]
    output = self.out(concat_output)
    output_end = time.time()

    rnn_use = attn_start - embedded_start
    attn_use = attn_end - attn_start
    remain_use = output_end - attn_end
    # print ('%.3f, %.3f, %.3f' % (rnn_use, attn_use, remain_use))
    # show_tensor("output", output)
    return output, hidden, attn_weights

  def init_outputs(self, seq_len, batch_size):
    outputs = torch.zeros(seq_len, batch_size, self.output_size)
    return outputs

  def create_input_seqs(self, seq_len, batch_size):
    sos = [helper.SOS_token] * batch_size
    sos = [sos] * seq_len
    return torch.LongTensor(sos)