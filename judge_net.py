import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from seq2seq import EncoderRNN, AttnDecoderRNN  # TODO
import util as util  # TODO
from envs import BaseHiEnv  # TODO
from typing import *

logger = logging.getLogger()


class InnerProductJudgeNet(nn.Module):

    def __init__(self, word_vec_dim):
        super().__init__()
        self.trainable = True
        self.word_vec_dim = word_vec_dim
        self.project1 = nn.Linear(self.word_vec_dim, self.word_vec_dim)

    def forward(self, Ks: torch.Tensor, Cs: torch.Tensor, *args):
        """
        :param Ks, keywords used to expand: (batch_size, n_keys, word_vector_dim)
        :param Cs, candidates searched by Ks: (batch_size, n_candidates, word_vector_dim)
        :return: probs as good / bad candiates: (batch_size, n_candidates, 2)
        """
        center = torch.mean(Ks, dim=-2, keepdim=True).transpose(1, 2)           # (batch_size, word_vector_dim, 1)

        Cs_projection = torch.relu(self.project1(Cs))
        products = torch.bmm(Cs_projection, center)                             # (batch_size, n_candidates, 1)
        rest = -1 * products
        result = torch.cat([products, rest], dim=-1)                             # (batch_size, n_candidates, 1)

        return result


class RNNJudgeNet(nn.Module):
    """
    keys: (n_keys, word_vec_dim)
    candidates: (n_candidates, word_vec_dim)
    query = [keys; 0; candidates]: (n_keys + 1 + n_candidates, word_vec_dim),
    where 0 is used to separate keys and candidates
    result = GRU-Encoder-Decoder-with-Attention(query): (n_candidates, 2),
    which indicates the possibility of ith candidates to be good
    """

    def __init__(self,
                 word_vec_dim,
                 hidden_state_size,
                 bidir=True,
                 rnn_cell='LSTM',
                 ):
        super().__init__()
        self.trainable = True
        self.word_vec_dim = word_vec_dim
        self.hidden_state_size = hidden_state_size
        self.encoder = EncoderRNN(self.word_vec_dim, self.hidden_state_size, bidir=bidir, rnn_cell=rnn_cell)
        self.decoder = AttnDecoderRNN(self.word_vec_dim, self.hidden_state_size, 2, rnn_cell=rnn_cell)
        self.encoder.apply(util.weight_init)
        self.decoder.apply(util.weight_init)

    def forward(self, Ks: torch.Tensor, Cs: torch.Tensor, *args):
        """
        :param Ks, keywords used to expand: (batch_size, n_keys, word_vector_dim)
        :param Cs, candidates searched by Ks: (batch_size, n_candidates, word_vector_dim)
        :return: probs as good / bad candiates: (batch_size, n_candidates, 2)
        """
        batch_size = Ks.shape[0]
        n_candidates = Cs.shape[1]

        sep = torch.zeros(batch_size, 1, self.word_vec_dim)
        query_string = torch.cat([Ks, sep, Cs], dim=1)            # (batch_size, n_keys + 1 + n_candidates, word_vector_dim)
        query_string_transposed = query_string.transpose(0, 1)    # (n_keys + 1 + n_candidates, batch_size, word_vector_dim)
        lengths = [query_string_transposed.shape[0]]              # (n_keys + 1 + n_candidates)

        encoder_outputs, encoder_hidden = self.encoder(query_string_transposed,
                                                       torch.tensor(lengths).long().cpu())
                                                                  # (n_keys + 1 + n_candidates, batch_size, hidden_state_size)
                                                                  # (n_layers=1, batch_size, hidden_state_size)

        decoder_hidden = encoder_hidden

        answers = []
        for i in range(n_candidates):
            # logger.debug(f"decoder_hidden: {decoder_hidden[:, :, 0:10]}")
            decoder_input = Cs[:, i].unsqueeze(0)      # TODO (new dim=1,a candidate=1, word_vector_dim)
            # (1, batch_size, hidden_state_size) 此处batch指的不是前面的那个了
            output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                                                                  # (1, batch_size, 2)
                                                                  # (n_layers=1, batch_size, hidden_state_size)
            answers.append(output)

        probs = torch.cat(answers, dim=0)                         # (n_candidates, batch_size, 2)
        probs = probs.transpose(0, 1)                             # (batch_size, n_candidates, 2)
        # probs = torch.softmax(probs, dim=-1)

        return probs


class RNNDotJudgeNet(nn.Module):

    def __init__(self,
                 word_vec_dim,
                 bidir=True,
                 rnn_cell='LSTM'):
        super().__init__()
        self.trainable = True
        self.word_vec_dim = word_vec_dim
        self.hidden_state_size = word_vec_dim
        self.encoder = EncoderRNN(self.word_vec_dim, self.word_vec_dim, bidir=bidir, rnn_cell=rnn_cell)
        self.encoder.apply(util.weight_init)

    def forward(self, Ks: torch.Tensor, Cs: torch.Tensor, *args):
        """
        :param Ks, keywords used to expand: (batch_size, n_keys, word_vector_dim)
        :param Cs, candidates searched by Ks: (batch_size, n_candidates, word_vector_dim)
        :return: probs as good / bad candiates: (batch_size, n_candidates, 2)
        """
        batch_size = Ks.shape[0]
        n_candidates = Cs.shape[1]

        sep = torch.zeros(batch_size, 1, self.word_vec_dim)
        query_string = torch.cat([Ks, sep, Cs], dim=1)            # (batch_size, n_keys + 1 + n_candidates, word_vector_dim)
        query_string_transposed = query_string.transpose(0, 1)    # (n_keys + 1 + n_candidates, batch_size, word_vector_dim)
        lengths = [query_string_transposed.shape[0]]

        encoder_outputs, encoder_states = self.encoder(query_string_transposed,
                                                       torch.tensor(lengths).long().cpu())
                                                                  # (n_keys + 1 + n_candidates, batch_size, hidden_state_size)
                                                                  # (n_layers=1, batch_size, hidden_state_size)

        encoder_hidden = torch.sum(encoder_states[0], dim=0).view(batch_size, self.hidden_state_size, 1)
        products = torch.bmm(Cs, encoder_hidden)                  # (batch_size, n_candidates, 1)

        rest = -1 * products
        result = torch.cat([products, rest], dim=-1)

        return result


class FieldJudgeNet(nn.Module):

    def __init__(self,
                 word_vec_dim,
                 fields: List[Tuple[str, ...]]):
        super().__init__()
        self.trainable = True
        self.word_vec_dim = word_vec_dim
        self.fields = fields
        self.field_cnt = len(self.fields)
        self.linear = nn.Parameter(torch.zeros(self.field_cnt, self.word_vec_dim, 2))
        nn.init.xavier_normal_(self.linear)

    def forward(self, Ks: torch.Tensor, Cs: torch.Tensor, env: BaseHiEnv):
        """
        :param Ks, keywords used to expand: (batch_size, n_keys, word_vector_dim)
        :param Cs, candidates searched by Ks: (batch_size, n_candidates, word_vector_dim)
        :return: probs as good / bad candidates: (batch_size, n_candidates, 2)
        """
        _, _, field = env.state()
        field_index = self.fields.index(field)
        W = self.linear[field_index]                              # (word_vec_dim, 2)

        result = torch.tensordot(Cs, W, dims=([-1], [-2]))        # (batch_size, n_candidates, 2)

        return result


class Pul2JudgeNet(nn.Module):

    def __init__(self,
                 word_vec_dim):
        super().__init__()
        self.trainable = True
        self.word_vec_dim = word_vec_dim
        self.linear = nn.Parameter(torch.zeros(self.word_vec_dim, 2))
        nn.init.xavier_normal_(self.linear)

    def forward(self, Ks: torch.Tensor, Cs: torch.Tensor, *args):
        """
        :param Ks, keywords used to expand: (batch_size, n_keys, word_vector_dim)
        :param Cs, candidates searched by Ks: (batch_size, n_candidates, word_vector_dim)
        :return: probs as good / bad candidates: (batch_size, n_candidates, 2)
        """
        center = Ks.mean(dim=-2, keepdim=True)                          # (batch_size, 1, word_vector_dim)
        diff = Cs - center                                              # (batch_size, n_candidates, word_vector_dim)
        result = torch.tensordot(diff, self.linear, dims=([-1], [-2]))  # (batch_size, n_candidates, 2)
        return result


if __name__ == '__main__':

    net1 = InnerProductJudgeNet(5)
    Ks = torch.zeros(10, 10, 5)
    print(Ks.shape)
    Cs = torch.zeros(10, 40, 5)
    print(Cs.shape)
    probs: torch.Tensor = net1(Ks, Cs)
    print(probs.shape)

    net2 = RNNJudgeNet(5, 5)
    prob: torch.Tensor = net2(Ks, Cs)
    print(prob)
    print(prob.shape)
