import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def bipartite_match(Xs: torch.Tensor, Ys: torch.Tensor):
    """
    :param Xs: (batch_size, set_size, element_dim)
    :param Ys: (output_dim, hidden_set_size, element_dim)
    :return: Each set X of Xs's approximate optimal value with each hidden set Y of Ys
    """
    inner_products = torch.tensordot(Xs, Ys, dims=([-1], [-1])) # (batch_size, set_size, output_dim, hidden_set_size)
    G = F.relu(inner_products)                                  # (batch_size, set_size, output_dim, hidden_set_size)

    if Xs.shape[1] <= Ys.shape[1]:
        match = torch.max(G, dim=1)                               # (batch_size, output_dim, hidden_set_size)
        value = torch.sum(match.values, dim=2)                    # (batch_size, output_dim)
    else:
        match = torch.max(G, dim=-1)                              # (batch_size, set_size, output_dim)
        value = torch.sum(match.values, dim=1)                    # (batch_size, output_dim)

    return value


def select_X_by_keys(Xcompressed: torch.Tensor, keys: torch.Tensor):
    """
    :param Xcompressed: (batch_size, set_size, compressed_dim)
    :param keys: (batch_size, K, self.compressed_dim)
    :return: prob: (batch_size, set_size, K)
    """
    products = torch.bmm(Xcompressed, keys.transpose(1, 2))
    prob = F.softmax(products, dim=1)
    return prob


class RepSet(nn.Module):

    def __init__(self,
                 element_dim,
                 output_dim,
                 hidden_set_size
               ):
        """
        hidden_set_weights: (output_dim, hidden_set_size, element_dim)
        """

        super().__init__()

        self.element_dim = element_dim
        self.output_dim = output_dim
        self.hidden_set_size = hidden_set_size

        self.hidden_sets = nn.Parameter(torch.zeros(self.output_dim, self.hidden_set_size, self.element_dim))
        nn.init.xavier_normal_(self.hidden_sets)

    def forward(self, Xs):
        return bipartite_match(Xs, self.hidden_sets)


class RepSetRNN(nn.Module):  # TODO 没写完，且没用

    def __init__(self,
                 word_vector_dim,
                 compressed_dim,
                 hidden_set_size,
                 set_representation_dim,
                 lstm_hidden_state_size
               ):
        super().__init__()

        self.word_vector_dim = word_vector_dim
        self.compressed_dim = compressed_dim
        self.hidden_set_size = hidden_set_size
        self.set_representation_dim = set_representation_dim
        self.lstm_hidden_state_size = lstm_hidden_state_size

        self.compress_layer = nn.Linear(self.word_vector_dim, self.hidden_set_size)
        self.lstm = nn.LSTM(self.compressed_dim, self.lstm_hidden_state_size)
        self.rep_set = RepSet(self.compressed_dim, self.set_representation_dim, self.hidden_set_size)

    def forward(self, Xs):
        """
        :param Xs: (batch_size, set_size, word_vector_dim)
        :return:
        """
        Xs_compressed = self.compress_layer(Xs)               # (batch_size, set_size, compressed_dim)
        Xs_set_rep = self.rep_set(Xs_compressed)              # (batch_size, set_representation_dim)


class RepSetKGroups(nn.Module):  # TODO 用的应该是这个

    def __init__(self,
                 word_vector_dim,
                 compressed_dim,
                 hidden_set_size,
                 K):

        super().__init__()

        self.word_vector_dim = word_vector_dim
        self.compressed_dim = compressed_dim
        self.hidden_set_size = hidden_set_size
        self.K = K

        self.compress_layer = nn.Linear(self.word_vector_dim, self.compressed_dim)
        # self.lstm = nn.LSTM(self.compressed_dim, self.lstm_hidden_state_size)
        self.rep_set = RepSet(self.compressed_dim, self.K * self.compressed_dim, self.hidden_set_size)

    def forward(self, Xs):
        """
        :param Xs: (batch_size, set_size, word_vector_dim)
        :return: prob: (batch_size, set_size, K)
        """
        Xs_compressed = self.compress_layer(Xs)                     # (batch_size, set_size, compressed_dim)
        Xs_set_rep: torch.Tensor = self.rep_set(Xs_compressed)      # (batch_size, set_representation_dim)
        K_keys = Xs_set_rep.view(-1, self.K, self.compressed_dim)   # (batch_size, K, compressed_dim)
        prob = select_X_by_keys(Xs_compressed, K_keys)              # (batch_size, set_size, K)
        return prob

    def check(self):
        # print("weights[0][0]:", self.compress_layer.weight[0][0].item())
        return self.compress_layer.weight[0][0].item() != float("nan")


if __name__ == '__main__':
    Xs = torch.arange(2 * 3 * 4).view(2, 3, 4)  # (2,3,4)
    Ys = torch.arange(3 * 3 * 4).view(3, 3, 4)  # (3,3,4)

    print(Xs.shape)
    print(Ys.shape)

    value = bipartite_match(Xs, Ys)
    print(value.shape)

    vectors = torch.rand(20, 768).unsqueeze(0)
    print("Testing SetNet:")
    net = RepSetKGroups(768, 126, 768, 3)
    prob = net(vectors)
    print(prob.size())
