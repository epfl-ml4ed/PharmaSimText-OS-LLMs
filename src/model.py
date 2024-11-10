import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

device = torch.device("cpu")


def softmax(q_values, temperature):
    """
    Apply softmax with temperature to a set of Q-values.

    :param q_values: Tensor of Q-values for each action.
    :param temperature: Temperature for softmax; higher values increase exploration.
    :return: Probabilities for each action.
    """
    q_values_temp = q_values / temperature
    exp_q_values = torch.exp(q_values_temp - torch.max(q_values_temp))
    return exp_q_values / torch.sum(exp_q_values)


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, sequence_length, hidden_size)
        attention_scores = F.softmax(self.attention_weights(hidden_states), dim=1)
        context_vector = torch.cumsum(attention_scores * hidden_states, dim=1)
        return context_vector, attention_scores


def MLP(input_dim, hidden_dim, activation, n_layers):
    layers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    return layers


class DRRN(nn.Module):
    def __init__(
            self,
            mem_type="sum",
            embedding_dim=300,
            n_layers_action=1,
            n_layers_state=1,
            n_layers_scorer=1,
            n_layers_lstm=1,
            hidden_dim_action=128,
            hidden_dim_state=128,
            hidden_dim_scorer=128,
            hidden_lstm=128,
            activation="relu",
            state_dim=300,
            llm="fasttext",
            use_attention=False,
    ):
        super(DRRN, self).__init__()
        self.activation = nn.ReLU if activation == "relu" else None
        self.mem_type = mem_type
        self.use_attention = use_attention
        self.llm = llm

        # Action and state network
        self.action = nn.ModuleList(MLP(embedding_dim, hidden_dim_action, self.activation, n_layers_action))
        self.input_state = state_dim
        self.state = nn.ModuleList(
            MLP(hidden_lstm if mem_type == "lstm" else self.input_state, hidden_dim_state, self.activation,
                n_layers_state))

        # Scorer network
        self.scorer = nn.ModuleList(
            MLP(hidden_dim_action + hidden_dim_state, hidden_dim_scorer, self.activation, n_layers_scorer) + [
                nn.Linear(hidden_dim_scorer, 1)])

        # Optional LSTM and Attention layers
        if mem_type == "lstm":
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_lstm, num_layers=n_layers_lstm,
                                batch_first=True)
        if use_attention:
            self.attention = AttentionLayer(hidden_lstm if mem_type == "lstm" else self.input_state)

    def forward(self, state_batch, act_batch):
        if self.mem_type in ["sum", "None"]:
            state_batch = torch.from_numpy(np.concatenate(state_batch, axis=0)).float().to(device)
            new_state = None
        elif self.mem_type == "lstm":
            state_batch, new_state = self._process_lstm_state(state_batch)
        else:
            raise NotImplementedError(f"Memory type {self.mem_type} not supported.")

        act_batch = torch.from_numpy(np.concatenate(list(itertools.chain.from_iterable(act_batch)), axis=0)).float().to(
            device)
        for layer in self.action:
            act_batch = layer(act_batch)
        for layer in self.state:
            state_batch = layer(state_batch)

        state_batch = torch.cat([state_batch[i].repeat(j, 1) for i, j in enumerate([len(a) for a in act_batch])], dim=0)
        z = torch.cat((state_batch, act_batch), dim=1)
        for layer in self.scorer:
            z = layer(z)
        act_values = z.squeeze(-1)
        return act_values.split([len(a) for a in act_batch]), new_state

    def _process_lstm_state(self, state_batch):
        if isinstance(state_batch, tuple):
            state_batch[0] = torch.from_numpy(state_batch[0]).float().to(device)
            output, (hn, cn) = self.lstm(state_batch[0], state_batch[1][0] if self.use_attention else state_batch[1])
            output = output.view(-1, output.size(-1))
            if self.use_attention:
                context_vector, _ = self.attention(output)
                output = context_vector + state_batch[1][1]
            return output, [hn, cn]
        elif isinstance(state_batch[0], np.ndarray):
            return self._process_padded_sequences(state_batch)
        else:
            raise TypeError("Unsupported type for state_batch in LSTM processing.")

    def _process_padded_sequences(self, state_batch):
        state_batch = [torch.from_numpy(x).float().to(device) for x in state_batch]
        padded_sequences = rnn_utils.pad_sequence(state_batch, batch_first=True, padding_value=0.0)
        seq_lengths = torch.LongTensor([len(seq) for seq in state_batch]).to(device)
        packed_sequences = rnn_utils.pack_padded_sequence(padded_sequences, seq_lengths, batch_first=True,
                                                          enforce_sorted=False)
        packed_hidden_states, (h, c) = self.lstm(packed_sequences)
        unpacked_hidden_states, _ = rnn_utils.pad_packed_sequence(packed_hidden_states, batch_first=True)
        state_batch_encoded = [unpacked_hidden_states[i][:seq_lengths[i]] for i in range(len(seq_lengths))]
        output = torch.cat(state_batch_encoded, dim=0)
        return output, [h[:, -1, :], c[:, -1, :]]

    @torch.no_grad()
    def act(self, states, act_ids, policy="softmax", epsilon=1, temperature=1):
        act_values, new_state = self.forward(states, act_ids)
        if policy == "softmax":
            act_probs = softmax(act_values[0], temperature)
            act_idxs = [torch.multinomial(act_probs, num_samples=1).item()]
        elif policy == "epsilon_greedy":
            act_idxs = [
                torch.randint(0, len(vals), (1,)).item() if random.random() < epsilon else vals.argmax(dim=0).item()
                for vals in act_values
            ]
        else:
            act_idxs = [vals.argmax(dim=0).item() for vals in act_values]
        return act_idxs, act_values, new_state, act_probs


def main():
    model = DRRN(embedding_dim=300, n_layers_action=1, n_layers_state=1, n_layers_scorer=1, hidden_dim_action=32,
                 hidden_dim_state=64, hidden_dim_scorer=128, activation="relu").to(device)
    s = [np.random.rand(1, 300)]
    a = [[np.random.rand(1, 300), np.random.rand(1, 300)]]
    print(model(s, a))


if __name__ == "__main__":
    main()
