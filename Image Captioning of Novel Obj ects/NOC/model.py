import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class EncoderCNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size, momentum=0.01)


    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)

        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, weights_matrix, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        # look up table from indices to vectors
        self.embed, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, False)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embed_size)
        self.max_seg_length = max_seq_length

    def forward(self, captions, lengths):
        """Decode image feature vectors and generates captions."""

        embeddings = self.embed(captions)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)

        x = self.linear(hiddens[0])

        # multiplying by the transpose of the embedding
        x = x.matmul(self.embed.weight.t())
        return x


    def sample(self, features, start_id, states=None):
        """Generate captions for given image features using greedy search."""
        # change this to beam search
        # features is the vocab wrapper.
        sampled_ids = []

        inputs = torch.tensor([start_id], dtype=torch.long)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        inputs = self.embed(inputs)

        inputs = inputs.unsqueeze(1)
#
        for i in range(self.max_seg_length):

            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            outputs = outputs.matmul(self.embed.weight.t())
            # add the features vector to the outputs
            m = nn.Softmax()
            final = torch.zeros(9956)

            for output in outputs:
                print(output[1575])
                final = output + features[0]

            values, index = torch.max(final, 0)

            sampled_ids.append(index.data.item())
            inputs = self.embed(index)
            inputs = inputs.view(1, 1, -1)
        return sampled_ids
