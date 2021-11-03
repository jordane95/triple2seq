import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    """fact encoder"""
    def __init__(self, input_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.knowledge_embedding = nn.Embedding(input_size, embedding_size)
        self.encoding = nn.Linear(embedding_size, hidden_size)
    
    def forward(self, triple):
        """
        triple: (s, r, o) where shape of s is (N)
        """
        s, r, o = triple
        x_s = self.knowledge_embedding(s) # shape (N, embedding_size)
        x_r = self.knowledge_embedding(r)
        x_o = self.knowledge_embedding(o)

        encoding_s = self.encoding(x_s) # shape (N, hidden_size)
        encoding_r = self.encoding(x_r)
        encoding_o = self.encoding(x_o)

        fact_embedding = torch.cat((encoding_s, encoding_r, encoding_o), dim=1)
        return fact_embedding


class Decoder(nn.Module):
    """question decoder with attention"""
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1, p=0.5):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.dropout = nn.Dropout(p)
        self.word_embedding = nn.Embedding(input_size, embedding_size)

        # fact_embedding + previous hidden state
        self.energy = nn.Linear(hidden_size*4, 3)
        self.sigmoid = nn.Sigmoid()

        # input: context_vector + input word
        self.rnn = nn.GRU(hidden_size + embedding_size, hidden_size, num_layers)

        # input: context_vector + word_vector + hidden_state
        self.fc_vocab = nn.Linear(hidden_size*2 + embedding_size, output_size)

    def forward(self, x, fact_embedding, hidden):
        """
        x: shape (N)
        fact_embedding: shape (N, 3*hidden_size)
        hidden: shape (1, N, hidden_size)
        """
        x = x.unsqueeze(0) # (1, N)
        word_embedding = self.dropout(self.word_embedding(x)) # (1, N, embedding_size)

        fact_embedding = fact_embedding.unsqueeze(0) # (1, N, hidden_size*3)

        energy = self.energy(torch.cat((fact_embedding, hidden), dim=2)) # (1, N, 3)
        attention = self.sigmoid(energy).permute(1, 2, 0) # (N, 3, 1)

        fact_embedding = fact_embedding.squeeze(0) # (N, hidden_size*3)

        fact_s, fact_r, fact_o = torch.split(fact_embedding, self.hidden_size, dim=1) # (N, hidden_size) for each
        fact_embedding = torch.stack((fact_s, fact_r, fact_o), dim=2) # shape (N, hidden_size, 3)

        context_vector = torch.bmm(fact_embedding, attention) # (N, hidden_size, 1)
        context_vector = context_vector.permute(2, 0, 1) # (1, N, hidden_size)

        output, hidden = self.rnn(torch.cat((word_embedding, context_vector), dim=2), hidden)

        prediction = self.fc_vocab(torch.cat((output, word_embedding, context_vector), dim=2)).squeeze(0) # (N, output_size)

        return prediction, hidden

class Triple2Seq(nn.Module):
    """triple to sequence model"""
    def __init__(self, encoder, decoder):
        super(Triple2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        
        # transform fact_embedding to same shape of hidden_state
        self.fc_transform = nn.Linear(self.encoder.hidden_size*3, self.decoder.hidden_size)
    
    def forward(self, triple, question, teacher_force_ratio=0.5):
        """
        triple: (s, r, o)
        question: (seq_length, N)
        """
        batch_size = question.shape[1]
        seq_length = question.shape[0]
        vocab_size = self.decoder.output_size
        
        predictions = torch.zeros(seq_length, batch_size, vocab_size)

        x = question[0]
        fact_embedding = self.encoder(triple)

        hidden = self.fc_transform(fact_embedding).unsqueeze(0) # h_0

        for t in range(1, seq_length):
            prediction, hidden = self.decoder(x, fact_embedding, hidden)
            predictions[t] = prediction

            best_guess = prediction.argmax(1) # (N)
            x = question[t] if random.random() < teacher_force_ratio else best_guess
        
        return predictions
