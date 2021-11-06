from data_utils import Vocab, SimpleQuestionsDataSet, get_data_loader, GloVeLoader, TransELoader # data processing

from triple2seq import Encoder, Decoder, Triple2Seq # model architecture

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load vocab and training dataset
word_vocab = Vocab(vocab_file='data/word_vocab.pkl')
entity_vocab = Vocab(vocab_file='data/entity_vocab.pkl')
relation_vocab = Vocab(vocab_file='data/relation_vocab.pkl')

training_set = SimpleQuestionsDataSet(mode='train')

data_loader = get_data_loader(dataset=training_set, batch_size=8, num_workers=8, shuffle=True, pin_memory=True)


# loading pretrained embeddings
glove = GloVeLoader(data_path='data/glove.6B.50d.txt', word_dim=50)

transe = TransELoader(entity_embedding_file='data/entity_transE_50d.pkl', relation_embedding_file='data/relation_transE_50d.pkl', embedding_size=50)

word_embeddings = glove.build_word_embeddings(word_vocab).to(device)

entity_embeddings, relation_embeddings = transe.load_knowledge_embedding()


# initialize the model
encoder = Encoder(
    entity_vocab_size=len(entity_vocab),
    relation_vocab_size=len(relation_vocab),
    embedding_size=50,
    hidden_size=100,
    knowledge_embeddings=(entity_embeddings.to(device), relation_embeddings.to(device))
).to(device)

decoder = Decoder(
    input_size=len(word_vocab),
    embedding_size=50,
    hidden_size=100,
    output_size=len(word_vocab),
    num_layers=1,
    p=0.5,
    word_embeddings=word_embeddings
).to(device)

model = Triple2Seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word_vocab.stoi["<PAD>"])

epochs = 10
learning_rate = 1e-3

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# training process

for epoch in range(epochs):
    for idx, (triples, questions) in enumerate(data_loader):

        triples = [atom.to(device) for atom in triples]
        questions = questions.to(device)

        predictions = model.forward(triples, questions)

        # predictions shape: (seq_length, batch_size, vocab_size)
        # questions shape: (seq_length, batch_size)
        predictions = predictions[1:].reshape(-1, predictions.shape[2])
        questions = questions[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(predictions.to(device), questions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    print("epoch {}: loss {}".format(epoch, loss))
