import pickle

import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class Vocab:
    ''' a general purpose vocabulary object which maps token into id '''
    def __init__(self, vocab_file):
        self.stoi = pickle.load(open(vocab_file, 'rb'))
        self.itos = {i:s for s, i in self.stoi.items()}
    
    def __len__(self):
        ''' return the size of vocabulary '''
        return len(self.stoi)


class SimpleQuestionsDataSet(Dataset):
    def __init__(self, mode='train', word_vocab=None, entity_vocab=None, relation_vocab=None):
        ''' read the train/valid/test dataset from txt file '''
        self.word_vocab = Vocab('data/word_vocab.pkl')
        self.entity_vocab = Vocab('data/entity_vocab.pkl')
        self.relation_vocab = Vocab('data/relation_vocab.pkl')
        
        file_name = "data/annotated_fb_data_" + mode + ".txt"
        
        self._dataset = list()
        
        with open(file_name, 'r', encoding='utf-8', newline='\n') as f:
            for line in f:
                tokens = line.split('\t')
                if (len(tokens)) != 4:
                    continue
                s, r, o, q = tokens
                subject_id = self.entity_vocab.stoi[s]
                relation_id = self.relation_vocab.stoi[r]
                object_id = self.entity_vocab.stoi[o]
                words_id = [self.word_vocab.stoi[w] for w in q.split()]
                self._dataset.append(((subject_id, relation_id, object_id), torch.tensor(words_id)))
    

    def __len__(self):
        return len(self._dataset)
    

    def __getitem__(self, index):
        return self._dataset[index]


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    

    def __call__(self, batch):
        ''' batch is a list of ((s, r, o), q) '''
        triples = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        entity_idxs = [triple[0] for triple in triples]
        relation_idxs = [triple[1] for triple in triples]
        object_idxs = [triple[2] for triple in triples]
        question_idxs = pad_sequence(questions, batch_first=False, padding_value=self.pad_idx)
        return (torch.tensor(entity_idxs), torch.tensor(relation_idxs), torch.tensor(object_idxs)), torch.tensor(question_idxs)


class GloVeLoader:
    def __init__(self, data_path='data/golve.6B.50d.txt', word_dim=50):
        self.data_path = data_path
        self.word_dim = word_dim
    

    @staticmethod
    def _load_word_vectors(data_path, word_dim):
        """ read the word vectors """
        with open(data_path, 'r', encoding='utf-8', newline='\n') as f:
            word_vec = dict()
            word_vec["<PAD>"] = np.zeros(word_dim).astype('float32')
            for line in f:
                tokens = line.strip().split()
                if (len(tokens)-1) != word_dim:
                    continue
                if tokens[0] == "<PAD>" or tokens[0] == "<UNK>":
                    continue
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype="float32")
        return word_vec
    

    def build_word_embeddings(self, vocab):
        """ build word embedding matrix for a given vocab """
        word_vec = GloVeLoader._load_word_vectors(self.data_path, self.word_dim)
        word_embeddings = np.random.uniform(-0.5, 0.5, (len(vocab), self.word_dim)).astype('float32')
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.itos[i])
            if vec is not None:
                word_embeddings[i] = vec
        return torch.tensor(word_embeddings)


class TransELoader:
    def __init__(self, entity_embedding_file='data/entity_transE_50d.pkl', relation_embedding_file='data/relation_transE_50d.pkl', embedding_size=50):
        self.entity_embedding_file = entity_embedding_file
        self.relation_embedding_file = relation_embedding_file
        self.embedding_size = embedding_size
    
    def load_knowledge_embedding(self):
        entity_embeddings = pickle.load(open(self.entity_embedding_file, "rb"))
        relation_embeddings = pickle.load(open(self.relation_embedding_file, "rb"))
        return torch.tensor(entity_embeddings), torch.tensor(relation_embeddings)


def get_data_loader(dataset=SimpleQuestionsDataSet(), batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    
    pad_idx = dataset.word_vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx)
    )

    return loader


if __name__ == "__main__":
    loader = get_loader()
