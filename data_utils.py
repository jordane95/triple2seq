import pandas as pd
from torch.nn.utils.rnn import pad_sequence # pad batch
from torch.utils.data import Dataset, Dataloader

tokenizer = lambda x: x.split()

class Vocabulary:
    def __init__(self, freq_thershold):
        self.freq_thershold = freq_thershold
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"} # index to string
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3} # string to index
    
    def __len__(self):
        '''return the size of vocabulary'''
        return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [tok.lower() for tok in tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        ''' build the vocabulary from a list of sentences'''
        frequencies = dict()
        idx = 4
        for sentence in sentence_list:
            for word in tokenize(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] > self.freq_thershold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class SimpleQuestions(Dataset):
    def __init__(self, file_path, freq_thershold=5):
        self.df = pd.read_csv(file_path, sep='\t', header=None, names=["subject_id", "relation_id", "object_id", "question"])

        self.subject_ids = self.df["subject_id"]
        self.relation_ids = self.df["relation_id"]
        self.object_ids = self.df["object_id"]
        self.question = self.df["questions"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_thershold)
        self.vocab.build_vocab(self.question.tolist())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        subject_id = self.subject_ids[index]
        relation_id = self.relation_ids[index]
        object_id = self.object_ids[index]
        question = self.questions[index]

        numericalized_question = [self.vocab.stoi["<SOS>"]]
        numericalized_question += self.vocab.numericalize(question)
        numericalized_question.append(self.vocab.stoi["<EOS>"])

        return subject_id, relation_id, object_id, question
