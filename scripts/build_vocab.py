import pickle


word_vocab_file = "../data/word_vocab.pkl"
entity_vocab_file = "../data/entity_vocab.pkl"
realtion_vocab_file = "../data/relation_vocab.pkl"

train_file = "../data/annotated_fb_data_train.txt"
valid_file = "../data/annotated_fb_data_valid.txt"
test_file = "../data/annotated_fb_data_test.txt"

files = [train_file, valid_file, test_file]

word_vocab = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
entity_vocab = {"<UNK>": 0}
relation_vocab = {"<UNK>": 0}
word_idx, entity_idx, relation_idx = 4, 1, 1
for fname in files:
    with open(fname, 'r', encoding='utf-8', newline='\n') as f:
        for line in f:
            datas = line.split('\t')
            if len(datas) != 4:
                continue
            s, r, o, q = datas
            if s not in entity_vocab:
                entity_vocab[s] = entity_idx
                entity_idx += 1
            if r not in relation_vocab:
                relation_vocab[r] = relation_idx
                relation_idx += 1
            if o not in entity_vocab:
                entity_vocab[o] = entity_idx
                entity_idx += 1
            words = q.split()
            for w in words:
                if w not in word_vocab:
                    word_vocab[w] = word_idx
                    word_idx += 1
# save the vocabs as binary pickle file
pickle.dump(entity_vocab, open(entity_vocab_file, "wb"))
pickle.dump(relation_vocab, open(realtion_vocab_file, "wb"))
pickle.dump(word_vocab, open(word_vocab_file, "wb"))
