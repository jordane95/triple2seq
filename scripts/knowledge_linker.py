#/usr/bin/python3

# script to get knowledge embedding from pre-trained transE on FreeBase

import pickle
import numpy as np

data_path = "../data/"
freebase_path = "../Freebase/"

embedding_size = 50


# for entity and relation in SimpleQuestions
# load vocabulary: string to idx mapping
print("loading knowledge vocabulary of SimpleQuesitons")
entity_vocab = pickle.load(open(data_path+'entity_vocab.pkl', "rb"))
relation_vocab = pickle.load(open(data_path+'relation_vocab.pkl', "rb"))

# idx to string mapping
reversed_entity_vocab = {i:s for s, i in entity_vocab.items()}
reversed_relation_vocab = {i:s for s, i in relation_vocab.items()}

print("SimpleQuestion vocab successfully loaded")




# for large entity and relation from FreeBase
print("loading Freebase vocab...")
entity_to_id = dict()
relation_to_id = dict()

with open(freebase_path+"knowledge graphs/entity2id.txt") as f:
    total_entity = int(f.readline().strip())
    for line in f.readlines():
        entity, idx = line.strip().split('\t')
        entity_to_id[entity] = int(idx)

with open(freebase_path+"knowledge graphs/relation2id.txt") as f:
    total_relation = int(f.readline().strip())
    for line in f.readlines():
        relation, idx = line.strip().split('\t')
        relation_to_id[relation] = int(idx)

print('Freebase vocab sucessfully loaded')

print("linking knowledge embeddings")
entity_vec_file = freebase_path + "embeddings/dimension_50/transe/entity2vec.bin"
relation_vec_file = freebase_path + "embeddings/dimension_50/transe/relation2vec.bin"

entity_vec = np.memmap(entity_vec_file, dtype='float32', mode='r', shape=(total_entity, embedding_size))
relation_vec = np.memmap(relation_vec_file, dtype='float32', mode='r', shape=(total_relation, embedding_size))

entity_embeddings = np.random.uniform(-0.5, 0.5, (len(entity_vocab), embedding_size)).astype('float32')
relation_embeddings = np.random.uniform(-0.5, 0.5, (len(relation_vocab), embedding_size)).astype('float32')

# mapping between SimpleQuestions and FreeBase via name
for entity_id_sq, entity_name_sq in reversed_entity_vocab.items():
    # format transformation from sq (SimpleQuestions) to fb (Freebase)
    # for example: www.freebase.com/m/01jp8ww -> m.01jp8ww
    entity_name_fb = ".".join(entity_name_sq.split('/')[-2:])
    if entity_name_fb not in entity_to_id:
        continue
    entity_id_fb = entity_to_id[entity_name_fb]
    entity_embeddings[entity_id_sq] = entity_vec[entity_id_fb]

for relation_id_sq, relation_name in reversed_relation_vocab.items():
    # do not need transform because the format is the same
    if relation_name not in relation_to_id:
        continue
    relation_id_fb = relation_to_id[relation_name]
    relation_embeddigs[relation_id_sq] = relation_vec[relation_id_fb]

print("knowledge embedding sucessfully linked")



# save the knowledge embedding matrix
print("saving knowledge embeddings")
entity_embedding_file = data_path + "entity_transE_50d.pkl"
relation_embedding_file = data_path + "relation_transE_50d.pkl"

pickle.dump(entity_embeddings, open(entity_embedding_file, "wb"))
pickle.dump(relation_embeddings, open(relation_embedding_file, "wb"))

print("knowledge embedding sucessfully saved")
