import json
import pickle
import random

import nltk
import collections
import numpy as np
import torchtext
from nltk.corpus import stopwords
import string
import pandas as pd
import os

target_path = "./data_omp_top_150/"
o_length = 250
m_length = 0
p_length = 0

top = 150

print(top)
# print("o_length,m_length,p_length", o_length, m_length, p_length)

if not os.path.exists(target_path):
    os.makedirs(target_path)


def read_orig():
    f = open('data_omp_420_140_40/category_concepts.json')
    concepts = json.load(f)

    f = open('data_omp_420_140_40/trainval_concept_label.json')
    concept_label = json.load(f)

    with open('data_omp_420_140_40/coco_adj_concepts.pkl', 'rb') as f:
        adj_concepts = pickle.load(f)

    with open('data_omp_420_140_40/coco_concepts_glove_word2vec.pkl', 'rb') as f:
        concepts_glove = pickle.load(f)

    # with open('vocab_trainval_concepts.pkl', 'rb') as f:
    #     trainval_concepts = pickle.load(f)

    return concepts, adj_concepts, concepts_glove, concept_label


concepts, adj_concepts, concepts_glove, concept_label = read_orig()

category_concepts = {k: concepts[k] for k in list(concepts)[:top]}

with open(target_path + 'category_concepts.json', 'w', encoding='utf-8') as f:
    json.dump(category_concepts, f)

trainval_concept_label = []

for sentence in concept_label:
    label = []
    for i in sentence["concept_labels"]:
        if i < top:
            label.append(i)
    trainval_concept_label.append({"file_name": sentence["file_name"], "img_id": sentence["img_id"], "concept_labels": label})

with open(target_path + 'trainval_concept_label.json', 'w', encoding='utf-8') as f:
    json.dump(trainval_concept_label, f)

concepts_glove = concepts_glove[:top, :]

with open(target_path + 'coco_concepts_glove_word2vec.pkl', 'wb') as f:
    pickle.dump(concepts_glove, f, pickle.HIGHEST_PROTOCOL)

adj_concepts_gen = {}
adj_concepts_gen['nums'] = adj_concepts['nums'][:top]
adj_concepts_gen['adj_all'] = adj_concepts['adj_all'][:top, :top]
# adj_concepts['adj_O_P'] = adj_concepts['adj_O_P'][:top, :top]
# adj_concepts['adj_O_M'] = adj_concepts['adj_O_M'][:top, :top]

with open(target_path + 'coco_adj_concepts.pkl', 'wb') as f:
    pickle.dump(adj_concepts_gen, f, pickle.HIGHEST_PROTOCOL)

