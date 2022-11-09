import json
import pickle
import random

import nltk
from nltk.tag.stanford import StanfordPOSTagger

import collections
import numpy as np
import torchtext
from nltk.corpus import stopwords
import string
import pandas as pd
import os

target_path = "./data_omp_train_top_300_five_caption/"
o_length = 210
m_length = 70
p_length = 20

gen_from_file = False
all_captions = True  # True generate the adj matrix from five caption

print("o_length,m_length,p_length,gen_from_file, all_captions", o_length, m_length, p_length, gen_from_file,
      all_captions)

if not os.path.exists(target_path):
    os.makedirs(target_path)

# category concepts
Object, Motion, Property = ['NN', 'NNS'], ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], ['CD', 'JJ', 'JJR', 'RB']


def read_orig():
    f = open('Concept_annotations_coco/category_concepts.json')
    concepts = json.load(f)

    with open('Concept_annotations_coco/coco_adj_concepts.pkl', 'rb') as f:
        adj_concepts = pickle.load(f)

    return concepts, adj_concepts


concepts_gen, adj_concepts_gen = read_orig()

jar = 'stanford-postagger/stanford-postagger-4.2.0.jar'
model = 'stanford-postagger/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

tags = pos_tagger.tag(concepts_gen.keys())
Objects, Motions, Properties = [], [], []
for tag in tags:
    if tag[1] in Object:
        Objects.append(tag[0])
    elif tag[1] in Motion:
        Motions.append(tag[0])
    elif tag[1] in Property:
        Properties.append(tag[0])
print("Nums of concepts:", len(Objects), len(Motions), len(Properties))


def gen_pos(vocab_path, pos_tagger):
    f = open(vocab_path)
    word_vocab = json.load(f)
    tags = pos_tagger.tag(word_vocab['idx2word'].values())
    with open(target_path + 'pos.json', 'w', encoding='utf-8') as f:
        json.dump(tags, f)
    return tags

tags = gen_pos(vocab_path='../../../vocab/coco_precomp_vocab.json', pos_tagger=pos_tagger)

# Get captions
def get_captions(datapath, all_captions=True):
    f_data = open(datapath)
    coco_datasset = json.load(f_data)

    captions, tokens, file_image = [], [], []
    captions_train, captions_val, captions_test, captions_restval = 0, 0, 0, 0
    # train = train(82783) + restval(30504) =113287, val = 5000, test = 5000
    if all_captions:
        for caption in coco_datasset["images"]:
            if caption["split"] == "train":
                captions_train += 1
                sentences = []
                for sentence in caption["sentences"]:
                    file_image.append(caption["filename"])
                    tokens.append(sentence["tokens"])
                    sentences.append(sentence["raw"])
                captions.append(sentences)
            elif caption["split"] == "restval":
                captions_restval += 1
                sentences = []
                for sentence in caption["sentences"]:
                    file_image.append(caption["filename"])
                    tokens.append(sentence["tokens"])
                    sentences.append(sentence["raw"])
                captions.append(sentences)
    else:
        for caption in coco_datasset["images"]:
            # train
            if caption["split"] == "train":
                captions_train += 1
                for sentence in caption["sentences"]:
                    tokens.append(sentence["tokens"])
                    captions.append(sentence["raw"])
            elif caption["split"] == "restval":
                captions_restval += 1
                for sentence in caption["sentences"]:
                    tokens.append(sentence["tokens"])
                    captions.append(sentence["raw"])
            # Remove Test
            # elif caption["split"] == "val":
            #     captions_val += 1
            #     for sentence in caption["sentences"]:
            #         tokens.append(sentence["tokens"])
            #         captions.append(sentence["raw"])
            #
            # elif caption["split"] == "test":
            #     captions_test += 1
            #     for sentence in caption["sentences"]:
            #         tokens.append(sentence["tokens"])
            #         captions.append(sentence["raw"])

    # captions = [c for s in captions for c in s]

    return captions, tokens, coco_datasset


coco_data_path = "/ceph/das-scratch/users/mliang/data/coco/coco_Karpathy/coco/dataset.json"

captions, tokens, coco_dataset = get_captions(coco_data_path, all_captions)


# Words counter
def word_counter(captions, all_captions=True):
    words = []
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    stop_words.update(["is", "are", ".."])
    if all_captions:
        for caption in captions:
            token = set()
            for s in caption:
                # token.update([i for i in nltk.word_tokenize(s.lower()) if i not in stop_words])
                token.update(nltk.word_tokenize(s.lower()))
            words.append(list(token))
    else:
        for caption in captions:
            # words.append([i for i in nltk.word_tokenize(caption.lower()) if i not in stop_words])
            words.append(nltk.word_tokenize(caption.lower()))  # remove stop
    counter = collections.Counter([tk for st in words for tk in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    with open(target_path + 'words_counter.json', 'w', encoding='utf-8') as f:
        json.dump(counter, f)

    with open('words.json', 'w', encoding='utf-8') as f:
        json.dump(words, f)

    return counter, words

counter, words = word_counter(captions, all_captions)

def gen_concept_vocab(pos_tagger, counter, o_length=210, m_length=60, p_length=30):
    Objects, Motions, Properties = [], [], []
    vocab = []
    words = [w[0] for w in counter]
    tags = pos_tagger.tag(words)
    for tag in tags:
        if tag[1] in Object:
            Objects.append(tag[0])
            vocab.append(tag[0])
        elif tag[1] in Motion:
            Motions.append(tag[0])
            vocab.append(tag[0])
        elif tag[1] in Property:
            Properties.append(tag[0])
            vocab.append(tag[0])

    # select top word
    category_concepts = vocab[:o_length + m_length + p_length]

    # Keeping consistency with original paper
    # category_concepts = []
    # f = open('Concept_annotations_coco/category_concepts.json')
    # category_concepts = json.load(f)

    # for word, value in category_concepts_orig.items():
    #     if word in vocab:
    #         category_concepts.append(word)
    #     else:
    #         print(word)

    # restrict the ratio of the concepts
    # category_concepts = Objects[:o_length] + Motions[:m_length] + Properties[:p_length]

    # random select word
    # r_list = []
    # for i in range(o_length + m_length + p_length):
    #     while True:
    #         r = random.randint(0, len(vocab))
    #         if len(r_list) < (o_length + m_length + p_length):
    #             if r not in r_list:
    #                 r_list.append(r)
    #         else:
    #             break
    # category_concepts = [vocab[i] for i in r_list]

    category_concepts_dict = {}
    for i, category in enumerate(category_concepts):
        category_concepts_dict[category] = i

    with open(target_path + 'category_concepts.json', 'w', encoding='utf-8') as f:
        json.dump(category_concepts_dict, f)

    return category_concepts_dict, category_concepts


category_concepts_dict, category_concepts = gen_concept_vocab(pos_tagger, counter, o_length, m_length, p_length)

count = 0
for key, val in category_concepts_dict.items():
    if key in concepts_gen:
        count += 1
    else:
        print("key, index, nums:", key, category_concepts_dict[key])
print("Ratio of category_concepts", count / len(category_concepts))

# concept label
def gen_concepts_label(coco_dataset, category_concepts_dict):
    trainval_concept_label = []
    category_concepts = category_concepts_dict.keys()
    for caption in coco_dataset["images"]:
        words_caption = set()
        for sentence in caption["sentences"]:
            word = nltk.word_tokenize(sentence["raw"].lower())
            words_caption.update(word)
        label = []
        for word in words_caption:
            if word in category_concepts:
                label.append(category_concepts_dict[word])
        trainval_concept_label.append(
            {"file_name": caption["filename"], "img_id": caption["imgid"], "concept_labels": label})

    with open(target_path + 'trainval_concept_label.json', 'w', encoding='utf-8') as f:
        json.dump(trainval_concept_label, f)


gen_concepts_label(coco_dataset, category_concepts_dict)


# glove embedding
def get_vector(embeddings, word):
    assert word in embeddings.stoi, f'*{word}* is not in the vocab!'
    return embeddings.vectors[embeddings.stoi[word]]


def gen_glove_embedding(category_concepts):
    glove = torchtext.vocab.GloVe()
    concepts_word2vec = []
    for concept in category_concepts:
        concepts_word2vec.append(get_vector(glove, concept).cpu().detach().numpy())
    concepts_word2vec = np.array(concepts_word2vec)

    with open(target_path + 'coco_concepts_glove_word2vec.pkl', 'wb') as f:
        pickle.dump(concepts_word2vec, f, pickle.HIGHEST_PROTOCOL)


gen_glove_embedding(category_concepts)

# adj file
def gen_adj_file(category_concepts, words, counter, o_length=210, m_length=60, p_length=30, gen_from_file=False):
    counter = dict((x, y) for x, y in counter)
    length = o_length + m_length + p_length

    all_num = []
    for concept in category_concepts:
        if concept in counter.keys():
            all_num.append(counter[concept])
        else:
            all_num.append(1)
            print(concept)

    adj_all = np.zeros((length, length))
    for i, i_word in enumerate(category_concepts):
        for j, j_word in enumerate(category_concepts):
            if i >= j:
                continue
            cnt = 0
            if gen_from_file and i_word in concepts_gen.keys() and j_word in concepts_gen.keys():
                cnt = adj_concepts_gen["adj_all"][concepts_gen[i_word], concepts_gen[j_word]]
            else:
                for sent in words:
                    if i_word in sent and j_word in sent:
                        cnt += 1
            adj_all[i, j] = cnt
            adj_all[j, i] = cnt

    adj = {}
    adj['nums'] = np.array(all_num, dtype=np.int32)
    adj['adj_all'] = np.array(adj_all, dtype=np.int32)

    with open(target_path + 'coco_adj_concepts.pkl', 'wb') as f:
        pickle.dump(adj, f, pickle.HIGHEST_PROTOCOL)

    return adj

adj = gen_adj_file(category_concepts, words, counter, o_length, m_length, p_length, gen_from_file)
print("Finished")
