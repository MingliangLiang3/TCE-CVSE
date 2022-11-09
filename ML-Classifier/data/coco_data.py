import pandas as pd
import json
import numpy as np

# f = open('category_concepts.json')
with open('Concepts.json') as f:
    concepts = json.load(f)
coco_datapath = "../../../data/caption_datasets_karpathy_split/dataset_coco.json"


def get_captions(concepts, coco_data_path, split):
    f_data = open(coco_data_path)
    coco_dataset = json.load(f_data)

    captions, tokens, file_image = [], [], []
    # train = train(82783) + restval(30504) =113287, val = 5000, test = 5000
    concept_labels = []
    for caption in coco_dataset["images"]:
        if caption["split"] in split:
            sentences = []
            concept_label, self_concept_label = [], []
            tokens = []
            for sentence in caption["sentences"]:
                file_image.append(caption["filename"])
                tokens.append(sentence["tokens"])
                sentences.append(sentence["raw"].replace("\n", ""))

                labels = []
                for token in sentence["tokens"]:
                    if token in concepts.keys():
                        labels.append(concepts[token])
                concept_label.append(labels)
            five_cation_labels = set([c for s in concept_label for c in s])

            for sentence in sentences:
                multi_label = np.zeros(len(concepts))
                multi_label[list(five_cation_labels)] = 1
                concept_labels.append(list(multi_label))
                captions.append(sentence)

            # exclude concept in the caption
            # for sentence, concept in zip(sentences, concept_label):
            #     exclude_self = [c for c in five_cation_labels if c not in concept]
            #     multi_label = np.zeros(300)
            #     multi_label[exclude_self] = 1
            #     concept_labels.append(list(multi_label))
            #     captions.append(sentence)

    return captions, tokens, coco_dataset, concept_labels

def data(captions, concept_labels):
    data = []
    for caption, concept in zip(captions, concept_labels):
        data.append([caption] + concept)
    df = pd.DataFrame(data, columns=["caption"] + list(concepts.keys()))
    return df


captions, tokens, coco_dataset, concept_labels = get_captions(concepts, coco_datapath, split=["train", "restval"])
df = data(captions, concept_labels)
df.to_csv("coco/coco_train.csv", index=False, sep="\t")
print(df.head())

captions, tokens, coco_dataset, concept_labels = get_captions(concepts, coco_datapath, split=["val"])
df = data(captions, concept_labels)
df.to_csv("coco/coco_val.csv", index=False, sep="\t")
print(df.head())

captions, tokens, coco_dataset, concept_labels = get_captions(concepts, coco_datapath, split=["test"])
df = data(captions, concept_labels)
df.to_csv("coco/coco_test.csv", index=False, sep="\t")
print(df.head())
