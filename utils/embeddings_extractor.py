#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Utilities for extracting embeddings for a given composition dataset.
"""

import argparse
from pathlib import Path
import numpy as np

from gensim_utils import read_gensim_model, save_gensim_model_preserve_order

def read_data_file(file_name, separator):
    words, phrases = set(), set()
    phrase_counter = 0

    with open(file_name, mode='r', encoding='utf8') as in_file:
        for line in in_file:
             splits = line.strip().split(separator)
             assert(len(splits) == 3), "error: incorrect line in data file: %s" % line

             w1 = splits[0]
             w2 = splits[1]
             phrase = splits[2]

             words.add(w1)
             words.add(w2)
             phrases.add(phrase)
             phrase_counter += 1

    # phrases should be unique
    assert(len(phrases) == phrase_counter), "error: duplicate phrases in %s?" % file_name

    return words, phrases

def read_dataset(dataset_dir, data_splits):
    all_words = set()
    all_phrases = set()
    for split in data_splits:
        file_name = str(dataset_dir.joinpath(split + "_text.txt"))
        word_set, phrase_set = read_data_file(file_name, " ")
        all_words.update(word_set)
        all_phrases.update(phrase_set)

    return all_words, all_phrases

def get_embedding_subset(full_embeddings, words, unk_key=None):
    word_idxs = []
    available_words = []
    not_found = 0

    if unk_key:
        word_idxs.append(full_embeddings.wv.vocab[unk_key].index)
        available_words.append(unk_key)

    for w in words:
        if w in full_embeddings.wv.vocab:
            word_idxs.append(full_embeddings.wv.vocab[w].index)
            available_words.append(w)
        else:
            print("%s not found" % w)
            not_found += 1

    word_reprs = np.take(full_embeddings.wv.syn0, word_idxs, axis=0)
    print("%d out of %d words not found" % (not_found, len(words)))

    return word_reprs, available_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", dest="dataset_dir", type=str, help="directory containing the text dataset (train_text.txt, test_text.txt, dev_text.txt)")
    parser.add_argument("-emb_file", dest="emb_file", type=str, help="file containing word embeddings (unfiltered)")
    parser.add_argument("-output_dir", dest="output_dir", type=str, help="output directory for writing the embeddings")
    parser.add_argument("--unk_key", dest="unk_key", type=str, help="unknown word key", default="<unk>")

    args = parser.parse_args()

    data_splits = ["train", "test", "dev"]
    all_words, all_phrases = read_dataset(Path(args.dataset_dir), data_splits)
    full_embeddings = read_gensim_model(args.emb_file)

    all_items = sorted(all_words) + sorted(all_phrases)
    item_reprs, available_items = get_embedding_subset(full_embeddings, all_items, args.unk_key)
    print("Vocabulary size %d" % len(available_items))
    print("Embeddings shape", item_reprs.shape)

    emb_path = Path(args.output_dir)
    if not emb_path.exists():
        emb_path.mkdir()
    emb_file_name = str(emb_path.joinpath("%s_filtered.txt" % str(Path(args.emb_file).stem)))
    save_gensim_model_preserve_order(available_items, item_reprs, emb_file_name)
    print("Embeddings written to %s" % emb_file_name)
