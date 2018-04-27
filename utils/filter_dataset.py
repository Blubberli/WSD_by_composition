#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Utilities for filtering a composition dataset based on a set of embeddings. 
Only the entries where both words and the phrase are available are kept in the dataset.
"""

import argparse
from pathlib import Path

from gensim_utils import read_gensim_model, save_gensim_model_preserve_order
from embeddings_extractor import get_embedding_subset

def is_available(word, vocab):
    if word in vocab:
        return True
    return False

def filter_data_file(input_file_name, output_file_name, separator, embeddings):
    total, filtered = 0, 0
    words = []
    phrases = []
    with open(input_file_name, mode='r', encoding='utf8') as in_file, \
         open(output_file_name, mode='w', encoding='utf8') as out_file:

        for line in in_file:
             splits = line.strip().split(separator)
             assert(len(splits) == 3), "error: incorrect line in data file: %s" % line

             w1 = splits[0]
             w2 = splits[1]
             phrase = splits[2]
             total += 1

             vocab = embeddings.wv.vocab

             if is_available(w1, vocab) and \
                is_available(w2, vocab) and \
                is_available(phrase, vocab): 

                words.append(w1)
                words.append(w2)
                phrases.append(phrase)

                out_file.write("%s\n" % separator.join([w1, w2, phrase]))
                filtered += 1
    return total, filtered, words, phrases

def filter_dataset(input_dir, output_dir, data_splits, embeddings_file, unk_key):
    embeddings = read_gensim_model(embeddings_file)
    all_words, all_phrases = [], []
    all_filtered, all_total = 0, 0

    for split in data_splits:
        input_file_name = str(input_dir.joinpath(split + "_text.txt"))
        output_file_name = str(output_dir.joinpath(split + "_text.txt"))
        total, filtered, words, phrases = filter_data_file(input_file_name, output_file_name, " ", embeddings)
        print("%d out of %d entries written in %s" % (filtered, total, output_file_name))
        all_filtered += filtered
        all_total += total
        all_words.extend(words)
        all_phrases.extend(phrases)

    print("Total dataset: %d out of %d" % (all_filtered, all_total))
    print("Vocabulary: %d items" % len(set(all_words)))

    if (all_filtered < all_total):
        # filter & rewrite the word embeddings to match the current dataset
        all_items = sorted(set(all_words)) + sorted(set(all_phrases))
        items_reprs, available_items = get_embedding_subset(embeddings, all_items, unk_key)
    
        emb_path = Path(args.output_dir).joinpath("embeddings")
        if not emb_path.exists():
            emb_path.mkdir()

        emb_file_name = str(emb_path.joinpath("%s_dataset.txt" % str(Path(embeddings_file).stem)))
        save_gensim_model_preserve_order(available_items, items_reprs, emb_file_name)
        print("Embeddings written to %s" % emb_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", dest="input_dir", type=str, help="directory containing the text dataset (train_text.txt, test_text.txt, dev_text.txt)")
    parser.add_argument("-emb_file", dest="emb_file", type=str, help="file containing word embeddings")
    parser.add_argument("-output_dir", dest="output_dir", type=str, help="directory to save the filtered variant")
    parser.add_argument("--unk_key", dest="unk_key", type=str, help="unknown word key", default="<unk>")

    args = parser.parse_args()

    data_splits = ["train", "test", "dev"]
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    filter_dataset(Path(args.input_dir), output_dir, data_splits, args.emb_file, args.unk_key)
