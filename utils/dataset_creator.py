#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Utilities for creating vadj_n composition datasets.
"""

import argparse
import math
from pathlib import Path
import random
import numpy as np
from types import SimpleNamespace

from gensim.models.keyedvectors import Vocab, Word2VecKeyedVectors

from gensim_utils import read_gensim_model, save_gensim_model

def read_phrases(all_phrase_emb, separator):
    phrase_list = []
    phrase_set = set()
    for phrase in all_phrase_emb.wv.vocab:
        phrase = phrase.strip()
        if separator in phrase:
            parts = phrase.split(separator)
            assert(len(parts) == 2), "error: invalid entry %s" % phrase
            phrase_list.append((parts[0], parts[1], phrase))
            phrase_set.add(phrase)

    assert(len(phrase_set) == len(phrase_list)), "error: phrase duplicates in the dataset?"
    return phrase_list

def partition_data(phrase_list, percentages_list, labels_list):
    assert(math.isclose(sum(percentages_list), 1.0, rel_tol=1e-2)), "error: percentages should sum to 1"
    assert(len(percentages_list) == len(labels_list)), "error: mismatch between labels & percentages"

    partitions = {}
    last_idx = 0

    phrases = phrase_list
    n_phrases = len(phrases)
    for (label, percentage) in zip(labels_list, percentages_list):
        current_size = int(percentage * n_phrases)
        partitions[label] = phrases[:current_size]
        phrases = phrases[current_size:]
    partitions[label].extend(phrases) # add any remaining phrases to the last partition

    all_partitions_size = sum([len(v) for k,v in partitions.items()])
    assert(all_partitions_size == len(phrase_list)), "error: faulty partitioning"

    return partitions

def get_embedding_subset(full_embeddings, words, existing_unk_key, generated_unk_key):
    word_idxs = []
    available_words = []
    for w in words:
        if w in full_embeddings.wv.vocab:
            if w == existing_unk_key:
                word_idxs.append(full_embeddings.wv.vocab[existing_unk_key].index)
                available_words.append(generated_unk_key)
            else:
                word_idxs.append(full_embeddings.wv.vocab[w].index)
                available_words.append(w)
    word_reprs = np.take(full_embeddings.wv.syn0, word_idxs, axis=0)
    assert(len(word_reprs) == len(words)), "error: length mismatch when extracting embeddings"
    assert(len(set(available_words)) == len(words)), "error: duplicates?"

    return word_reprs, available_words

def filter_data(data, all_w1_emb, all_w2_emb, all_phrases_emb):
    filtered_data = []
    for tup in data:
        w1, w2, phrase = tup
        has_w1 = w1 in all_w1_emb.wv.vocab
        has_w2 = w2 in all_w2_emb.wv.vocab
        has_phrase = phrase in all_phrase_emb.wv.vocab

        if (has_w1 and has_w2 and has_phrase):
            filtered_data.append(tup)

    print("Filtered data, %d out of %d remaining entries" % (len(filtered_data), len(data)))
    return filtered_data

def write_data_subsets(partitions, output_dir_path, subset_names):
    appends = ["", "_v", "_a_v"]

    subsets = []
    for i in range(len(subset_names)):
        print("%s subset" % subset_names[i])
        dir_path = output_dir_path.joinpath(subset_names[i])
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        w1_set = set()
        w2_set = set()
        phrase_set = set()

        # write the train/test/dev subsets
        for partition_name, partition_entries in partitions.items():
            file_name = str(dir_path.joinpath(partition_name + "_text.txt"))
            with open(file_name, mode="w", encoding="utf8") as f_out:
                for tup in partition_entries:
                    f_out.write("%s %s %s\n" % (tup[0] + appends[i], tup[1], tup[2]))
                    w1_set.add(tup[0] + appends[i])
                    w2_set.add(tup[1])
                    phrase_set.add(tup[2])

            print("%s written with %d entries" % (file_name, len(partition_entries)))
        print("%s subset: %d w1s, %d w2s, %d phrases" % (subset_names[i], 
                    len(w1_set), len(w2_set), len(phrase_set)))
        subsets.append((dir_path, w1_set, w2_set, phrase_set))

    return subsets

def write_embeddings(subset, emb_name, all_w1_emb, all_w2_emb, all_phrase_emb, unk_key,
    all_w1_file_name, all_w2_file_name, all_phrases_file_name):
        dir_path, w1_set, w2_set, phrase_set = subset
        emb_path = dir_path.joinpath("embeddings")
        if not emb_path.exists():
            emb_path.mkdir()

        # write a file containing the word and phrase embeddings; 
        # the <unk> vectors are mapped to <unk_w1> and <unk_w2> for w1 and w2 and kept as <unk>
        # for the phrase

        emb_file_name = str(emb_path.joinpath("word_phrase_%s_%s_%s_%s.bin" % (emb_name, 
            str(Path(all_w1_file_name).stem), 
            str(Path(all_w2_file_name).stem),
            str(Path(all_phrases_file_name).stem))))

        w1_list = sorted(list(w1_set))
        w1_embeddings, available_w1 = get_embedding_subset(all_w1_emb, w1_list, unk_key, "<unk_w1>")
        w2_list = sorted(list(w2_set))
        w2_embeddings, available_w2 = get_embedding_subset(all_w2_emb, w2_list, unk_key, "<unk_w2>")
        phrase_list = sorted(list(phrase_set))
        phrase_embeddings, available_phrases = get_embedding_subset(all_phrase_emb, phrase_list, unk_key, unk_key)

        available_w = available_w1
        available_w.extend(available_w2)
        print("Vocabulary size %d" % len(available_w))
        available_w.extend(available_phrases)
        print("Total size %d" % len(available_w))

        word_embeddings = np.append(w1_embeddings, w2_embeddings, axis=0)
        w_embeddings = np.append(word_embeddings, phrase_embeddings, axis=0)
        save_gensim_model(available_w, w_embeddings, emb_file_name)        

def process_path(path):
    splits = path.strip().split(",")
    print("extracting embeddings from", splits)
    assert(len(splits) == 3), "error: wrong path %s" % path

    path = SimpleNamespace(
        w1_embeddings_file = splits[0],
        w2_embeddings_file = splits[1],
        phrase_embeddings_file = splits[2])

    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", dest="output_dir", type=str, help="output directory for writing the datasets")
    parser.add_argument("--separator", type=str, help="constituent separator", default="_vadj_n_")

    parser.add_argument("--glove", dest="glove_paths", type=str, help="embeddings files: w1,w2,phrase (.bin format)", default="")
    parser.add_argument("--skipgram", dest="skipgram_paths", type=str, help="embeddings files: w1,w2,phrase (.bin format)", default="")
    parser.add_argument("--cbow", dest="cbow_paths", type=str, help="embeddings files: w1,w2,phrase (.bin format)", default="")

    args = parser.parse_args()
    random.seed(1)

    if args.glove_paths == "" and args.skipgram_paths == "" and args.cbow_paths == "":
        print("Please specify at least one set of embeddings")
    else:
        paths = {}
        if args.glove_paths:
            paths["glove"] = process_path(args.glove_paths)
        if args.skipgram_paths:
            paths["skipgram"] = process_path(args.skipgram_paths)
        if args.cbow_paths:
            paths["cbow"] = process_path(args.cbow_paths)

        subsets = []
        for emb_name, emb_paths in paths.items():
            all_w1_emb = read_gensim_model(emb_paths.w1_embeddings_file)
            all_w2_emb = read_gensim_model(emb_paths.w2_embeddings_file)
            all_phrase_emb = read_gensim_model(emb_paths.phrase_embeddings_file)

            if not subsets:
                data = read_phrases(all_phrase_emb, args.separator)
                filtered_data = filter_data(data, all_w1_emb, all_w2_emb, all_phrase_emb)
                random.shuffle(filtered_data)

                partitions = partition_data(filtered_data, [0.7, 0.2, 0.1], ["train", "test", "dev"])

                output_dir_path = Path(args.output_dir)
                subset_names = ["a_n", "v_n", "av_n"]

                subsets = write_data_subsets(partitions, output_dir_path, subset_names)

            for subset in subsets:
                write_embeddings(subset, emb_name, all_w1_emb, all_w2_emb, all_phrase_emb, "<unk>",
                    emb_paths.w1_embeddings_file, emb_paths.w2_embeddings_file, emb_paths.phrase_embeddings_file)
