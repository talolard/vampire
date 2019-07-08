import argparse
import json
import os
from typing import List, Dict
import time
import nltk
import numpy as np
import pandas as pd
import spacy
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

import multiprocessing
from multiprocessing import Pool
from allennlp.common.file_utils import cached_path
from vampire.common.util import read_text, save_sparse, write_to_json


def load_data(data_path: str, tokenize: bool = False, tokenizer_type: str = "just_spaces", metadata_fields: List[str] = None) -> List[Dict[str, str]]:
    if tokenizer_type == "just_spaces":
        tokenizer = SpacyWordSplitter()
    elif tokenizer_type == "spacy":
        nlp = spacy.load('en')
        tokenizer = Tokenizer(nlp.vocab)
    elif tokenizer_type == "twitter":
        tokenizer = TweetTokenizer()
    tokenized_examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            example = json.loads(line)
            if tokenize:
                if tokenizer_type == 'just_spaces':
                    tokens = list(map(str, tokenizer.split_words(example['text'])))
                elif tokenizer_type == 'spacy':
                    tokens = list(map(str, tokenizer(example['text'])))
                elif tokenizer_type == "twitter":
                    tokens = tokenizer.tokenize(example['text'])
                text = ' '.join(tokens)
            else:
                text = example['text']
            if metadata_fields:
                metadata = {metadata: example[metadata] for metadata in metadata_fields}
                example = metadata
                example['text'] = text

            else:
                example = { 'text': text}
            tokenized_examples.append(example)
    return tokenized_examples

def write_list_to_file(ls, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "w+")
    for example in ls:
        out_file.write(example)
        out_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path(s) to the train jsonl file(s).")
    parser.add_argument("--dev-path", type=str, required=False,
                        help="Path to the dev jsonl file.")
    parser.add_argument("--stopwords-path", type=str, required=False,
                        help="Path to the stopwords file.")
    parser.add_argument("--serialization-dir", "-s", type=str, required=True,
                        help="Path to store the preprocessed output.")
    parser.add_argument("--vocab-size", type=int, required=False, default=10000,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--vocabulary", type=str, required=False, default=None,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--tokenize", action='store_true',
                        help="Path to store the preprocessed corpus vocabulary (output file name).") 
    parser.add_argument("--tokenizer-type", type=str, default="just_spaces",
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--metadata", nargs="+", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    args = parser.parse_args()

    if not os.path.isdir(args.serialization_dir):
        os.mkdir(args.serialization_dir)
    
    vocabulary_dir = os.path.join(args.serialization_dir, "vocabulary")

    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    
    tokenized_train_examples = load_data(cached_path(args.train_path), args.tokenize, args.tokenizer_type, args.metadata)
    master_train_examples = [example['text'] for example in tokenized_train_examples]

    if args.dev_path:
        tokenized_dev_examples = load_data(cached_path(args.dev_path), args.tokenize, args.tokenizer_type, args.metadata)
        master_dev_examples = [example['text'] for example in tokenized_dev_examples]
    else:
        master_dev_examples = []

    if args.stopwords_path:
        stopwords = read_text(args.stopwords_path)
    else:
        stopwords = "english"

    vocabulary = read_text(args.vocabulary) if args.vocabulary else None
    vocab_size = args.vocab_size if not args.vocabulary else len(vocabulary)

    count_vectorizer = CountVectorizer(stop_words=stopwords, max_features = vocab_size, vocabulary=vocabulary, token_pattern=r'\b[^\d\W]{3,30}\b')

    print("fitting count vectorizer...")
    
    text = master_train_examples + master_dev_examples

    count_vectorizer.fit(tqdm(text, desc="all"))

    print("transforming examples...")
    vectorized_train_examples = count_vectorizer.transform(tqdm(master_train_examples, desc="train"))
    if args.dev_path:
        vectorized_dev_examples = count_vectorizer.transform(tqdm(master_dev_examples, desc="dev"))
   
    # add @@unknown@@ token vector
    if not args.vocabulary:
        vectorized_train_examples = sparse.hstack((np.array([0] * len(master_train_examples))[:,None], vectorized_train_examples))
        if args.dev_path:
            vectorized_dev_examples = sparse.hstack((np.array([0] * len(master_dev_examples))[:,None], vectorized_dev_examples))

    if args.dev_path:    
        master = sparse.vstack([vectorized_train_examples, vectorized_dev_examples])
    else:
        master = vectorized_train_examples

    # generate background frequency
    print("generating background frequency...")
    bgfreq = dict(zip(count_vectorizer.get_feature_names(), np.asarray(master.sum(1)).squeeze(1) / args.vocab_size))
    
    print("saving data...")
    save_sparse(vectorized_train_examples, os.path.join(args.serialization_dir, "train.npz"))
    if args.dev_path:
        save_sparse(vectorized_dev_examples, os.path.join(args.serialization_dir, "dev.npz"))

    write_to_json(bgfreq, os.path.join(args.serialization_dir, "vampire.bgfreq"))
    
    if args.metadata:
        for field in args.metadata:
            metadata_train_items = [str(item[field]) for item in tokenized_train_examples]
            write_list_to_file(metadata_train_items, os.path.join(args.serialization_dir, field + "_train.txt"))

            if args.dev_path:
                metadata_dev_items = [str(item[field]) for item in tokenized_dev_examples]
                write_list_to_file(metadata_dev_items, os.path.join(args.serialization_dir, field + "_dev.txt"))
            else:
                metadata_dev_items = []

            write_list_to_file(['@@UNKNOWN@@'] + list(set(metadata_train_items + metadata_dev_items)), os.path.join(vocabulary_dir, field + "_labels.txt"))

    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(), os.path.join(vocabulary_dir, "vampire.txt"))
    write_list_to_file(['*tags', '*labels', 'vampire'], os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))

