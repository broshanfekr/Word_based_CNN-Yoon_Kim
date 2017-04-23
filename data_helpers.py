import numpy as np
import re
import itertools
from collections import Counter
import tarfile
from bs4 import BeautifulSoup
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
import gensim
import copy
from nltk.tokenize import wordpunct_tokenize
import random


def Load_Model(name='./myIMDB_model.d2v'):
    return Word2Vec.load(name)

def clean_str(review_docs, max_seq_len_cutoff = 50):

    output_docs = []

    for string in review_docs:
        words = wordpunct_tokenize(string)

        if(len(words) > max_seq_len_cutoff):
            words = words[:max_seq_len_cutoff]

        for index, w in enumerate(words):
            if (w.replace('.', '', 1).isdigit()):
                words[index] = '<num>'
        output_docs.append(words)
    return output_docs

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv.append('<pad>')
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    index_list = []
    for word in sentences:
        tmp = vocabulary[word]
        index_list.append(tmp)
    x = np.array(index_list)
    return x

def build_input_data_from_word2vec(sentence, word2vec_vocab, word2vec_vec):
    """
    Maps sentenc and vectors based on a word2vec model.
    """
    X_data = []
    for word in sentence:
        try:
            word2vec_index = word2vec_vocab[word].index
            word_vector = word2vec_vec[word2vec_index]
        except:
            word2vec_index = word2vec_vocab['<un_known>'].index
            word_vector = word2vec_vec[word2vec_index]
            #word_vector = np.random.uniform(low=-0.25, high=0.25, size=word2vec_vec.shape[1])
        X_data.append(word_vector)
    X_data = np.asarray(X_data)
    return X_data

def load_data(dataset_path, word2vec_model_path, n_class=2, max_seq_len_cutoff=50):
    """
    Loads and preprocessed data from dataset file.
    """

    dataset_file = open(dataset_path, "r", encoding='utf-8')
    dataset_content = dataset_file.readlines()

    x_text = []
    y = []
    for element in dataset_content:
        element = element.lower()
        element = element.split("\t")
        label = int(element[0])
        text = element[1].strip()
        if (len(text) == 0):
            continue
        x_text.append(text)
        tmp_lable = np.zeros(n_class)
        if(n_class == 2):
            tmp_lable[label] = 1
        else:
            tmp_lable[label - 1] = 1
        y.append(tmp_lable)


    x_text = clean_str(x_text, max_seq_len_cutoff)

    sequence_length = max(len(x) for x in x_text)

    vocabulary, vocabulary_inv = build_vocab(x_text)
    y = np.asarray(y)

    word2vec_Model = Load_Model(word2vec_model_path)
    word2vec_vocab = word2vec_Model.vocab
    word2vec_vec = word2vec_Model.syn0

    print("word2vec len is: ", len(word2vec_vec))
    tmp = word2vec_vocab['real']
    tmp1 = copy.deepcopy(tmp)
    word_vector = np.random.uniform(low=-0.25, high=0.25, size=(1,word2vec_vec.shape[1]))
    word2vec_vec = np.append(word2vec_vec, word_vector, axis=0)
    tmp1.index = len(word2vec_vec)-1
    word2vec_vocab['<un_known>'] = tmp1

    return [x_text, y, sequence_length, vocabulary, vocabulary_inv, word2vec_vocab, word2vec_vec]

def batch_iter(data, batch_size, seq_length, emmbedding_size,word2vec_vocab, word2vec_vec, is_shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1


    # Shuffle the data at each epoch
    if is_shuffle:
        random.shuffle(data)


    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        tmp_data = copy.deepcopy(data[start_index:end_index])

        batch_seq_len = []
        tmp_docs = []
        tmp_labels = []
        for x in tmp_data:
            batch_seq_len.append(len(x[0]))
            doc_vector = build_input_data_from_word2vec(x[0], word2vec_vocab, word2vec_vec)
            if(len(doc_vector) < seq_length):
                num_padding = seq_length - len(x[0])
                x_bar = np.zeros([num_padding, emmbedding_size])
                doc_vector = np.concatenate([doc_vector, x_bar], axis=0)
            tmp_docs.append(doc_vector)
            tmp_labels.append(x[1])
        tmp_docs = np.asarray(tmp_docs)
        tmp_labels = np.asarray(tmp_labels)

        tmp_data = list(zip(tmp_docs, tmp_labels))
        yield [tmp_data, batch_seq_len]