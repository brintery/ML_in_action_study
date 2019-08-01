# Copyright (c) 2018 Mindon
#
# -*- coding:utf-8 -*-
# @Script: bayes.py
# @Author: Mindon
# @Email: gaomindong@live.com
# @Create At: 2018-05-04 17:49:24
# @Last Modified By: Mindon Gao
# @Last Modified At: 2019-08-01 22:45:04
# @Description: bayes create data set.

# %%
import numpy as np
import math

# %%


def load_data_set():
    """
    load_data_set: create dataset

    Returns:
        posting_list: two demensional list, each list has several words
        class_vec: class label for each list
    """

    posting_list = [['my', 'dog', 'has', 'flea',
                     'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him',
                     'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so',
                     'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid',
                     'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak',
                     'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless',
                     'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocable_list(data_set):
    """
    create_vocable_list: create set from dataset

    Args:
        data_set: input dataset

    Returns:
        set of the dataset
    """
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_word2_vec(vocab_list, input_set):
    """
    check if word occur in sentence, it will set 1 if word in this sentence

    Args:
        vocab_list: the list of all words
        input_set: input data set

    Returns:
        return_vec: the list of 1 or 0, which show if words in this sentence
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return return_vec


def train_nb(train_matrix, train_category):
    """
    train data with native bayes

    Args:
        train_matrix: word vec of each sentence
        train_category: label of each sentence

    Returns:
        p0_vect:
        p1_vect:
        p_abusive:
    """
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])

    p_abusive = sum(train_category)/float(num_train_docs)

    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)

    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = [math.log(x) for x in p1_num/p1_denom]
    p0_vect = [math.log(x) for x in p0_num/p0_denom]

    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2_classify, p0_vect, p1_vect, p_abusive):
    """
    change multipy to add through log
    multipy：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
    add：P(F1|C)*P(F2|C)....P(Fn|C)P(C) ->
        log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))

    Args:
        vec2_classify: word vect which is need to classify
        p0_vect: class 0, the P(Wi|C0)
        p1_vect: class 1, the P(Wi|C1)
        p_abusive: probility of abusive, P(C1)

    Returns:
        1 or 0
    """
    p1 = sum(vec2_classify * p1_vect) + math.log(p_abusive)
    p0 = sum(vec2_classify * p0_vect) + math.log(1.0 - p_abusive)

    if p1 > p0:
        return 1
    else:
        return 0


# %%
list_posts, list_class = load_data_set()

# %%
vocab_list = create_vocable_list(list_posts)

# %%
train_mat = []
for sentence in list_posts:
    train_mat.append(set_of_word2_vec(vocab_list, sentence))

# %%
p0_vect, p1_vect, p_abusive = train_nb(np.array(train_mat),
                                       np.array(list_class))

# %%
test_entry = ['love', 'my', 'dalmation']
this_doc = np.array(set_of_word2_vec(vocab_list, test_entry))
print(test_entry, 'classified as: ', classify_nb(
    this_doc, p0_vect, p1_vect, p_abusive))
test_entry = ['stupid', 'garbage']
this_doc = np.array(set_of_word2_vec(vocab_list, test_entry))
print(test_entry, 'classified as: ', classify_nb(
    this_doc, p0_vect, p1_vect, p_abusive))
