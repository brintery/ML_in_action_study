# Copyright (c) 2018 Mindon
#
# -*- coding:utf-8 -*-
# @Script: bayes.py
# @Author: Mindon
# @Email: gaomindong@live.com
# @Create At: 2018-05-04 17:49:24
# @Last Modified By: Mindon
# @Last Modified At: 2018-05-04 18:04:55
# @Description: bayes create data set.



def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocable_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set|set(document)
    return list(vocab_set)
    



