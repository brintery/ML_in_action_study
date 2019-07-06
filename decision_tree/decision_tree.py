# Copyright (c) 2018 Mindon
#
# -*- coding:utf-8 -*-
# @Script: decision_tree.py
# @Author: Mindon
# @Email: gaomindong@live.com
# @Create At: 2018-03-08 14:31:10
# @Last Modified By: Mindon Gao
# @Last Modified At: 2019-07-06 23:44:02
# @Description: algorithm of decision tree.

from math import log
import operator
import pickle
import copy

# create test dataset
def create_dataset():
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], 
                [1, 0, 'no'], [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    
    return data_set, labels

# calculate shannon entrotype
def calc_shannonent(data_set):
    # the row length of the dataset
    num_entries = len(data_set)
    # counts the number when the label occur
    label_count = {}
    for feat in data_set:
        current_label = feat[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1
    
    # calculate shannon entrotype
    shannon_ent = 0.0
    for key in label_count:
        # p(xi)
        prob = float(label_count[key])/num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_dataset(data_set, index, value):
    """
    split_dataset: use feat[index]=value split dataset, exclude column of this index
    Args:
        index: the column of the index(feature) for each row(sample)
        value: the value of the column of the index
    Returns:
        ret_dataset: the sub dataset when the feat[index]=value, like the serval leaf node
    """
    ret_dataset = []
    for feat in data_set:
        if feat[index] == value:
            split_dataset = feat[:index]
            split_dataset.extend(feat[index+1:])
            ret_dataset.append(split_dataset)
    return ret_dataset


def choose_best_feature(data_set):
    num_feat = len(data_set[0])-1
    base_ent = calc_shannonent(data_set)
    best_info_gain, best_feat = 0.0, -1

    for i in range(num_feat):
        feat_list = [example[i] for example in data_set]
#        print(feat_list)
        unique_val = set(feat_list)
        new_ent = 0.0
        
        for value in unique_val:
            sub_dataset = split_dataset(data_set, i, value)
            prob = len(sub_dataset)/float(len(data_set))
            new_ent += prob*calc_shannonent(sub_dataset)
        
        info_gain = base_ent - new_ent
        print('info gain = ', info_gain, 
            'best feature = ', i, base_ent, new_ent)
        if info_gain>best_info_gain:
            best_info_gain = info_gain
            best_feat = i
    return best_feat


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_list.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_classcount = sorted(class_count.items(), key=operator.itemgetter(1), 
                               reverse=True)
    return sorted_classcount[0][0]


def creat_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    
    best_feat = choose_best_feature(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])
    
    feat_values = [example[best_feat] for example in data_set]
    unique_val = set(feat_values)
    for value in unique_val:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = creat_tree(split_dataset(data_set, 
               best_feat,value),  sub_labels)
    return my_tree

def classfy(input_tree, feat_label, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_label.index(first_str)
    
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    print('+++', first_str, 'xxx', second_dict, '---', key, '>>>', value_of_feat)
    
    if isinstance(value_of_feat, dict):
        class_label = classfy(value_of_feat, feat_label, test_vec)
    else:
        class_label = value_of_feat
    return class_label

def store_tree(input_tree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(input_tree, fw)
    

def grab_tree(filename):
    with open(filename, 'rb') as fr:
        tree = pickle.load(fr)
    return tree


data, label = create_dataset()
tree = creat_tree(data, copy.deepcopy(label))
label = classfy(tree, label, [1, 1])
#store_tree(tree, 'fish_or_flipper.txt')
#read_tree = grab_tree('fish_or_flipper.txt')
    
