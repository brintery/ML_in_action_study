# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:37:46 2018

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import operator
from os import listdir

def file2matrix(filename):
    number_of_lines = len(open(filename).readlines())
    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    
    for line in open(filename).readlines():
        format_line = line.strip().split('\t')
        return_mat[index, :] = format_line[0:3]
        class_label_vector.append(int(format_line[-1]))
        index += 1
    
    return(return_mat, class_label_vector)


def auto_norm(data):
    min_val = data.min(0)
    max_val = data.max(0)
    
    ranges = max_val - min_val
    norm_data = np.zeros(np.shape(data))
    m = data.shape[0]
    
    norm_data = data - np.tile(min_val, (m, 1))
    norm_data = norm_data / np.tile(ranges, (m, 1))
    
    return(norm_data, ranges, min_val)


def classify(inX, data, label, k):
    size = data.shape[0]
    dist = (((np.tile(inX, (size, 1)) - data)**2).sum(axis=1)**0.5)
    
    dist_label = dist.argsort()
    class_count = {}
    
    for i in range(k):
        vote_label = label[dist_label[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
        sorted_classcount = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return(sorted_classcount[0][0], class_count)


def dating_class_test():
    ho_rotio = 0.1
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_val = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_rotio)

    print('num_test_vecs= ', num_test_vecs)
    error_count = 0.0

    for i in range(num_test_vecs):
        classifier_result, class_label = classify(norm_mat[i, :], 
                                                  norm_mat[num_test_vecs:m, :],
                                                  dating_labels[num_test_vecs:m], 3)
        print("the classifier com back with: ", classifier_result, "the real answer is: ", 
              dating_labels[i])
        if(classifier_result != dating_labels[i]): 
            error_count += 1.0
    print("the total error rate is: ", error_count/float(num_test_vecs))
    print(error_count)
    return(classifier_result, class_label)
    
    


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    data, label = file2matrix('datingTestSet2.txt')
    norm_data, ranges, min_val = auto_norm(data)
    in_arr = np.array([ffMiles, percent_tats, ice_cream])
    classifier_result, class_label = classify((in_arr-min_val)/ranges, norm_data, label, 3)
    print("You will probably like this person: ", result_list[classifier_result - 1])



#data, label = file2matrix('datingTestSet2.txt')
#norm_data, ranges, min_val = auto_norm(data)

#ax = plt.figure().add_subplot(111)
#ax.scatter(data[:, 0], data[:, 1],  np.array(label), np.array(label))
#classify_person()
# class_result, class_label = dating_class_test()