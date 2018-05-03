# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:46:38 2018

@author: dell

kNN2.py
"""


import pandas as pd
import matplotlib.pyplot as plt


def file_to_matrix(filename):
    data = pd.read_table(filename, sep='\t', names=[
                         'game', 'fly', 'icecream', 'label'])
    label = data.pop('label')

    return data, label


def auto_norm(in_data):
    min_val = in_data.min()
    range_val = in_data.max() - in_data.min()
    norm_data = in_data.apply(lambda x: (x-x.min())/(x.max()-x.min()))
#    norm_data = in_data.apply(lambda x: x-min_val/range_val)

    return norm_data, min_val, range_val


def classify(test, k, data, label):
    data.loc[:, 'k_value'] = data.apply(
        lambda x: (((test-x)**2).sum())**0.5, axis=1)
    classify_data = (pd.DataFrame(
        [data['k_value'], label]).T).sort_values(by='k_value')
    count_data = classify_data.iloc[0:k, :].groupby('label').count()
#   test = (((test-data)**2).sum(axis=1))**0.5

    return(int(count_data['k_value'].idxmax()), count_data)


def dating_class_test():
    ho_rotio = 0.1
    dating_data_mat, dating_labels = file_to_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_val = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_rotio)

    print('num_test_vecs= ', num_test_vecs)
    error_count = 0.0

    for i in range(num_test_vecs):
        classifier_result, class_label = classify(norm_mat.iloc[i, :],
                                                  3,
                                                  norm_mat.iloc[num_test_vecs:m, :],
                                                  dating_labels[num_test_vecs:m])
        print("the classifier com back with: ", classifier_result, "the real answer is: ",
              dating_labels[i])
        if(classifier_result != dating_labels[i]):
            error_count += 1.0
    print("the total error rate is: ", error_count/float(num_test_vecs))
    print(error_count)
    return(classifier_result, class_label)


#data, label = file_to_matrix('datingTestSet2.txt')
#norm_data, min_val, range_val = auto_norm(data)
#test_data = [3000, 6, 0.5]
#data_class, count_data = classify((test_data-min_val)/range_val, 50, norm_data, label)
class_result, class_label = dating_class_test()
