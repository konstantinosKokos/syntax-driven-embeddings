import pandas as pd
import csv 
import copy

def load(filepath='/home/jovyan/kokos-playground/Data/TOEFL.csv'):
    '''
    description goes here
    '''
    with open(filepath, 'r') as f:
        r = csv.reader(f)
        data = []
        for row in r: data.append(row)
    tests = pd.DataFrame(data=data, columns=['original','ground_truth','test_1','test_2','test_3'])
    return tests

def reduce(tests, minimum, vocabulary):
    '''
    description goes here
    '''
    reduced = copy.deepcopy(tests)
    to_delete = []
    for ii in range(len(tests)):
        badword = False
        test = tests.iloc[ii]
        for jj in range(0,5):
            if vocabulary[test[jj]][0] < minimum:
                badword = True
                break
        if badword: to_delete.append(ii)
    reduced = reduced.drop(reduced.index[to_delete])
    return reduced

def multifold_test(tests, similarities, vocabulary, split=10):
    '''
    description goes here
    '''
    results = np.zeros([split])
    current_index = 0
    batch_sizes = []
    num_errorss = []
    batch_size = 0
    num_errors = 0.0
    for ii in range(len(tests)):
        if current_index != int(ii/(len(tests)/split)):
            results[current_index] = (batch_size - num_errors)/batch_size
            batch_sizes.append(batch_size)
            num_errorss.append(num_errors)
            current_index = int(ii/(len(tests)/split))
            batch_size = 0
            num_errors = 0.0
        test = tests.iloc[ii]
        original = vocabulary[test[0]][2]
        sims = np.zeros([4])
        for jj in range(0,4):
               sims[jj] = similarities[original, vocabulary[test[jj+1]][2]]
        indices = np.argsort(sims * -1)
        if indices[0] != 0: num_errors +=1
        batch_size += 1
    results[split-1] = (batch_size-num_errors)/batch_size
    return results, batch_sizes, num_errorss
    
    
