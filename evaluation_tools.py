import pandas as pd
import csv 
import copy
from matrix_tools import pair_similarity
import numpy as np
from tqdm import tqdm_notebook

def construct_toefl(filepath='/home/jovyan/kokos-playground/Data/TOEFL.csv'):
    '''
    description goes here
    '''
    with open(filepath, 'r') as f:
        r = csv.reader(f)
        data = []
        for row in r: data.append(row)
    tests = pd.DataFrame(data=data, columns=['original','ground_truth','test_1','test_2','test_3'])
    return tests

def reduce_toefl(tests, minimum, vocabulary):
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
    
def construct_bless(pathfile, vocabulary, similarities):
    '''
    description goes here
    '''
    with open(pathfile, 'r') as f:
        z = csv.reader(f, delimiter='\t')
        data = []
        for row in z: 
            newrow = []
            newrow.append(row[0].split('-')[0])
            newrow.append(row[2])
            newrow.append(row[3].split('-')[0])
            newrow.append(0)
            data.append(newrow)
    bless = pd.DataFrame(data=data, columns=['root','category','word', 'score'])
    bless = reduce_bless(bless, vocabulary)
    bless['score'] = bless['score'].astype(float)
    for ii in tqdm_notebook(range(len(bless))):
        bless.set_value(ii, 'score', pair_similarity(bless.iloc[ii]['root'], bless.iloc[ii]['word'], vocabulary, similarities))
    return bless

def reduce_bless(bless, vocabulary):
    '''
    description goes here
    '''
    to_delete = []
    for ii in range(len(bless)):
        if bless.iloc[ii]['root'] not in vocabulary.keys() or vocabulary[bless.iloc[ii]['root']][0]< 40 or bless.iloc[ii]['word'] not in vocabulary.keys(): to_delete.append(ii)
    bless = bless.drop(bless.index[to_delete])
    bless.index = [i for i in range(len(bless))]
    return bless

def summarize(bless):
    '''
    description goes here
    '''
    words = {}
    attributes = []
    coords = []
    events = []
    hypers = []
    meros = []
    randomjs = []
    randomvs = []
    randomns = []
    currentword = bless.iloc[0]['root']
    for ii in range(len(bless)):
        if currentword != bless.iloc[ii]['root']: 
            words[currentword] = [attributes, coords, events, meros, hypers, randomjs, randomvs, randomns]
            attributes = []
            coords = []
            events = []
            hypers = []
            meros = []
            randomjs = []
            randomvs = []
            randomns = []
            currentword = bless.iloc[ii]['root']
        if bless.iloc[ii]['category'] == 'mero': meros.append(bless.iloc[ii]['score'])
        if bless.iloc[ii]['category'] == 'hyper': hypers.append(bless.iloc[ii]['score'])
        if bless.iloc[ii]['category'] == 'attri': attributes.append(bless.iloc[ii]['score'])
        if bless.iloc[ii]['category'] == 'coord': coords.append(bless.iloc[ii]['score'])
        if bless.iloc[ii]['category'] == 'event': events.append(bless.iloc[ii]['score'])  
        if bless.iloc[ii]['category'] == 'random-j': randomjs.append(bless.iloc[ii]['score'])
        if bless.iloc[ii]['category'] == 'random-v': randomvs.append(bless.iloc[ii]['score'])
        if bless.iloc[ii]['category'] == 'random-n': randomns.append(bless.iloc[ii]['score'])
    words[currentword] = [attributes, coords, events, meros, hypers, randomjs, randomvs, randomns]

    '''for word in words:
        centered_vectors = []
        for vector in words[word]:
            centered_vectors.append(center_and_mean(vector))
        words[word] = centered_vectors
    '''
    
    for word in words:
        vector = []
        for ii in range(len(words[word])):
            newvalue = max(words[word][ii])
            vector.append(newvalue)
        words[word] = vector
    '''
    attributes = [max(words[word][0]) for word in words]
    coords = [max(words[word][1]) for word in words]
    events = [max(words[word][2]) for word in words]
    meros = [max(words[word][3]) for word in words]
    hypers = [max(words[word][4]) for word in words]
    randomjs = [max(words[word][5]) for word in words]
    randomvs = [max(words[word][6]) for word in words]
    randomns = [max(words[word][7]) for word in words]
    
    for ii, word in enumerate(words):
        vector = [attributes[ii], coords[ii], events[ii], meros[ii], hypers[ii], randomjs[ii], randomvs[ii], randomns[ii]]
        vector = center_and_mean(vector)
        words[word] = vector
    '''
    
    return words#, attributes, coords, events, meros, hypers, randoms
            
def center_and_mean(vector):
    vector = vector - np.mean(vector)
    return vector/ np.std(vector, ddof=1)