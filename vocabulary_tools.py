import pickle
import collections
import tqdm
import numpy as np

def vocabularize(listfile, nlp, pos_decorated=True, lemmatized=True, return_history=False):
    # vocabularize: iterate over some sample texts to produce the vocabulary and the word frequence statistics
    # Inputs:
    # - listfile: file containing the directory paths to the sample files
    # - nlp: the spacy module to process the samples
    # Outputs:
    # - vocabulary: a dictionary mapping tokens to a [1x2] numpy containing [token frequency, token-document frequency] 
    # - history: a list of the vocabulary size at each iteration
    # Options:
    # - pos_decorated: whether to attach part of speech tags to the vocabulary tokens
    # - lemmatized: whether to convert words to their lemmas
    # - return_history: whether to keep a vocabulary size history 
    vocabulary = collections.OrderedDict()
    if return_history: history = []
    for ii,filepath in tqdm.tqdm_notebook(enumerate(listfile)):
        with open(filepath[0:-1]) as file:
            sample = file.read()
            local_vocabulary = []
            for line in sample.splitlines():
                for sentence in line.split('.'):
                    if len(sentence.split()) < 4: continue 
                    parsed = nlp(sentence)
                    for token in parsed:
                        if lemmatized: lemma = token.lemma_
                        else: lemma=token
                        if pos_decorated:
                            if token.pos_ == 'PROPN': pos = 'NOUN'
                            else: pos = token.pos_
                            key = lemma + ' | ' + pos
                        else: key = lemma
                        if key not in local_vocabulary: local_vocabulary.append(key)
                        if key not in vocabulary.keys(): vocabulary[key] = np.array([1,0,0],dtype='int') # TF / TDF / ID
                        else: vocabulary[key][0] += 1 # term frequency
            for key in local_vocabulary: vocabulary[key][1] += 1 # document frequency 
            if return_history: history.append(len(vocabulary)) # for statistics
    if return_history: return vocabulary, history
    return vocabulary

def keys_to_lemmata(keys):
    if keys.__class__==str: 
        lemmata = keys.split(' | ')[0]
        return lemmata
    elif keys.__class__==list:
            lemmata = []
            for key in keys:
                lemmata.append(key(' | ')[0])
            return lemmata

def cut_stopwords(vocabulary, stopwords, verbose=True):
    # NEEDS FIXING
    for_deletion = []
    length_0 = float(len(vocabulary))
    print('Cutting stoplist')
    print('Initial size: ', len(vocabulary))
    for key in tqdm.tqdm_notebook(vocabulary):
        lemma = key.split(' | ')[0]
        if lemma in stopwords:
            for_deletion.append(key)
    for key in for_deletion: del vocabulary[key]
    print('Final size: ', len(vocabulary))
    print('Compression(%): ', (length_0 - len(vocabulary))/length_0)
    return vocabulary

def key_to_lemma(key):
    lemma = []

def cut_frequency(vocabulary, threshold, position=0, no_delete=True, verbose=True):
    if verbose: print('Deleting entries with position', position, 'value less than', threshold)
    length_0 = float(len(vocabulary))
    reduction = 0
    if verbose: print('Initial size: ', len(vocabulary))
    for_deletion = []
    for key in vocabulary:
        if vocabulary[key][position] < threshold: for_deletion.append(key)
    for key in for_deletion: 
        reduction +=1
        if not no_delete: del vocabulary[key]
    if verbose: 
        print('Final size: ', length_0-reduction)
        print('Compression(#): ', reduction)
        print('Compression(%): ', (reduction/length_0))
    return vocabulary, for_deletion

def cut_top_edge(vocabulary, threshold=0.35, position=0, no_delete=False, verbose=True):
    # Cut all tokens that appear at least as often as (threshold* the occurences of the top word) (might be bad to use this)
    for_deletion = []
    if verbose: print('Deleting all tokens that appear at least', threshold, 'times as the most common token')
    length_0 = float(len(vocabulary))
    reduction = 0
    if verbose: print('Initial size: ', len(vocabulary))
    if threshold > 1: threshold = 1./threshold
    for ii, key in enumerate(vocabulary):
        if ii==0: 
            for_deletion.append(key)
            comparison = vocabulary[key][position]
            if verbose: print('Most common word:', key, 'with', comparison, 'occurrences')
        else:
            if vocabulary[key][position] < threshold * comparison: 
                if verbose: print('Last token deleted:', for_deletion[-1], 'with', vocabulary[for_deletion[-1]][position], 'occurences')
                break
            for_deletion.append(key)
    for key in for_deletion: 
        reduction += 1
        if not no_delete: del vocabulary[key]
    if verbose:
        print('Final size: ', length_0-reduction)
        print('Compression(#): ', reduction)
        print('Compression(%): ', (reduction/length_0))
    return vocabulary, for_deletion

def indexize(vocabulary, index=0):
    # this function adds an index value to the vocabulary to allow for int-> key mapping
    # first sort by occurrence to improve performance
    vocabulary = collections.OrderedDict(sorted(vocabulary.items(), key=lambda x: x[1][index], reverse=True))
    for ii, key in enumerate(vocabulary):
        temp = vocabulary[key]
        temp[2] = ii
        vocabulary[key] = temp
    return vocabulary

        
    

