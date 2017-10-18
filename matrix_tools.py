# tools for co-occurrence matrix 
import numpy as np

def naive_update(matrix, vocabulary, parsed, window_size=3, ignore_unknown=True):
    # perform the positional co-occurrence counting to update the matrix
    # TODO: allow adaptive window in case of unknown words
    if matrix.dtype=np.uint16: peak = 65535
    for ii, token in enumerate(parsed):
        lemma = token.lemma_
        if token.pos_ == 'PROPN': pos = 'NOUN'
        else: pos=token.pos_
        key = lemma + ' | ' + pos
        if key not in vocabulary.keys(): continue
        index = vocabulary[key][1]
        startpoint = max([0, ii-window_size])
        endpoint = min([len(parsed), ii+window_size+1])
        span=parsed[startpoint:endpoint]
        wordindexes = [] # indexes to change in the matrix
        for jj, word in enumerate(span):
            wordlemma = word.lemma_
            if word.pos_ == 'PROPN': wordpos='NOUN'
            else: wordpos = word.pos_
            wordkey = wordlemma + ' | ' + wordpos
            if wordkey not in vocabulary.keys(): continue
            wordindexes.append(vocabulary[wordkey][1])
        for wordindex in wordindexes:
            #if index>wordindex: continue
            #matrix[index_comatrix(matrix, index, wordindex, return_index=True)] += 1
            if matrix[index,wordindex]< peak: matrix[index, wordindex] += 1
        
    return matrix