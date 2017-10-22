# tools for co-occurrence matrix 
import numpy as np
from scipy.spatial.distance import cosine
from vocabulary_tools import idx_to_words

def construct_similarities(comatrix):
    '''
    iterate over a matrix to produce word-pair similarities
    '''
    similarities = np.zeros((comatrix.shape[0], comatrix.shape[0]))
    for ii in range(comatrix.shape[0]):
        for jj in range(ii, comatrix.shape[0]): # diagonal symmetry - fill only the upper right triangle
            similarities[ii, jj] = 1 - cosine(comatrix[ii,:], comatrix[jj,:])
            similarities[jj, ii] = similarities[ii, jj]
    return similarities

def most_similar(word, vocabulary, reverse_vocabulary, similarities, return_similarities=False):
    ii = vocabulary[word][2]
    vector = similarities[ii, :]
    idx = np.argsort(vector * -1)
    if return_similarities: return idx_to_words(idx, reverse_vocabulary)
    return idx_to_words(idx, reverse_vocabulary)

def naive_update(matrix, vocabulary, parsed, window_size=3, ignore_unknown=True):
    # perform the positional co-occurrence counting to update the matrix
    # TODO: allow adaptive window in case of unknown words
    if matrix.dtype==np.uint16: peak = 65535
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

def syntactic_update(matrix, vocabulary, parsed, update_method, symmetrical=False):
	if matrix.dtype==np.uint16: peak = 65535
	for ii, token in enumerate(parsed):
		lemma = token.lemma_
		key = lemma
		if key not in vocabulary.keys(): continue
		index = vocabulary[key][2]
		wordindexes = update_method(vocabulary, token) # indexes to change in the matrix
		for wordindex in wordindexes:
            if matrix[index, wordindex] < peak: matrix[index, wordindex] += 1
			if symmetrical:
				if matrix[wordindex, index] < peak: matrix[wordindex, index] += 1
    return matrix

def deptree_naive(vocabulary, word):
	# deptree_naive: generates the context for a word by traversing up the dependency tree and adding all ancestors along the way.	
	# Inputs:
	# - symmetrical: if set to True, the dependency relations are stored symmetrically (i.e. if the head of word A is added to the dependency list of A, A itself is also added to the dependency list of its head)
	wordindexes = []
	while word.dep_ != 'ROOT': # traverse up the dependency tree until you reach the root (the dependency of the root is 'ROOT')
		word = word.head
		wordkey = word.lemma_
		wordindexes.append(vocabulary[wordkey][2])
    return wordindexes

def deptree_headchildren(vocabulary, token):
	# deptree_headchildren: generates the context for a word from its head plus its children (after Levy and Goldberg, 2014)
	wordindexes = []
	if token.dep_ != 'prep':
		for child in token.children:
			dep_child = child.dep_
			if dep_child == 'prep':
				child = list(child.children)[0]
				dep_child = 'prep_with'
			childkey = child.lemma_
			wordindexes.append(vocabulary[childkey][2])
		head = token.head
		dep = token.dep_
		if dep == 'pobj':
			head = head.head
			dep = 'prep_with'
		if dep != 'ROOT':
			headkey = head.lemma_
			wordindexes.append(vocabulary[headkey][2])
    return wordindexes