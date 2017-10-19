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

def deptree_update_naive(matrix, vocabulary, parsed, symmetrical=False):
	# deptree_update_naive: generates the context for a word by traversing up the dependency tree and adding all ancestors along the way.	
	# Inputs:
	# - symmetrical: if set to True, the dependency relations are stored symmetrically (i.e. if the head of word A is added to the dependency list of A, A itself is also added to the dependency list of its head)
	if matrix.dtype==np.uint16: peak = 65535
	for ii, token in enumerate(parsed):
		lemma = token.lemma_
		key = lemma
		if key not in vocabulary.keys(): continue
		index = vocabulary[key][2]
		wordindexes = [] # indexes to change in the matrix
		word = token
		while word.dep_ != 'ROOT': # traverse up the dependency tree until you reach the root (the dependency of the root is 'ROOT')
			word = word.head
			wordkey = word.lemma_
			wordindexes.append(vocabulary[wordkey][2])
		for wordindex in wordindexes:
            if matrix[index, wordindex] < peak: matrix[index, wordindex] += 1
			if symmetrical:
				if matrix[wordindex, index] < peak: matrix[wordindex, index] += 1
    return matrix

def deptree_update_headchildren(matrix, vocabulary, parsed):
	# deptree_update_headchildren: generates the context for a word from its head plus its children (after Levy and Goldberg, 2014)
	if matrix.dtype==np.uint16: peak = 65535
	for ii, token in enumerate(parsed):
		lemma = token.lemma_
		key = lemma
		if key not in vocabulary.keys(): continue
		index = vocabulary[key][2]
		wordindexes = [] # indexes to change in the matrix
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
		for wordindex in wordindexes:
            if matrix[index, wordindex] < peak: matrix[index, wordindex] += 1
    return matrix
