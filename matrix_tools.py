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

def naive_update(matrix, vocabulary, parsed, window_size=3):
	# perform the positional co-occurrence counting to update the matrix
	peak = np.inf
	if matrix.dtype==np.uint16: peak = 65535
	for ii, token in enumerate(parsed):
		key = token.lemma_
		if key not in vocabulary.keys(): continue
		index = vocabulary[key][2]
		startpoint = max([0, ii-window_size])
		endpoint = min([len(parsed), ii+window_size+1])
		span=parsed[startpoint:endpoint]
		wordindexes = [] # indexes to change in the matrix
		for jj, word in enumerate(span):
			wordkey = word.lemma_
			if wordkey not in vocabulary.keys(): continue
			wordindexes.append(vocabulary[wordkey][2])
		for wordindex in wordindexes:
			#if index>wordindex: continue
			#matrix[index_comatrix(matrix, index, wordindex, return_index=True)] += 1
			if matrix[index,wordindex]< peak: matrix[index, wordindex] += 1
		
	return matrix

def syntactic_update(matrix, vocabulary, parsed, update_method, symmetrical=False, debug=False):
	'''
	Updates matrix by adding the co-occurrences within the given parsed sentence.
	Inputs:
	- matrix: a co-occurrence matrix of size (len(vocabulary), len(vocabulary))
	- vocabulary: the vocabulary
	- parsed: a SpaCy-parsed sentence
	- update_method: the method that should be used to count the co-occurrences
	- symmetrical: if set to True, the dependency relations are stored symmetrically (i.e. if the co-occurrence count of word A with word B is incremented, then also the co-occurrence count of word B with word A will be incremented)
	- debug: if set to True, the co-occurrenced will be printed instead of stored in the matrix
	Outputs:
	- the updated matrix
	'''
	peak = np.inf
	if matrix.dtype==np.uint16: peak = 65535
	if debug: contexts = {w.lemma_: [] for w in parsed if w.lemma_ in vocabulary}
	else: contexts = {vocabulary[w.lemma_][2]: [] for w in parsed if w.lemma_ in vocabulary}
	for ii, token in enumerate(parsed):
		key = token.lemma_
		if key not in vocabulary.keys(): continue
		index = vocabulary[key][2]
		wordindexes = update_method(vocabulary, token, debug) # indexes to change in the matrix
		if debug: contexts[key].extend(wordindexes)
		else: contexts[index].extend(wordindexes)
		if symmetrical:
			for wordindex in wordindexes:
				if debug: contexts[wordindex].append(key)
				else: contexts[wordindex].append(index)	
	
	for index, context in contexts.items():
		context = set(context)
		if debug: print(index, 'has context', context)
		else:
			for wordindex in context:
				if matrix[index, wordindex] < peak: matrix[index, wordindex] += 1
	return matrix

def deptree_naive(vocabulary, token, debug=False):
	'''
	Generates the context for a token by traversing up the dependency tree and adding all ancestors along the way.	
	Inputs:
	- vocabulary: the vocabulary
	- token: the word for which the co-occurrences must be counted
	- debug: if set to True, a list of lemmas will be output; else a list of indices will be output
	Outputs:
	- wordindexes: a list of matrix indices of all the words that belong to the context of token (if debug is set to True this will be a list of lemmas)
	'''
	wordindexes = []
	while token.dep_ != 'ROOT': # traverse up the dependency tree until you reach the root (the dependency of the root is 'ROOT')
		token = token.head
		addWord(token, wordindexes, vocabulary, debug)
	return wordindexes

def deptree_headchildren(vocabulary, token, debug=False):
	'''
	Generates the context for a token from its head plus its children (after Levy and Goldberg, 2014).
	Inputs:
	- vocabulary: the vocabulary
	- token: the word for which the co-occurrences must be counted
	- debug: if set to True, a list of lemmas will be output; else a list of indices will be output
	Outputs:
	- wordindexes: a list of matrix indices of all the words that belong to the context of token (if debug is set to True this will be a list of lemmas)
	'''
	wordindexes = []
	if token.dep_ != 'prep':
		for child in token.children:
			if child.dep_ == 'prep': # if token forms the head of a preposition, we will directly connect it to the object of the preposition, skipping the preposition itself
				grandchildren = list(child.children)
				if len(grandchildren) > 0:
					child = grandchildren[0]
			addWord(child, wordindexes, vocabulary, debug)
			conjunct = next((w for w in child.children if w.dep_ == 'conj'), None) # Add potential conjuncts of child to the context as well
			while conjunct != None:
				addWord(conjunct, wordindexes, vocabulary, debug)
				conjunct = next((w for w in conjunct.children if w.dep_ == 'conj'), None)
		currentword = token
		if currentword.dep_ == 'pobj': # if token forms the object of a preposition, we will directly connect it to the head of the preposition, skipping the preposition itself
			currentword = currentword.head
		while currentword.head.dep_ == 'conj':
			addWord(currentword.head, wordindexes, vocabulary, debug)
			currentword = currentword.head
		if currentword.dep_ != 'ROOT':
			addWord(currentword.head, wordindexes, vocabulary, debug)
	return wordindexes
	
def deptree_noun_chunks(vocabulary, token, debug=False):
	'''
	Generates the context for a token based on noun chunks. If word W belongs to a noun chunk, all the other words within that nounchunk plus the syntactic neighbours of the noun chunk are added to the context of W.
	Inputs:
	- vocabulary: the vocabulary
	- token: the word for which the co-occurrences must be counted
	- debug: if set to True, a list of lemmas will be output; else a list of indices will be output
	Outputs:
	- wordindexes: a list of matrix indices of all the words that belong to the context of token (if debug is set to True this will be a list of lemmas)
	'''
	parsed = token.doc
	noun_chunks = parsed.noun_chunks
	wordindexes = []
	for nc in noun_chunks:
		if token in nc:
			root = nc.root
			for child in root.children:
				if child not in nc:
					if child.dep_ == 'prep': # if token forms the head of a preposition, we will directly connect it to the object(s) of the preposition, skipping the preposition itself
						grandchildren = list(child.children)
						if len(grandchildren) > 0:
							child = grandchildren[0]
					addWord(child, wordindexes, vocabulary, debug)
					conjunct = next((w for w in child.children if w.dep_ == 'conj'), None) # Add potential conjuncts of child to the context as well
					while conjunct != None:
						addWord(conjunct, wordindexes, vocabulary, debug)
						conjunct = next((w for w in conjunct.children if w.dep_ == 'conj'), None)
			currentword = root
			if currentword.dep_ == 'pobj': # if token forms the object of a preposition, we will directly connect it to the head of the preposition, skipping the preposition itself
				currentword = currentword.head
			while currentword.head.dep_ == 'conj': # Add potential conjuncts of the head of the noun chunk to the context as well
				addWord(currentword.head, wordindexes, vocabulary, debug)
				currentword = currentword.head
			addWord(currentword.head, wordindexes, vocabulary, debug)
			for word in nc: # Add all other words in the noun chunk to the context
				if word == token: continue
				addWord(word, wordindexes, vocabulary, debug)
	return wordindexes

def addWord(word, wordindexes, vocabulary, debug=False):
	'''
	Adds a word to a list if the word occurs in the vocabulary.
	Inputs:
	- word: SpaCy Token object of the word that needs to be added to the list
	- wordindexes: the list that the word needs to be added to
	- vocabulary: the vocabulary
	- debug: if set to True, the lemma of the word will be added to the list; else the index will be added
	'''
	key = word.lemma_
	if key in vocabulary.keys():
		if debug: wordindexes.append(key)
		else: wordindexes.append(vocabulary[key][2])
