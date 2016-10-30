# -*- coding: utf-8 -*-
#
# This file is part of semantic_query.
# Copyright (C) 2016 CIAPPLE.
#
# This is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

# Main example

# Author: Hussein AL-NATSHEH <h.natsheh@ciapple.com>
# Affiliation: CIAPPLE, Jordan

import os, argparse, pickle, json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from utils import Mapper
from collections import OrderedDict
import numpy as np
from itertools import islice
from stop_words import get_stop_words
import IPython

def count_docs (m_corpus, w_corpus, paragraphs_per_article):
	articles_count = 0
	docs_count = 0
	for sub in os.listdir(m_corpus):
		subdir = os.path.join(m_corpus, sub)
		for fname in os.listdir(subdir):
			articles_count += 1
			for i, line in enumerate(open(os.path.join(subdir, fname))):
				if i == 0:
					continue
				docs_count += 1
	for sub in os.listdir(w_corpus):
		subdir = os.path.join(w_corpus, sub)
		for fname in os.listdir(subdir):
			paragraphs_count = 0
			articles_count += 1
			for i, line in enumerate(open(os.path.join(subdir, fname))):
				if i == 0:
					continue
				docs_count += 1
				if i == paragraphs_per_article:
					break
	return docs_count, articles_count

def load_corpus (m_corpus, w_corpus, docs_count, paragraphs_per_article):
	docs = np.array(range(docs_count), dtype=np.object)
	doc_id = 0
	index = dict()
	for sub in os.listdir(m_corpus):
		subdir = os.path.join(m_corpus, sub)
		for fname in os.listdir(subdir):
			article_id = 'm'+'_'+str(fname[:-4])
			paragraphs_count = 0
			for i, line in enumerate(open(os.path.join(subdir, fname))):
				if i == 0:
					title = line
					continue
				docs[doc_id] = str(title)+'__'+str(line)
				index[doc_id] = str(article_id)+'_'+str(paragraphs_count)
				paragraphs_count += 1
				doc_id += 1
	for sub in os.listdir(w_corpus):
		subdir = os.path.join(w_corpus, sub)
		for fname in os.listdir(subdir):
			article_id = 'w'+'_'+str(fname[:-4])
			paragraphs_count = 0
			for i, line in enumerate(open(os.path.join(subdir, fname))):
				if i == 0:
					title = line
					continue
				docs[doc_id] = str(title)+'__'+str(line)
				index[doc_id] = str(article_id)+'_'+str(paragraphs_count)
				paragraphs_count += 1
				doc_id += 1
				if i == paragraphs_per_article:
					break
	return docs, index

class Mapper():
	def __init__(self, corpus):
		self.corpus = corpus
		self.documents = dict()
		for sub in os.listdir(self.corpus):
			subdir = os.path.join(self.corpus, sub)
			for fname in os.listdir(subdir):
				for i, line in enumerate(open(os.path.join(subdir, fname))):
					if i == 0:
						self.documents[fname[:-4]] = line
						break

	def get_title(self, article_id):
		return self.documents[article_id]

def top(n, sorted_results):
	return list(islice(sorted_results.iteritems(), n))

def query(decomposed_bow, documents, index, query_id=2, n_results=10):

	sims = cosine_similarity(decomposed_bow[query_id].reshape(1,-1), decomposed_bow)
	scores =  sims[0][:].reshape(-1,1)
	results= dict()
	for i in range(len(decomposed_bow)):
		if i == query_id:
			continue
		results[i] = scores[i]
	sorted_results = OrderedDict(sorted(results.items(), key=lambda k: k[1], reverse=True))
	topn = top(n_results, sorted_results)

	print 'the query document:', index[query_id]
	print documents[query_id]

	for i, j in topn:
		print 'score:', j, ' Document ID: ', index[i]
		print documents[i]
		print '-----------------------------------------------------------'

if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("--m_corpus", default='../m_output_parser', type=str) # path to m corpus
	parser.add_argument("--w_corpus", default='../w_output_parser', type=str) # path to w corpus
	parser.add_argument("--paragraphs_per_article", default=5, type=int) # max number of paragraphs per article to load from w corpus
	parser.add_argument("--vectorizer_type", default="tfidf", type=str) # possible values: "tfidf" and "count"
	parser.add_argument("--mx_ngram", default=3, type=int) # the upper bound of the ngram range
	parser.add_argument("--mn_ngram", default=1, type=int) # the lower bound of the ngram range
	parser.add_argument("--stop_words", default=1, type=int) # filtering out English stop-words
	parser.add_argument("--min_count", default=20, type=int) # minimum frequency of the token to be included in the vocabulary
	parser.add_argument("--max_df", default=0.95, type=float) # how much vocabulary percent to keep at max based on frequency
	parser.add_argument("--vec_size", default=50, type=int) # the size of the vector in the semantics space
	parser.add_argument("--query_id", default=2, type=int) # doc_id to use as query
	parser.add_argument("--n_results", default=10, type=int) # number of query results to return
	parser.add_argument("--decomposed_file", default=None, type=str) # load dumped decomposed vectors (pickle file)
	parser.add_argument("--docs_file", default='documents.pickle', type=str) # load dumped decomposed vectors (pickle file)
	parser.add_argument("--index_file", default='index.json', type=str) # load dumped decomposed vectors (pickle file)
	parser.add_argument("--debug", default=0, type=int) # IPython embed
	args = parser.parse_args()
	m_corpus = args.m_corpus
	w_corpus = args.w_corpus
	paragraphs_per_article = args.paragraphs_per_article
	vectorizer_type = args.vectorizer_type
	mx_ngram = args.mx_ngram
	mn_ngram =  args.mn_ngram
	stop_words = args.stop_words
	if stop_words:
		stop_words = get_stop_words('ar')
	else:
		stop_words = None
	min_count = args.min_count
	max_df = args.max_df
	n_components = args.vec_size
	query_id = args.query_id
	n_results = args.n_results
	decomposed_file = args.decomposed_file
	docs_file = args.docs_file
	index_file = args.index_file
	debug = args.debug

	if decomposed_file is None:

		docs_count, articles_count = count_docs (m_corpus, w_corpus, paragraphs_per_article)
		documents, index = load_corpus (m_corpus, w_corpus, docs_count, paragraphs_per_article)
		print 'number of documents :', docs_count, ' number of articles :',articles_count

		if vectorizer_type == "count":
			vectorizer = CountVectorizer(input='content',
				         analyzer='word', stop_words=stop_words, min_df=min_count, 
				         ngram_range=(mn_ngram, mx_ngram), max_df=max_df)
		elif vectorizer_type == "tfidf":
			vectorizer = TfidfVectorizer(input='content',
				         analyzer='word', stop_words=stop_words, min_df=min_count, 
				         ngram_range=(mn_ngram, mx_ngram), max_df=max_df)
		else:
			raise NameError('Please check your vectorizer option. It must be either "tfidf" or "count"')

		bow = vectorizer.fit_transform(documents)

		vocab = vectorizer.get_feature_names()
		print 'size of the vocabulary:', len(vocab)

		svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
		decomposed_bow = svd.fit_transform(bow)
		print 'shape of the SVD decomposed bag_of_words', decomposed_bow.shape
		print 'explained variance ratio sum', svd.explained_variance_ratio_.sum()

		decomposed_bow.dump('decomposed.pickle')
		json.dump(index,open(index_file,'wb'))
		pickle.dump(documents,open(docs_file,'wb'))
	
	else:
		decomposed_bow = np.load(decomposed_file)

		if index_file[-4:] =='json':
			index = json.load(open(index_file,'rb'))
		else:
			index = pickle.load(open(index_file,'rb'))

		documents = pickle.load(open(docs_file,'rb'))
		print 'number of documents :', len(index)

	query(decomposed_bow, documents, index, query_id=query_id, n_results=n_results)

	if debug:
		IPython.embed()
