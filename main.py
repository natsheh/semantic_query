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

import os, argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from utils import Mapper
from collections import OrderedDict
import numpy as np
from itertools import islice
from stop_words import get_stop_words
import IPython

def count_docs (corpus):
	articles_count = 0
	docs_count = 0
	for sub in os.listdir(corpus):
		subdir = os.path.join(corpus, sub)
		for fname in os.listdir(subdir):
			articles_count += 1
			for line in  open(os.path.join(subdir, fname)):
				docs_count += 1
	return docs_count, articles_count

def load_corpus (corpus, docs_count):
	docs = np.array(range(docs_count), dtype=np.object)
	doc_id = 0
	index = dict()
	articles_dict = dict()
	for sub in os.listdir(corpus):
		subdir = os.path.join(corpus, sub)
		for fname in os.listdir(subdir):
			article_id = fname[:-4]
			paragraphs_count = 0
			for i, line in enumerate(open(os.path.join(subdir, fname))):
				if i == 0:
					articles_dict[fname[:-4]] = line
				docs[doc_id] = line
				index[doc_id] = str(article_id)+'_'+str(paragraphs_count)
				paragraphs_count += 1
				doc_id += 1
	return docs, index

class Mapper():
	def __init__(self, corpus):
		self.corpus = corpus
		self.articles = dict()
		for sub in os.listdir(self.corpus):
			subdir = os.path.join(self.corpus, sub)
			for fname in os.listdir(subdir):
				for i, line in enumerate(open(os.path.join(subdir, fname))):
					if i == 0:
						self.articles[fname[:-4]] = line
						break

	def get_title(self, article_id):
		return self.articles[article_id]

def top(n, sorted_results):
	return list(islice(sorted_results.iteritems(), n))

def query(decomposed_bow, query_id=2, n_results=10):

	sims = cosine_similarity(decomposed_bow[query_id].reshape(1,-1), decomposed_bow)
	scores =  sims[0][:].reshape(-1,1)
	results= dict()
	for i in range(len(decomposed_bow)):
		if i == query_id:
			continue
		results[i] = scores[i]
	sorted_results = OrderedDict(sorted(results.items(), key=lambda k: k[1], reverse=True))
	topn = top(n_results, sorted_results)

	print 'the query document:'
	print articles[query_id]
	for i, j in topn:
		print j, articles[i]
		print '-----------------------------------------------------------'

if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("--corpus", default='../output_parser', type=str) # path to corpus
	parser.add_argument("--vectorizer_type", default="tfidf", type=str) # possible values: "tfidf" and "count"
	parser.add_argument("--mx_ngram", default=3, type=int) # the upper bound of the ngram range
	parser.add_argument("--mn_ngram", default=1, type=int) # the lower bound of the ngram range
	parser.add_argument("--stop_words", default=1, type=int) # filtering out English stop-words
	parser.add_argument("--min_count", default=10, type=int) # minimum frequency of the token to be included in the vocabulary
	parser.add_argument("--max_df", default=0.95, type=float) # how much vocabulary percent to keep at max based on frequency
	parser.add_argument("--vec_size", default=50, type=int) # the size of the vector in the semantics space
	parser.add_argument("--query_id", default=2, type=int) # doc_id to use as query
	parser.add_argument("--n_results", default=10, type=int) # number of query results to return
	parser.add_argument("--decomposed", default=None, type=str) # load dumped decomposed vectors (pickle file)
	parser.add_argument("--debug", default=0, type=int) # IPython embed
	args = parser.parse_args()
	corpus = args.corpus
	vectorizer_type = args.vectorizer_type
	mx_ngram = args.mx_ngram
	mn_ngram =  args.mn_ngram
	stop_words = args.stop_words
	if stop_words:
		stop_words = get_stop_words('ar')
	else:
		stop_words = None
	n_components = args.vec_size
	min_count = args.min_count
	max_df = args.max_df
	query_id = args.query_id
	n_results = args.n_results
	decomposed = args.decomposed
	debug = args.debug

	docs_count, articles_count = count_docs (corpus)
	articles, index = load_corpus (corpus, docs_count)
	print 'number of documents :', docs_count, ' number of articles :',articles_count
	
	if decomposed is None:
		#articles = Articles(corpus)
		#articles.print_stats()

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

		bow = vectorizer.fit_transform(articles)

		vocab = vectorizer.get_feature_names()
		print 'size of the vocabulary:', len(vocab)

		svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
		decomposed_bow = svd.fit_transform(bow)
		print 'shape of the SVD decomposed bag_of_words', decomposed_bow.shape
		print 'explained variance ratio sum', svd.explained_variance_ratio_.sum()
		decomposed_bow.dump('decomposed.pickle')
	
	else:
		decomposed_bow = np.load(decomposed)

	query(decomposed_bow, query_id=query_id, n_results=n_results)

	#mapper = Mapper(corpus)
	#articles_dict = mapper.articles

	if debug:
		IPython.embed()
