# -*- coding: utf-8 -*-
#
# This file is part of semantic_query.
# Copyright (C) 2016 CIAPPLE.
#
# This is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

# Parameters design and tuning

# Author: Hussein AL-NATSHEH <h.natsheh@ciapple.com>
# Affiliation: CIAPPLE, Jordan

import os, argparse, pickle, json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, MiniBatchDictionaryLearning
from sklearn.metrics.pairwise import cosine_similarity
from utils import Mapper
from collections import OrderedDict
import numpy as np
from itertools import islice
from stop_words import get_stop_words
from bs4 import BeautifulSoup
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
	if w_corpus is not None:
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
	if w_corpus is not None:
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

def query_by_id(transformed, documents, index, query_id=2, n_results=10):

	sims = cosine_similarity(transformed[query_id].reshape(1,-1), transformed)
	scores =  sims[0][:].reshape(-1,1)
	results= dict()
	for i in range(len(transformed)):
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

def query_by_text(transformer, transformed, decomposition_type, documents, index, query_text, n_results=10):
	query_text = BeautifulSoup(query_text, "lxml").p.contents
	print 'query: ', query_text[0]
	if decomposition_type == 'mbdl':
		query = transformer.steps[1][1].transform(transformer.steps[0][1].transform(query_text).toarray())
	else:
		query = transformer.transform(query_text)
	sims = cosine_similarity(query.reshape(1,-1), transformed)
	scores =  sims[0][:].reshape(-1,1)
	results= dict()
	for i in range(len(transformed)):
		results[i] = scores[i]
	sorted_results = OrderedDict(sorted(results.items(), key=lambda k: k[1], reverse=True))
	topn = top(n_results, sorted_results)

	for i, j in topn:
		print 'score:', j, ' Document ID: ', index[i]
		print documents[i]
		print '-----------------------------------------------------------'

if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("--m_corpus", default='../m_output_parser', type=str) # path to m corpus
	parser.add_argument("--w_corpus", default='None', type=str) # path to w corpus
	parser.add_argument("--paragraphs_per_article", default=5, type=int) # max number of paragraphs per article to load from w corpus
	parser.add_argument("--vectorizer_type", default="tfidf", type=str) # possible values: "tfidf" and "count"
	parser.add_argument("--decomposition_type", default="svd", type=str) # possible values: "svd", "mbdl" or "None"
	parser.add_argument("--mx_ngram", default=3, type=int) # the upper bound of the ngram range
	parser.add_argument("--mn_ngram", default=1, type=int) # the lower bound of the ngram range
	parser.add_argument("--stop_words", default=1, type=int) # filtering out English stop-words
	parser.add_argument("--min_count", default=20, type=int) # minimum frequency of the token to be included in the vocabulary
	parser.add_argument("--max_df", default=0.97, type=float) # how much vocabulary percent to keep at max based on frequency
	parser.add_argument("--vec_size", default=30, type=int) # the size of the vector in the semantics space
	parser.add_argument("--query_text", default='query.txt', type=str) # query text 
	parser.add_argument("--query_id", default=2, type=int) # doc_id to use as query
	parser.add_argument("--n_results", default=10, type=int) # number of query results to return
	parser.add_argument("--transformed_file", default=None, type=str) # load dumped transformed vectors (pickle file)
	parser.add_argument("--docs_file", default='documents.pickle', type=str) # documents file
	parser.add_argument("--index_file", default='index.pickle', type=str) # index file
	parser.add_argument("--transformer_file", default='transformer.pickle', type=str) # transformer file
	parser.add_argument("--debug", default=0, type=int) # IPython embed
	args = parser.parse_args()
	m_corpus = args.m_corpus
	w_corpus = args.w_corpus
	if w_corpus == 'None':
		w_corpus = None
	paragraphs_per_article = args.paragraphs_per_article
	vectorizer_type = args.vectorizer_type
	decomposition_type = args.decomposition_type
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
	query_text = args.query_text
	query_id = args.query_id
	n_results = args.n_results
	transformed_file = args.transformed_file
	docs_file = args.docs_file
	index_file = args.index_file
	transformer_file = args.transformer_file
	debug = args.debug

	if transformed_file is None:

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

		if decomposition_type == 'mbdl':
			decomposer = MiniBatchDictionaryLearning(n_components=n_components)
		elif decomposition_type == 'svd':
			decomposer = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
		else:
			decomposer = None

		if decomposer is not None:
			transformer = Pipeline(steps=[('vectorizer',vectorizer), ('decomposer',decomposer)])
		else:
			transformer = vectorizer

		if decomposition_type == 'mbdl':
			transformed = transformer.steps[1][1].fit_transform(transformer.steps[0][1].fit_transform(documents).toarray())
		else:
			transformed = transformer.fit_transform(documents)
		
		vocab = transformer.steps[0][1].get_feature_names()
		print 'size of the vocabulary:', len(vocab)
		print 'shape of the transformed bag_of_words', transformed.shape

		transformed.dump('transformed.pickle')
		pickle.dump(index,open(index_file,'wb'))
		pickle.dump(documents,open(docs_file,'wb'))
		pickle.dump(transformer,open(transformer_file,'wb'))
	
	else:
		transformed = np.load(transformed_file)

		index = pickle.load(open(index_file,'rb'))

		documents = pickle.load(open(docs_file,'rb'))
		print 'number of documents :', len(index)
		transformer = pickle.load(open(transformer_file,'rb'))

	if query_text is None:
		query_by_id(transformed, documents, index, query_id=query_id, n_results=n_results)
	else:
		query_by_text(transformer, transformed, decomposition_type, documents, index, query_text, n_results=10)

	if debug:
		IPython.embed()
