# -*- coding: utf-8 -*-
#
# This file is part of semantic_query.
# Copyright (C) 2016 CIAPPLE.
#
# This is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

# Semantic Query API

# Author: Hussein AL-NATSHEH <h.natsheh@ciapple.com>
# Affiliation: CIAPPLE, Jordan

import os, argparse, pickle, json
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, MiniBatchDictionaryLearning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

from collections import OrderedDict
from itertools import islice
from stop_words import get_stop_words
from bs4 import BeautifulSoup

from flask import Flask, make_response, make_response, request, current_app
from flask_httpauth import HTTPBasicAuth
from flask_restful import Resource, Api, reqparse

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

	results = dict()
	results[0] = {'query': documents[query_id], 'query_doc_id': index[query_id]}
	for rank, (doc, score) in enumerate(topn):
		results[rank+1] = {'score': str(score), 'doc_id':  str(index[doc]), 'document': documents[doc]}

	return results

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

	results = dict()
	results[0] = {'query': query_text[0]}
	for rank, (doc, score) in enumerate(topn):
		title = documents[doc].split('__')[0]
		results[rank+1] = {'score': str(score), 'doc_id':  str(index[doc]), 'title': title, 'document': documents[doc]}

	return results

class Query(Resource):
	def get(self):
		try:
			q = request.args.get('q')
			response = {}
			if q is None:
				response['results'] = query_by_id(transformed, documents, index, query_id=query_id, n_results=n_results)
			else:
				response['results'] =  query_by_text(transformer, transformed, decomposition_type, documents, index, q, n_results=10)
			return response

		except Exception as e:
			return {'error': str(e)}

app = Flask(__name__, static_url_path="")
auth = HTTPBasicAuth()

api = Api(app)
api.add_resource(Query, '/Query/')

if __name__ == '__main__':
	m_corpus = '../m_output_parser'
	w_corpus = None
	paragraphs_per_article = 5
	vectorizer_type = 'tfidf'
	decomposition_type = 'svd'
	mx_ngram = 3
	mn_ngram =  1
	stop_words = get_stop_words('ar')
	min_count = 5
	max_df = 0.98
	n_components = 25
	query_id = 2
	n_results = 5
	transformed_file = 'transformed.pickle'
	docs_file = 'trigram_svd25_documents.pickle'
	index_file = 'trigram_svd25_index.pickle'
	transformer_file = 'trigram_svd25_transformer.pickle'

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
	print 'Ready to call!!'
	app.run(debug=True, threaded=True)
