# -*- coding: utf-8 -*-
#
# This file is part of semantic_query.
# Copyright (C) 2016 CIAPPLE.
#
# This is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

# Load and index answers from corpus

# Author: Hussein AL-NATSHEH <h.natsheh@ciapple.com>
# Affiliation: CIAPPLE, Jordan

import os, argparse, pickle, json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, MiniBatchDictionaryLearning
import numpy as np
from stop_words import get_stop_words
import IPython

def count_docs (m_corpus, w_corpus, paragraphs_per_article, threshold=500):
	articles_count = 0
	docs_count = 0
	for sub in os.listdir(m_corpus):
		subdir = os.path.join(m_corpus, sub)
		for fname in os.listdir(subdir):
			articles_count += 1
			doc = ''
			for i, line in enumerate(open(os.path.join(subdir, fname))):
				if len(doc) > threshold:
					doc = ''
					docs_count += 1
				if i == 0:
					title = str(line)
				if i > 0:
					doc = str(line) + 'لمعرفة المزيد يرجى الإستماع أو قراءة كافة المقال بعنوان__' + title

	if w_corpus is not None:
		for sub in os.listdir(w_corpus):
			subdir = os.path.join(w_corpus, sub)
			for fname in os.listdir(subdir):
				articles_count += 1
				doc = ''
				for i, line in enumerate(open(os.path.join(subdir, fname))):
					if len(doc) > threshold:
						doc = ''
						docs_count += 1
					if i == 0:
						title = str(line)
					if i > 0:
						doc = str(line) + 'لمعرفة المزيد يرجى الإستماع أو قراءة كافة المقال بعنوان__' + title
					if i == paragraphs_per_article:
							break
					if i > 1:
	 					doc = line1+' '+str(line)
	return docs_count, articles_count

def load_corpus (m_corpus, w_corpus, docs_count, paragraphs_per_article, threshold=500):
	docs = np.array(range(docs_count), dtype=np.object)
	doc_id = 0
	index = dict()
	for sub in os.listdir(m_corpus):
		subdir = os.path.join(m_corpus, sub)
		for fname in os.listdir(subdir):
			article_id = 'm'+'_'+str(fname[:-4])
			doc = ''
			for i, line in enumerate(open(os.path.join(subdir, fname))):
				if len(doc) > threshold:
					docs[doc_id] = unicode(doc, 'utf8')
					doc = ''
					index[doc_id] = str(article_id)+'_'+str(i)
					doc_id += 1
				if i == 0:
					title = str(line)
				if i > 0:
					doc = str(line) + 'لمعرفة المزيد يرجى الإستماع أو قراءة كافة المقال بعنوان__' + title

	if w_corpus is not None:
		for sub in os.listdir(w_corpus):
			subdir = os.path.join(w_corpus, sub)
			for fname in os.listdir(subdir):
				article_id = 'w'+'_'+str(fname[:-4])
				for i, line in enumerate(open(os.path.join(subdir, fname))):
					if len(doc) > threshold:
						docs[doc_id] = unicode(doc, 'utf8')
						doc = ''
						index[doc_id] = str(article_id)+'_'+str(i)
						doc_id += 1
					if i == 0:
						title = str(line)
					if i == paragraphs_per_article:
						break
					if i > 0:
						doc = str(line) + 'لمعرفة المزيد يرجى الإستماع أو قراءة كافة المقال بعنوان__' + title
	return docs, index

if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("--m_corpus", default='../new_output_parser', type=str) # path to m corpus
	parser.add_argument("--w_corpus", default='None', type=str) # path to w corpus
	parser.add_argument("--paragraphs_per_article", default=4, type=int) # max number of paragraphs per article to load from w corpus
	parser.add_argument("--vectorizer_type", default="tfidf", type=str) # possible values: "tfidf" and "count"
	parser.add_argument("--decomposition_type", default="svd", type=str) # possible values: "svd", "mbdl" or "None"
	parser.add_argument("--mx_ngram", default=2, type=int) # the upper bound of the ngram range
	parser.add_argument("--mn_ngram", default=1, type=int) # the lower bound of the ngram range
	parser.add_argument("--stop_words", default=1, type=int) # filtering out English stop-words
	parser.add_argument("--min_count", default=5, type=int) # minimum frequency of the token to be included in the vocabulary
	parser.add_argument("--max_df", default=0.98, type=float) # how much vocabulary percent to keep at max based on frequency
	parser.add_argument("--vec_size", default=300, type=int) # the size of the vector in the semantics space
	parser.add_argument("--transformed_file", default='transformed.pickle', type=str) # load dumped transformed vectors (pickle file)
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
	transformed_file = args.transformed_file
	docs_file = args.docs_file
	index_file = args.index_file
	transformer_file = args.transformer_file
	debug = args.debug


	docs_count, articles_count = count_docs (m_corpus, w_corpus, paragraphs_per_article)
	documents, index = load_corpus (m_corpus, w_corpus, docs_count, paragraphs_per_article)
	print 'number of documents :', docs_count, ' number of articles :',articles_count

	if vectorizer_type == "count":
		vectorizer = CountVectorizer(input='content',
			         analyzer='word', stop_words=stop_words, min_df=min_count, 
			         ngram_range=(mn_ngram, mx_ngram), max_df=max_df)
	elif vectorizer_type == "tfidf":
		vectorizer = TfidfVectorizer(input='content', decode_error='ignore',
			         analyzer='word', stop_words=stop_words, min_df=min_count, 
			         ngram_range=(mn_ngram, mx_ngram), max_df=max_df, lowercase=False)
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

	transformed.dump(transformed_file)
	pickle.dump(index,open(index_file,'wb'))
	pickle.dump(documents,open(docs_file,'wb'))
	pickle.dump(transformer,open(transformer_file,'wb'))

	if debug:
		IPython.embed()
