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
#from utils import Articles
import numpy as np

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
	for sub in os.listdir(corpus):
		subdir = os.path.join(corpus, sub)
		for fname in os.listdir(subdir):
			article_id = fname[:-4]
			paragraphs_count = 0
			for line in  open(os.path.join(subdir, fname)):
				docs[doc_id] = line
				index[doc_id] = str(article_id)+'_'+str(paragraphs_count)
				paragraphs_count += 1
				doc_id += 1
	return docs, index

if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("--corpus", default='../output_parser', type=str) # path to corpus
	parser.add_argument("--vectorizer_type", default="tfidf", type=str) # possible values: "tfidf" and "count"
	parser.add_argument("--mx_ngram", default=3, type=int) # the upper bound of the ngram range
	parser.add_argument("--mn_ngram", default=1, type=int) # the lower bound of the ngram range
	parser.add_argument("--stop_words", default=0, type=int) # filtering out English stop-words
	parser.add_argument("--min_count", default=20, type=int) # minimum frequency of the token to be included in the vocabulary
	parser.add_argument("--max_df", default=0.95, type=float) # how much vocabulary percent to keep at max based on frequency
	parser.add_argument("--vec_size", default=100, type=int) # the size of the vector in the semantics space

	args = parser.parse_args()
	corpus = args.corpus
	vectorizer_type = args.vectorizer_type
	mx_ngram = args.mx_ngram
	mn_ngram =  args.mn_ngram
	stop_words = args.stop_words
	if stop_words:
		stop_words = 'arabic' #to be implemented
	else:
		stop_words = None
	n_components = args.vec_size
	min_count = args.min_count
	max_df = args.max_df
	compress = args.compress
	out_dir = args.out_dir

	#articles = Articles(corpus)
	#articles.print_stats()

	docs_count, articles_count = count_docs (corpus)
	articles, index = load_corpus (corpus, docs_count)
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

	bow = vectorizer.fit_transform(articles)

	vocab = vectorizer.get_feature_names()
	print 'size of the vocabulary:', len(vocab)

	svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
	decomposed_bow = svd.fit_transform(bow)
	print 'shape of the SVD decomposed bag_of_words', decomposed_bow.shape
	print 'explained variance ratio sum', svd.explained_variance_ratio_.sum()

	sims = cosine_similarity(decomposed_bow[0].reshape(1,-1), decomposed_bow[1:docs_count])
	print sims[0][:20].reshape(20,1)
