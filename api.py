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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

from collections import OrderedDict
from itertools import islice
from bs4 import BeautifulSoup

from flask import Flask, request
from flask_httpauth import HTTPBasicAuth
from flask_restful import Resource, Api

def top(n, sorted_results):
	return list(islice(sorted_results.iteritems(), n))

def query_by_text(transformer, transformed, documents, index, query_text, n_results=10):
	query_text = BeautifulSoup(query_text, "lxml").p.contents
	print 'query: ', query_text[0]
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
		title = documents[doc].split('\n__')[0]
		results[rank+1] = {'score': str(score), 'doc_id':  str(index[doc]), 'title': title, 'document': documents[doc]}

	return results

class Query(Resource):
	def get(self):
		try:
			q = request.args.get('q')
			try:
				size = request.args.get('size')
				n_results = int(size)
				if n_results > 100:
					n_results = 100
			except:
				n_results = 10
			response = {}
			response['results'] =  query_by_text(transformer, transformed, documents, index, q, n_results=n_results)
			return response

		except Exception as e:
			return {'error': str(e)}

app = Flask(__name__, static_url_path="")
auth = HTTPBasicAuth()

api = Api(app)
api.add_resource(Query, '/Query/')

if __name__ == '__main__':

	transformed_file = 'transformed.pickle'
	docs_file = 'documents.pickle'
	index_file = 'index.pickle'
	transformer_file = 'transformer.pickle'

	transformed = np.load(transformed_file)

	index = pickle.load(open(index_file,'rb'))

	documents = pickle.load(open(docs_file,'rb'))
	print 'number of documents :', len(index)

	transformer = pickle.load(open(transformer_file,'rb'))

	print 'Ready to call!!'
	app.run(host='0.0.0.0', threaded=True)
