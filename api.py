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

from flask import Flask, request, make_response
from flask_httpauth import HTTPBasicAuth
from flask_restful import Resource, Api, reqparse

def top(n, sorted_results):
	return list(islice(sorted_results.iteritems(), n))

def query_by_text(transformer, transformed, documents, index, query_text, url, n_results=10):
	query = transformer.transform(query_text)
	sims = cosine_similarity(query.reshape(1,-1), transformed)
	scores =  sims[0][:].reshape(-1,1)
	results= dict()
	for i in range(len(transformed)):
		results[i] = scores[i]
	sorted_results = OrderedDict(sorted(results.items(), key=lambda k: k[1], reverse=True))
	topn = top(n_results, sorted_results)

	results = np.array(range(n_results), dtype=np.object)
	for rank, (answer, score) in enumerate(topn):
		title = documents[answer].split('__')[-1]
		title = title.split('\n')[0]
		title_t = title.replace (" ", "_")
		doc_id = str(index[answer])
		reference = url + title_t
		results[rank] = {'reference': reference, 'score': str(score), 'doc_id':  doc_id, 'title': title, 'answer': documents[answer]}

	return results.tolist()

class Query(Resource):
	def post(self):
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('question', type=str, required=True, help='Query text')
			parser.add_argument('userId', type=str, required=False, help='User ID')
			parser.add_argument('questionId', type=str, required=False, help='Question ID')
			parser.add_argument('limit', type=int, required=False, help='Size of the returned results')
			args = parser.parse_args()
			q = request.args.get('question')
			question = BeautifulSoup(q, "lxml").p.contents
			try:
				size = request.args.get('limit')
				n_results = int(size)
				if n_results > 100:
					n_results = 100
			except:
				n_results = 3
			user_id = request.args.get('userId')
			question_id = request.args.get('questionId')
			response = {}
			response['userId'] = user_id
			response['questionId'] = question_id
			response['limit'] = n_results
			response['interesteId'] = 'future_feature'
			response['results'] =  query_by_text(transformer, transformed, documents, index, question, url, n_results=n_results)
			if str(type(question)) == "<type 'list'>":
				question = question[0]
			response['question'] = question
			resp = make_response()
			resp.headers['Access-Control-Allow-Origin'] = '*'
			resp.headers['content-type'] = 'application/json'
			resp.data = response
			return response

		except Exception as e:
			return {'error': str(e)}

	def get(self):
		try:
			q = request.args.get('question')
			question = BeautifulSoup(q, "lxml").p.contents
			try:
				user_id = request.args.get('userId')
			except:
				user_id = 'uid1'
			try:
				question_id = request.args.get('questionId')
			except:
				question_id = 'qid1'
			try:
				size = request.args.get('limit')
				n_results = int(size)
				if n_results > 100:
					n_results = 100
			except:
				n_results = 3
			response = dict()
			response['userId'] = user_id
			response['questionId'] = question_id
			response['limit'] = n_results
			response['interesteId'] = 'future_feature'
			results =  query_by_text(transformer, transformed, documents, index, question, url, n_results=n_results)
			response['results'] = results
			if str(type(question)) == "<type 'list'>":
				question = question[0]
			response['question'] = question
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

	url_config = json.load(open('url_config.json', 'r'))
	url = url_config['url']

	print 'Ready to call!!'
	app.run(host='0.0.0.0', threaded=True)
