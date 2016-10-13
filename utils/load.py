# -*- coding: utf-8 -*-
#
# This file is part of semantic_query.
# Copyright (C) 2016 CIAPPLE.
#
# This is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

# Load corpus data


# Author: Hussein AL-NATSHEH <h.natsheh@ciapple.com>
# Affiliation: CIAPPLE, Jordan

import os

class Articles(object):
	def __init__(self, corpus):
		self.corpus = corpus
		self.index = dict()
		self.articles_count = 0
		self.docs_count = 0

	def __iter__(self):
		corpus = self.corpus
		for sub in os.listdir(corpus):
			subdir = os.path.join(corpus, sub)
			for fname in os.listdir(subdir):
				article_id = fname[:-4]
				paragraphs_count =0
				for line in open(os.path.join(subdir, fname)):
					paragraphs_count +=1
					self.docs_count += 1
					yield line
					self.index[str(self.docs_count)] = str(article_id)+'_'+str(paragraphs_count)
				self.articles_count += 1

	def print_stats(self):
		print 'number of articles: ', self.articles_count
		print 'number of docs: ', self.docs_count
