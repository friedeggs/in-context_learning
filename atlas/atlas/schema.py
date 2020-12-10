import sys, os 
import abc
import copy
from collections import defaultdict, OrderedDict, namedtuple
from itertools import product as cartesian_product
import exrex
import json
import numpy as np
import random
import re
import scipy
import scipy.special
from termcolor import colored
import traceback

from .util import (
	load_file,
	set_seed,
)

Content = namedtuple('Content', 'function unnaturalness')
Form = namedtuple('Form', 'function unnaturalness')
FormPair = namedtuple('FormPair', 'src_form tgt_form')
Sample = namedtuple('Sample', 'content src_form tgt_form')

class Schema():
	def __init__(self, contents, src_forms, tgt_forms):
		self.contents = contents
		self.forms = {
			'src_form': src_forms,
			'tgt_form': tgt_forms,
		}

def create_date_schema():
	contents = {
		'year':
			OrderedDict({
				0: Content(lambda: np.random.randint(1970, 2020), 0),
				1: Content(lambda: np.random.randint(2030, 2040), 1),
				2: Content(lambda: np.random.randint(1, 10_000), 2),
				# 3: Content(lambda: np.random.randint(2050, 1_000_000), 2),
			}),
		'month': 
			OrderedDict({
				0: Content(lambda: np.random.randint(1, 12+1), 0),
				1: Content(lambda: np.random.randint(40, 1000), 1),
				# 3: Content(lambda: np.random.randint(50, 1_000_000), 1),
			}),
		'day': 
			OrderedDict({
				0: Content(lambda: np.random.randint(1, 28+1), 0),
				1: Content(lambda: np.random.randint(40, 1000), 1),
				# 3: Content(lambda: np.random.randint(50, 1_000_000), 1),
			}),
	}

	date_forms = OrderedDict({
		0: Form(lambda _: f'{_.year}-{_.month:02d}-{_.day:02d}', 0),
		1: Form(lambda _: f'{_.month:02d}/{_.day:02d}/{_.year}', 0),
		# 2: Form(lambda _: f'{_.month} {_.day} {_.year}', 1),
		# 3: Form(lambda _: '{' + f'"month": {_.month}, "day": {_.day}, "year": {_.year}' + '}', 1),
		# 4: Form(lambda _: f'!{_.month}!{_.day}!{_.year}!', 2),
		2: Form(lambda _: f'{_.month:02d} {_.day:02d} {_.year}', 1),
		3: Form(lambda _: '{' + f'"month": {_.month:02d}, "day": {_.day:02d}, "year": {_.year}' + '}', 1),
		4: Form(lambda _: f'!{_.month:02d}!{_.day:02d}!{_.year}!', 2),
	})
	forms = {
		'src_form': date_forms,
		'tgt_form': date_forms,
	}

	DateSchema = namedtuple('DateSchema', 'year month day')
	DateSchema.contents = contents
	DateSchema.forms = forms
	return DateSchema

def create_name_schema():
	first_names = load_file('data/common_first_names.txt')
	last_names = load_file('data/common_last_names.txt')

	contents = {
		'firstName':
			OrderedDict({
				0: Content(lambda: np.random.choice(first_names), 0),
				1: Content(lambda: exrex.getone('[a-z]{%d}' % np.random.randint(5, 10)).capitalize(), 1),
			}),
		'lastName': 
			OrderedDict({
				0: Content(lambda: np.random.choice(last_names), 0),
				1: Content(lambda: exrex.getone('[a-z]{%d}' % np.random.randint(5, 10)).capitalize(), 1),
			}),
	}

	name_forms = OrderedDict({
		0: Form(lambda _: f'{_.firstName} {_.lastName}', 0),
		1: Form(lambda _: f'{_.lastName}, {_.firstName}', 0),
		2: Form(lambda _: '{' + f'"firstName": {_.firstName}, "lastName": {_.lastName}' + '}', 1),
		3: Form(lambda _: f'!{_.firstName}!{_.lastName}!', 1),
	})
	forms = {
		'src_form': name_forms,
		'tgt_form': name_forms,
	}

	NameSchema = namedtuple('NameSchema', 'firstName lastName')
	NameSchema.contents = contents
	NameSchema.forms = forms
	return NameSchema

def create_spelling_schema():
	pass

def create_url_schema():
	with open('data/urls2.txt') as f:
		real_urls = f.readlines()
		real_urls = list(map(lambda l: l.strip().split('\t'), real_urls))
	# print(real_urls[:5])
	words = [el for url in real_urls for el in url[1].split(' ')] # TODO or use common words from some vocabulary
	words = list(filter(lambda x: x.isalpha(), words)) 
	# print(words[:50])
	# set_seed()
	# strs = list(filter(lambda x: 10 < len(x), [exrex.getone('.*')[:30] for _ in range(1000)]))

	contents = {
		'url':
			OrderedDict({
				0: Content(lambda: real_urls[np.random.choice(len(real_urls))][0], 0),
				1: Content(lambda: 'http://' + exrex.getone('[a-zA-Z0-9]{%d}' % np.random.randint(5, 10)) + '.com', 1),
			}),
		'text': 
			OrderedDict({
				0: Content(lambda: real_urls[np.random.choice(len(real_urls))][1], 0),
				1: Content(lambda: ' '.join([np.random.choice(words) for _ in range(random.randint(2, 15))]), 1),
			}),
	}

	url_forms = OrderedDict({
		0: Form(lambda _: f'<a href="{_.url}">{_.text}</a>', 0),
		1: Form(lambda _: f'[{_.text}]({_.url})', 0),
		2: Form(lambda _: '{' + f'"url": {_.url}, "text": {_.text}' + '}', 1),
		3: Form(lambda _: f'!{_.url}!{_.text}!', 1),
	})
	forms = {
		'src_form': url_forms,
		'tgt_form': url_forms,
	}

	UrlSchema = namedtuple('UrlSchema', 'url text')
	UrlSchema.contents = contents
	UrlSchema.forms = forms
	return UrlSchema

def sample(schema_type, src_form_idx: int, tgt_form_idx: int, content_idx, seed=None):
	s = schema_type
	ci = content_idx
	sfi = src_form_idx
	tfi = tgt_form_idx
	if seed is not None:
		set_seed(seed)
	content = s(**{f: s.contents[f][getattr(ci, f)].function() for f in s._fields})
	src_form = s.forms['src_form'][sfi].function(content)
	tgt_form = s.forms['tgt_form'][tfi].function(content)
	return Sample(content, src_form, tgt_form)

def _exactly_k_unnatural(s, idxs, attr, fields, k=1):
	cnt = 0
	for f, idx in zip(fields, idxs):
		if getattr(s, attr)[f][idx].unnaturalness > 0:
			cnt += 1
			if cnt > k:
				return False
	return cnt == k

def exactly_k_unnatural(schema_type, attr, k=1):
	s = schema_type
	if attr == 'contents':
		nt = schema_type
	elif attr == 'forms':
		nt = FormPair
	else:
		raise Exception(f'Unrecognized attribute {attr}')
	lists = [getattr(s, attr)[f].keys() for f in nt._fields]
	all_possibilities = cartesian_product(*lists)
	possibilities = list(filter(lambda _: _exactly_k_unnatural(s, _, attr, nt._fields, k), all_possibilities))
	if attr == 'contents':
		possibilities = [s(**{f: p for f, p in zip(nt._fields, poss)}) for poss in possibilities]
	elif attr == 'forms':
		possibilities = list(filter(lambda _: _[0] != _[1], possibilities))
		possibilities = [nt(**{f: p for f, p in zip(nt._fields, poss)}) for poss in possibilities]
	return possibilities

def print_possibilities(s, possibilities, attr):
	for p in possibilities:
		print({f: getattr(s, attr)[f][v].unnaturalness for f, v in p._asdict().items()})

def print_samples(samples):
	for _ in samples:
		print(f'{_.src_form} \t-> \t{_.tgt_form}')

