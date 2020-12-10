import sys, os 
import copy
from collections import defaultdict, OrderedDict, namedtuple
from itertools import product as cartesian_product
import exrex
import json
import numpy as np
import pandas as pd
import random
import re
import scipy
import scipy.special
from termcolor import colored
import traceback

from data.model_info import model_info
# from formatting import (
# 	run_spelling_and_interleave_reverse
# )
from process import (
	GPT3, MockGPT3,
	read_cache, 
)
from process_transformers import Runner
import util
from util import set_seed

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model_id = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

def compute_perplexity_th(target_str, logprobs):
	labels = tokenizer.encode(target_str) 
	loss_fn = nn.CrossEntropyLoss()
	loss = loss_fn(labels, logprobs)
	return torch.exp(loss)

def get_perplexity(target_str, lps, gpt3, **run_kwargs):
	"""
	Args:
		target_str (str)
		lps (list of dict(str, float))
	"""
	tokens = util.get_tokenization(' ' + target_str + '\n')
	p = 0.
	computed = True
	for idx, tok in enumerate(tokens):
		_tok = None
		_p = -np.inf
		if idx >= len(lps):
			computed = False
		if not computed:
			print(run_kwargs)
			response = gpt3.complete(**run_kwargs)
			# gpt3.run_staged_queries()
			# response = gpt3.complete(**run_kwargs)
			if response is not None:
				_lps = response['choices'][0]['logprobs']['top_logprobs']
				lps = [None] * idx
				for _lp in _lps:
					lp = OrderedDict(sorted(_lp.items(), key=lambda x: -x[1]))
					lps.append(lp)
				# 	print(list(lp.items())[0])
				# print('--')
				computed = True

		if idx < len(lps):
			lp = lps[idx]
			_p = lp.get(tok, -np.inf)
			p += _p 

			_tok, _p = list(lp.items())[0]
			if tok != _tok:
				computed = False

		run_kwargs['prompt'] += tok
	return np.exp(-p/max(len(tokens), 1))

def compute_perplexity(target_str, logprobs):
	"""
	Args:
		target_str (str)
		logprobs (list of dict(str, float))
	"""
	tokens = util.get_tokenization(' ' + target_str + '\n')
	p = 0.
	for idx, (lp, tok) in enumerate(zip(logprobs, tokens), 1):
		_p = lp.get(tok, -np.inf)
		p += _p 
	return np.exp(-p/max(len(tokens), 1))

def estimate_perplexity(target_str, logprobs):
	"""
	Sum every possible way in which we can form target_str from the logprobs.
	Technically incorrect.
	Args:
		target_str (str)
		logprobs (list of dict(str, float))
	"""
	target_str = ' ' + target_str + '\n' # TODO whitespace leniency? # TODO account for + '\n'
	print(target_str.rstrip())
	curr_prefixes = defaultdict(lambda: -np.inf, {'': 0.})
	probs = [-np.inf]
	for idx, lp in enumerate(logprobs, 1):
		next_prefixes = defaultdict(list)
		for _s, _p in lp.items():
			for s, p in curr_prefixes.items():
				ss = s + _s
				if target_str.startswith(ss):
					next_prefixes[ss].append(p + _p)
		curr_prefixes = defaultdict(lambda: -np.inf, {k: scipy.special.logsumexp(v) for k, v in next_prefixes.items()})
		# print(curr_prefixes.keys())
		probs.append(curr_prefixes[target_str]/idx)
	return np.exp(-scipy.special.logsumexp(probs))

Content = namedtuple('Content', 'function unnaturalness')
Form = namedtuple('Form', 'function unnaturalness')
FormPair = namedtuple('FormPair', 'src_form tgt_form')
Sample = namedtuple('Sample', 'content src_form tgt_form')

# CSV_PATH = 'results_sample.csv'
# CSV_PATH = 'results_systematicity.csv'
CSV_PATH = 'results_error_analysis.csv'
rows = []
keys_for_comparison = [
	'engine',
	'temperature',
	'max_tokens',
	# 'staged',
	# 'prompt',
	'stop',
	'num_examples',
	# 'response',
	# 'rel',
	'x',
	'y',
	'test_idx',
]
keys_to_keep = [
	'engine',
	'temperature',
	'max_tokens',
	# 'staged',
	'prompt',
	'stop',
	'num_examples',
	# 'response',
	'rel',
	'x',
	'y',
	'pred',
	# 'balanced',
	# 'n',
	# 'score',
	'src_form',
	'tgt_form',
	'content',
	'src_form_un',
	'tgt_form_un',
	'content_un',
	'schema_type',
	'test_idx',
	# 'perplexity',
	'perplexity_pred',
	'perplexity_y',
	'perplexity_natural',
	'task',
	'templates',
]
data = []

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
	first_names = util.load_file('data/common_first_names.txt')
	last_names = util.load_file('data/common_last_names.txt')

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

def evaluate(gpt3, train_samples=None, test_samples=None, train_examples=None, test_examples=None, formatter=None, additional_kwargs={}, **kwargs):
	global rows, data
	if train_examples is None:
		train_examples = [(_.src_form, _.tgt_form) for _ in train_samples]
	if test_examples is None:
		test_examples = [(_.src_form, _.tgt_form) for _ in test_samples]
	score = 0
	for idx, (x, y) in enumerate(test_examples):
		# if '1988-01-11' not in x: continue
		response, rel, _kwargs = gpt3.few_shot(
			train_examples, 
			x=x, y=y, 
			formatter=formatter,
			**kwargs,
		)
		rel = util.escape_ansi(rel)
		if rel == 'EQUALS':
			score += 1
		cur_data = []
		try:
			pred = response['choices'][0]['text'].lstrip().rstrip()
			if 'logprobs' in kwargs and kwargs['logprobs'] is not None:
				# print(colored('|', 'yellow').join(response['choices'][0]['logprobs']['tokens']))
				arr = response['choices'][0]['logprobs']['top_logprobs']
				for obj in arr:
					obj = OrderedDict(sorted(obj.items(), key=lambda x: -x[1]))
					# print(json.dumps(obj, indent=4)) # , sort_keys=True))
					cur_data.append(list(obj.items()))
				data.append(cur_data)
			# print(dict(cur_data[0]))
		except Exception as e:
			# print(e)
			try:
				pred = response[0]
			except Exception:
				pred = None
		row = {**_kwargs, **additional_kwargs}
		row['num_examples'] = len(train_examples)
		row['x'] = x
		row['y'] = y
		row['pred'] = pred
		row['rel'] = rel
		# row['test_idx'] = idx
		# if y is not None:
		# 	perplexity = compute_perplexity(y, list(map(dict, cur_data))) # TODO not technically correct
		run_kwargs = {**kwargs, **{'prompt': row['prompt']}} 
		del run_kwargs['prefix']
		del run_kwargs['return_kwargs']
		# del run_kwargs['verbose']
		# del run_kwargs['staged']
		run_kwargs['staged'] = False
		# if pred is not None:
		# 	perplexity = get_perplexity(pred, list(map(dict, cur_data)), gpt3, **run_kwargs)
		# 	row['perplexity_pred'] = perplexity
		# 	if y is not None:
		# 		if 'task' in additional_kwargs and additional_kwargs['task'] != 'arithmetic_in_words':
		# 			perplexity = get_perplexity(y, list(map(dict, cur_data)), gpt3, **run_kwargs)
		# 			row['perplexity_y'] = perplexity
		# 	if 'task' in additional_kwargs and additional_kwargs['task'] == 'unnatural_addition':
		# 		sep1 = additional_kwargs['sep1']
		# 		sep2 = additional_kwargs['sep2']
		# 		a = int(row['prompt'].split('\n')[-1].split(sep1)[0])
		# 		b = int(row['prompt'].split('\n')[-1].split(sep1)[1].split(sep2[:-1])[0].rstrip())
		# 		perplexity = get_perplexity(f'{a - b}', list(map(dict, cur_data)), gpt3, **run_kwargs)
		# 		row['perplexity_natural'] = perplexity
		# 	if 'task' in additional_kwargs and additional_kwargs['task'] == 'DateSchema' and len(train_examples) > 5:
		# 		y2 = '!'.join(np.array(y.split('!'))[[0,2,1,3,4]])
		# 		perplexity = get_perplexity(y2, list(map(dict, cur_data)), gpt3, **run_kwargs)
		# 		row['perplexity_natural'] = perplexity
		# del row['prompt']
		rows.append(row)
	# save_df()
	return score

def run_schema_task(gpt3, engine, schema_type, **kwargs):
	default_kwargs = {
		'temperature': 0, 
		'prefix': None, 
		'engine': engine, 
		'max_tokens': 20, 
		'staged': True, 
		'return_kwargs': True,
		'stop': '\n',
		'verbose': False,
		'logprobs': 100,
	}
	kwargs = {**default_kwargs, **kwargs}

	set_seed(0)
	print(sample(schema_type, 0, 1, schema_type(*([0] * len(schema_type._fields)))))

	n_train = 3
	n_test = 10 # 5

	poss_fc = list(cartesian_product(
		exactly_k_unnatural(schema_type, 'forms'), 
		exactly_k_unnatural(schema_type, 'contents', 0))) \
		+ list(cartesian_product(
		exactly_k_unnatural(schema_type, 'forms', 0), 
		exactly_k_unnatural(schema_type, 'contents'))) \
		+ list(cartesian_product(
		exactly_k_unnatural(schema_type, 'forms', 0), 
		exactly_k_unnatural(schema_type, 'contents', 0)))

	# df = load_df()
	# df = df[(df.rel != 'EQUALS') & (df.num_examples == 5)]
	# poss_f = [FormPair(x, y) for x, y in list(df[['src_form', 'tgt_form']].values)]
	# poss_c = list(map(eval, df['content'].values))
	# poss_fc = list(zip(poss_f, poss_c))
	print('Number of possible form and content combinations to process: %d' % len(poss_fc))

	s = schema_type
	samples = [[sample(schema_type, pf.src_form, pf.tgt_form, pc, seed) for seed in range(n_train + n_test)] \
		for pf, pc in poss_fc]
	for sm, (pf, pc) in zip(samples, poss_fc):
		# print_samples(sm)
		# print()
		content_un = False
		for field in s._fields:
			if s.contents[field][getattr(pc, field)].unnaturalness > 0:
				content_un = True
		additional_kwargs = {
			'src_form': pf.src_form,
			'tgt_form': pf.tgt_form,
			'content': pc,
			'src_form_un': s.forms['src_form'][pf.src_form].unnaturalness,
			'tgt_form_un': s.forms['tgt_form'][pf.tgt_form].unnaturalness,
			'content_un': content_un, # s.contents[pc].unnaturalness,
			'seed': 0,
			'schema_type': schema_type.__name__
		}
		evaluate(gpt3, sm[:n_train], sm[n_train:], additional_kwargs=additional_kwargs, **kwargs)

def run_perplexity_investigation(gpt3, engine, schema_type, n_train=5, n_test=1000, **kwargs):
	default_kwargs = {
		'temperature': 0, 
		'prefix': None, 
		'engine': engine, 
		'max_tokens': 20, 
		'staged': True, 
		'return_kwargs': True,
		'stop': '\n',
		'verbose': False,
		'logprobs': 100,
	}
	kwargs = {**default_kwargs, **kwargs}

	set_seed(0)
	# print(sample(schema_type, 0, 1, schema_type(*([0] * len(schema_type._fields)))))

	n_train = 5 # 3
	n_test = 1000 # 100 # 10 # 5

	poss_fc = [
		(FormPair(0, 4), schema_type(*([0] * len(schema_type._fields))))
	]
	# poss_fc = list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms'), 
	# 	exactly_k_unnatural(schema_type, 'contents', 0))) \
	# 	+ list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms', 0), 
	# 	exactly_k_unnatural(schema_type, 'contents'))) \
	# 	+ list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms', 0), 
	# 	exactly_k_unnatural(schema_type, 'contents', 0)))
	print(poss_fc[0])
	print('Number of possible form and content combinations to process: %d' % len(poss_fc))

	s = schema_type
	samples = [[sample(schema_type, pf.src_form, pf.tgt_form, pc, seed) for seed in range(n_train + n_test)] \
		for pf, pc in poss_fc]
	for sm, (pf, pc) in zip(samples, poss_fc):
		# print_samples(sm)
		# print()
		content_un = False
		for field in s._fields:
			if s.contents[field][getattr(pc, field)].unnaturalness > 0:
				content_un = True
		additional_kwargs = {
			'src_form': pf.src_form,
			'tgt_form': pf.tgt_form,
			'content': pc,
			'src_form_un': s.forms['src_form'][pf.src_form].unnaturalness,
			'tgt_form_un': s.forms['tgt_form'][pf.tgt_form].unnaturalness,
			'content_un': content_un, # s.contents[pc].unnaturalness,
			'seed': 0,
			'schema_type': schema_type.__name__,
			'task': schema_type.__name__,
		}
		evaluate(gpt3, sm[:n_train], sm[n_train:], additional_kwargs=additional_kwargs, **kwargs)

def run_perplexity_investigation_sampled_train(gpt3, engine, schema_type, n_train=5, n_test=1000, tgt_form_type=4, **kwargs):
	default_kwargs = {
		'temperature': 0, 
		'prefix': None, 
		'engine': engine, 
		'max_tokens': 20, 
		'staged': True, 
		'return_kwargs': True,
		'stop': '\n',
		'verbose': False,
		'logprobs': 100,
	}
	kwargs = {**default_kwargs, **kwargs}

	set_seed(0)
	# print(sample(schema_type, 0, 1, schema_type(*([0] * len(schema_type._fields)))))

	# n_train = 5 # 3
	# n_test = 1000 # 100 # 10 # 5

	poss_fc = [
		(FormPair(0, tgt_form_type), schema_type(*([0] * len(schema_type._fields))))
	]
	# poss_fc = list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms'), 
	# 	exactly_k_unnatural(schema_type, 'contents', 0))) \
	# 	+ list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms', 0), 
	# 	exactly_k_unnatural(schema_type, 'contents'))) \
	# 	+ list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms', 0), 
	# 	exactly_k_unnatural(schema_type, 'contents', 0)))
	print(poss_fc[0])
	print('Number of possible form and content combinations to process: %d' % len(poss_fc))

	s = schema_type
	samples = [[[sample(schema_type, pf.src_form, pf.tgt_form, pc, seed=i * (n_train + 1) + j) for j in range(n_train + 1)] for i in range(n_test)] \
		for pf, pc in poss_fc]
	for sms, (pf, pc) in zip(samples, poss_fc):
		# print_samples(sm)
		# print()
		for idx, sm in enumerate(sms):
			content_un = False
			for field in s._fields:
				if s.contents[field][getattr(pc, field)].unnaturalness > 0:
					content_un = True
			test_sm = sm[n_train]
			additional_kwargs = {
				# 'src_form': pf.src_form,
				# 'tgt_form': pf.tgt_form,
				# 'content': pc,
				'src_form_type': pf.src_form,
				'tgt_form_type': pf.tgt_form,
				'content_type': pc,
				'src_form': test_sm.src_form,
				'tgt_form': test_sm.tgt_form,
				'content': test_sm.content,
				'src_form_un': s.forms['src_form'][pf.src_form].unnaturalness,
				'tgt_form_un': s.forms['tgt_form'][pf.tgt_form].unnaturalness,
				'content_un': content_un, # s.contents[pc].unnaturalness,
				'seed': 0,
				'schema_type': schema_type.__name__,
				'task': schema_type.__name__,
				'test_idx': idx,
			}
			evaluate(gpt3, sm[:n_train], sm[n_train:], additional_kwargs=additional_kwargs, **kwargs)

# def run_perplexity_investigation_spelling(gpt3, engine, schema_type, **kwargs):
# 		(run_spelling_and_interleave_reverse, [], {}),

def run_spelling_and_interleave_reverse_evaluate(gpt3, engine): 
	set_seed()
	n_train = 50
	n_test = 10 # 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append(('-'.join(list(word.upper())), word))

	kwargs = {
		'temperature': 0, 
		'prefix': None, 
		'engine': engine, 
		'max_tokens': 50, # 20, 
		'staged': True, 
		'return_kwargs': True,
		'stop': '\n',
		'verbose': False,
		'logprobs': 100,
		# 'formatter': formatter,
	}
	additional_kwargs = {
		'task': 'spelling_and_interleave_reverse',
		'schema_type': 'spelling'
	}
	evaluate(gpt3, train_examples=examples[:n_train], test_examples=examples[n_train:], additional_kwargs=additional_kwargs, **kwargs)

def run_date_investigation(gpt3, engine, schema_type, **kwargs):
	default_kwargs = {
		'temperature': 0, 
		'prefix': None, 
		'engine': engine, 
		'max_tokens': 20, 
		'staged': True, 
		'return_kwargs': True,
		'stop': '\n',
		'verbose': False,
	}
	kwargs = {**default_kwargs, **kwargs}

	set_seed(0)
	print(sample(schema_type, 0, 1, schema_type(*([0] * len(schema_type._fields)))))

	n_train = 3
	n_test = 100 # 5

	# poss_fc = list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms'), 
	# 	exactly_k_unnatural(schema_type, 'contents', 0))) \
	# 	+ list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms', 0), 
	# 	exactly_k_unnatural(schema_type, 'contents'))) \
	# 	+ list(cartesian_product(
	# 	exactly_k_unnatural(schema_type, 'forms', 0), 
	# 	exactly_k_unnatural(schema_type, 'contents', 0)))
	# poss_fc = [(exactly_k_unnatural(schema_type, 'forms', 2)[0], exactly_k_unnatural(schema_type, 'contents', len(schema_type._fields))[0])]
	# poss_fc = [(FormPair(0, 4), schema_type(*([0] * len(schema_type._fields))))]
	poss_fc = [
		(FormPair(0, 4), schema_type(*([0] * len(schema_type._fields))))
	]
	print(poss_fc[0])

	# df = load_df()
	# df = df[(df.rel != 'EQUALS') & (df.num_examples == 5)]
	# poss_f = [FormPair(x, y) for x, y in list(df[['src_form', 'tgt_form']].values)]
	# poss_c = list(map(eval, df['content'].values))
	# poss_fc = list(zip(poss_f, poss_c))
	print('Number of possible form and content combinations to process: %d' % len(poss_fc))

	s = schema_type
	samples = [[sample(schema_type, pf.src_form, pf.tgt_form, pc, seed) for seed in range(n_train + n_test)] \
		for pf, pc in poss_fc]
	print(samples[0][0]) # Sample(content=DateSchema(year=2014, month=6, day=1), src_form='2014-06-01', tgt_form='!06!01!2014!')
	for poss_idx, (pf, pc) in enumerate(poss_fc):
		for idx in [1,14,18,59,71,72,77,85,91,93]: # 10 # 3 x 5 x 10 = 150 47/150
			sm = copy.deepcopy(samples[poss_idx][n_train + idx])
			c = sm.content
			for i in range(5):
				modified = s(
					day=np.random.randint(1,28+1),
					month=np.random.randint(1,12+1),
					year=np.random.randint(1970,2020),
				)
				sfi = pf.src_form
				tfi = pf.tgt_form

				content = s(year=c.year,month=c.month,day=modified.day)
				src_form = s.forms['src_form'][sfi].function(content)
				tgt_form = s.forms['tgt_form'][tfi].function(content)
				samples[poss_idx].append(Sample(content, src_form, tgt_form))

				content = s(year=c.year,month=modified.month,day=c.day)
				src_form = s.forms['src_form'][sfi].function(content)
				tgt_form = s.forms['tgt_form'][tfi].function(content)
				samples[poss_idx].append(Sample(content, src_form, tgt_form))

				content = s(year=modified.year,month=c.month,day=c.day)
				src_form = s.forms['src_form'][sfi].function(content)
				tgt_form = s.forms['tgt_form'][tfi].function(content)
				samples[poss_idx].append(Sample(content, src_form, tgt_form))

	# for poss_idx, (pf, pc) in enumerate(poss_fc):
	# 	# 234  2016-08-16  !08!16!2016!  !16!08!2016!         9
	# 	# 294  2018-01-18  !01!18!2018!  !18!01!2018!        76
	# 	# 297  2018-05-18  !05!18!2018!  !18!05!2018!        79
	# 	for idx in [9,76,79]: # 3 # 3 x 5 x 3 = 45 11/45
	# 		sm = copy.deepcopy(samples[poss_idx][n_train + idx + 100])
	# 		c = sm.content
	# 		for i in range(5):
	# 			modified = s(
	# 				day=np.random.randint(1,28+1),
	# 				month=np.random.randint(1,12+1),
	# 				year=np.random.randint(1970,2020),
	# 			)
	# 			sfi = pf.src_form
	# 			tfi = pf.tgt_form

	# 			content = s(year=c.year,month=c.month,day=modified.day)
	# 			src_form = s.forms['src_form'][sfi].function(content)
	# 			tgt_form = s.forms['tgt_form'][tfi].function(content)
	# 			samples[poss_idx].append(Sample(content, src_form, tgt_form))

	# 			content = s(year=c.year,month=modified.month,day=c.day)
	# 			src_form = s.forms['src_form'][sfi].function(content)
	# 			tgt_form = s.forms['tgt_form'][tfi].function(content)
	# 			samples[poss_idx].append(Sample(content, src_form, tgt_form))

	# 			content = s(year=modified.year,month=c.month,day=c.day)
	# 			src_form = s.forms['src_form'][sfi].function(content)
	# 			tgt_form = s.forms['tgt_form'][tfi].function(content)
	# 			samples[poss_idx].append(Sample(content, src_form, tgt_form))

	for sm, (pf, pc) in zip(samples, poss_fc):
		# print_samples(sm)
		# print()
		content_un = False
		for field in s._fields:
			if s.contents[field][getattr(pc, field)].unnaturalness > 0:
				content_un = True
		additional_kwargs = {
			'src_form': pf.src_form,
			'tgt_form': pf.tgt_form,
			'content': pc,
			'src_form_un': s.forms['src_form'][pf.src_form].unnaturalness,
			'tgt_form_un': s.forms['tgt_form'][pf.tgt_form].unnaturalness,
			'content_un': content_un, # s.contents[pc].unnaturalness,
			'seed': 0,
			'schema_type': schema_type.__name__
		}
		evaluate(gpt3, sm[:n_train], sm[n_train:], additional_kwargs=additional_kwargs, **kwargs)
		# evaluate(gpt3, sm[:n_train] + [sm[n_train], sm[n_train+2]], sm[-150:], additional_kwargs=additional_kwargs, **kwargs)
		# evaluate(gpt3, sm[:n_train] + [sm[n_train], sm[n_train+2]], sm[-45:], additional_kwargs=additional_kwargs, **kwargs)


def save_df(csv_path=CSV_PATH, df=None):
	global rows
	if df is None:
		# print(rows)
		df = pd.DataFrame(rows) # , columns=column_names)
		# print(len(df))
	if os.path.isfile(csv_path):
		df_prev = pd.read_csv(csv_path)
		df = pd.concat([df, df_prev], sort=False)
	df['is_duplicate'] = df[keys_for_comparison].duplicated()
	df = df[~df['is_duplicate']]
	# print(df['rel'].value_counts())
	_keys_to_keep = set(keys_to_keep).intersection(df.keys())
	df = df[_keys_to_keep]
	# print(df)
	# print(len(df))
	df.to_csv(csv_path)

"""
from sample import load_df, get_latex_stats
df = load_df(); len(df)
df[(df.rel != 'EQUALS') & (df.num_examples == 3)]
df[(df.src_form_un > 0) & !(df.tgt_form_un > 0)]
"""
def load_df(csv_path=CSV_PATH):
	df = pd.read_csv(csv_path)
	_keys_to_keep = set(keys_to_keep).intersection(df.keys())
	df = df[_keys_to_keep]
	df = df.dropna(subset=['pred', 'schema_type']) # , 'content_un'])
	return df

"""
from sample import load_df, get_latex_stats, get_latex_plot_code, generate_latex_plots
df = load_df(); len(df)
df = df[df.num_examples == 3]
generate_latex_plots(df)

from sample import load_df, get_latex_stats, get_latex_plot_code, generate_latex_plots
df = load_df(); len(df)
df = df[(df.engine == 'davinci') & (df.num_examples == 3)]
df = df[(df.src_form == 0) & (df.tgt_form == 4)]
df[(df.rel != 'EQUALS')][['x','y','pred','test_idx']]
((df[(df.rel != 'EQUALS')][['x','y','pred','test_idx']].test_idx) % 3).value_counts()
df[(df.rel != 'EQUALS')][['x','y','pred','test_idx']].test_idx.values

from sample import load_df, get_latex_stats, get_latex_plot_code, generate_latex_plots
df = load_df(); len(df)
df = df[(df.engine == 'davinci') & (df.num_examples == 5)]
df = df[(df.src_form == 0) & (df.tgt_form == 4)]
df[(df.rel != 'EQUALS')][['x','y','pred','test_idx']]
((df[(df.rel != 'EQUALS')][['x','y','pred','test_idx']].test_idx) % 3).value_counts()
"""
def generate_latex_plots(df):
	plot_data = [
		(
			'Natural content; Natural form',
			(df.src_form_un == 0) & (df.tgt_form_un == 0) & (df.content_un == 0),
			"""
	color=gray,
	mark=square,
""",
		),
		(
			'Unnatural content; Natural form',
			(df.src_form_un == 0) & (df.tgt_form_un == 0) & ~(df.content_un == 0),
			"""
	color=red,
	mark=square,
""",
		),
		(
			'Natural content; Unnatural form',
			~((df.src_form_un == 0) & (df.tgt_form_un == 0)) & (df.content_un == 0),
			"""
	color=orange,
	mark=square,
""",
		),
		(
			'Unnatural source form',
			~(df.src_form_un == 0) & (df.tgt_form_un == 0) & (df.content_un == 0),
			"""
	color=yellow,
	mark=square,
""",
		),
		(
			'Unnatural target form',
			(df.src_form_un == 0) & ~(df.tgt_form_un == 0) & (df.content_un == 0),
			"""
	color=green,
	mark=square,
""",
		),
		(
			'Form to/from JSON',
			~((df.src_form != 2) & (df.tgt_form != 2)) & (df.content_un == 0),
			"""
	color=blue,
	mark=square,
""",
		),
		(
			'Form to/from ``!\'\'',
			~((df.src_form != 3) & (df.tgt_form != 3)) & (df.content_un == 0),
			"""
	color=violet,
	mark=square,
""",
		),
	]
	util.write_to_file(
		'figs/plot.tex', 
		get_latex_plot_code(df, 'Sensitivity to Content and Form Across Models -- Overview', plot_data) + "\n"
	)

	plot_data = [
		(
			'Natural content; Unnatural form; NameSchema',
			~((df.src_form_un == 0) & (df.tgt_form_un == 0)) & (df.content_un == 0) & (df.schema_type == 'NameSchema'),
			"""
	color=orange,
	mark=square,
""",
		),
		(
			'Natural content; Unnatural form; DateSchema',
			~((df.src_form_un == 0) & (df.tgt_form_un == 0)) & (df.content_un == 0) & (df.schema_type == 'DateSchema'),
			"""
	color=green,
	mark=square,
""",
		),
		(
			'Natural content; Unnatural form; UrlSchema',
			~((df.src_form_un == 0) & (df.tgt_form_un == 0)) & (df.content_un == 0) & (df.schema_type == 'UrlSchema'),
			"""
	color=cyan,
	mark=square,
""",
		),
	]
	util.write_to_file(
		'figs/plot_schemas.tex', 
		get_latex_plot_code(df, 'Sensitivity to Unnatural Form Across Models and Schema Types', plot_data) + "\n"
	)

	plot_data = [
		(
			'Form to/from natural; NameSchema',
			((df.src_form < 2) & (df.tgt_form < 2)) & (df.content_un == 0) & (df.schema_type == 'NameSchema'),
			"""
	color=red,
	mark=square,
""",
		),
		(
			'Form to/from JSON; NameSchema',
			~((df.src_form != 2) & (df.tgt_form != 2)) & (df.content_un == 0) & (df.schema_type == 'NameSchema'),
			"""
	color=teal,
	mark=square,
""",
		),
		(
			'Form to/from ``!\'\'; NameSchema',
			~((df.src_form != 3) & (df.tgt_form != 3)) & (df.content_un == 0) & (df.schema_type == 'NameSchema'),
			"""
	color=blue,
	mark=square,
""",
		),
	]
	util.write_to_file(
		'figs/plot_names.tex', 
		get_latex_plot_code(df, 'Sensitivity to Form Across Models; NameSchema', plot_data) + "\n"
	)

	plot_data = [
		(
			'Form to/from natural; DateSchema',
			((df.src_form < 2) & (df.tgt_form < 2)) & (df.content_un == 0) & (df.schema_type == 'DateSchema'),
			"""
	color=red,
	mark=square,
""",
		),
		(
			'Form to/from JSON; DateSchema',
			~((df.src_form != 2) & (df.tgt_form != 2)) & (df.content_un == 0) & (df.schema_type == 'DateSchema'),
			"""
	color=teal,
	mark=square,
""",
		),
		(
			'Form to/from ``!\'\'; DateSchema',
			~((df.src_form != 3) & (df.tgt_form != 3)) & (df.content_un == 0) & (df.schema_type == 'DateSchema'),
			"""
	color=blue,
	mark=square,
""",
		),
	]
	util.write_to_file(
		'figs/plot_dates.tex', 
		get_latex_plot_code(df, 'Sensitivity to Form Across Models; DateSchema', plot_data) + "\n"
	)

	plot_data = [
		(
			'Form to/from natural; UrlSchema',
			((df.src_form < 2) & (df.tgt_form < 2)) & (df.content_un == 0) & (df.schema_type == 'UrlSchema'),
			"""
	color=red,
	mark=square,
""",
		),
		(
			'Form to/from JSON; UrlSchema',
			~((df.src_form != 2) & (df.tgt_form != 2)) & (df.content_un == 0) & (df.schema_type == 'UrlSchema'),
			"""
	color=teal,
	mark=square,
""",
		),
		(
			'Form to/from ``!\'\'; UrlSchema',
			~((df.src_form != 3) & (df.tgt_form != 3)) & (df.content_un == 0) & (df.schema_type == 'UrlSchema'),
			"""
	color=blue,
	mark=square,
""",
		),
	]
	util.write_to_file(
		'figs/plot_urls.tex', 
		get_latex_plot_code(df, 'Sensitivity to Form Across Models; UrlSchema', plot_data) + "\n"
	)

	plot_data = [
		(
			'Form to/from natural',
			((df.src_form < 2) & (df.tgt_form < 2)) & (df.content_un == 0),
			"""
	color=red,
	mark=square,
""",
		),
		(
			'Form to/from JSON',
			~((df.src_form != 2) & (df.tgt_form != 2)) & (df.content_un == 0),
			"""
	color=teal,
	mark=square,
""",
		),
		(
			'Form to/from ``!\'\'',
			~((df.src_form != 3) & (df.tgt_form != 3)) & (df.content_un == 0),
			"""
	color=blue,
	mark=square,
""",
		),
	]
	util.write_to_file(
		'figs/plot_forms.tex', 
		get_latex_plot_code(df, 'Sensitivity to Form Across Models', plot_data) + "\n"
	)

def get_latex_plot_code(df, plot_title, plot_data):
	latex_code = """
\\begin{tikzpicture}
\\begin{axis}[
	title={%s},""".lstrip() % plot_title \
	+ """
	xlabel={Number of parameters},
	ylabel={Accuracy (\\%)},
	xmin=0, xmax=200000000000,
	ymin=0, ymax=100,
	legend pos=outer north east, %north west,
	ymajorgrids=true,
	grid style=dashed,
	xmode=log,
	legend cell align={left},
	every axis plot/.append style={thick}
]
"""
	line_names = []
	for line_name, condition, options in plot_data:
		stats = get_latex_stats(df, condition)
		latex_code += """
\\addplot[%s]
	coordinates {
	%s
	};
""" % (options, stats)
		line_names.append(line_name)

	latex_code += """
\\legend{
	%s
}
""" % (',%\n\t'.join(line_names + ['']).rstrip())
	
	latex_code += """
\\end{axis}
\\end{tikzpicture}
"""
	return latex_code

def get_latex_stats(df, condition):
	"""
	condition = (df.num_examples == 3) & ~((df.src_form_un == 0) & (df.tgt_form_un == 0))
	condition = (~(df.src_form_un == 0) & (df.tgt_form_un == 0)) | ((df.src_form_un == 0) & ~(df.tgt_form_un == 0))
	"""
	# model_info = util.load_file('model_info.json')
	scores = df[(df.rel == 'EQUALS') & condition]['engine'].value_counts() / df[condition]['engine'].value_counts()
	# , df[condition]['engine'].value_counts()
	data = []
	for model_name in [
			'gpt2', 
			'gpt2-medium', 
			'gpt2-large', 
			'gpt2-xl',
		] + [
			'ada', 
			'babbage', 
			'curie', 
			'davinci'
		]:
		num_params = model_info[model_name]['num_parameters']
		data += [(num_params, 100. * np.nan_to_num(scores.get(model_name, 0.)))]
	return str(data).replace('), (', ')(')[1:-1]
	# return data

def print_logprobs(engines):
	for engine, cur_data in zip(engines, data):
		print('Engine: %s' % engine)
		for items in zip(*cur_data):
			print('\t '.join(list(map(lambda x: x[0].replace("\n","nl").replace("<|endoftext|>","<eot>").ljust(5) + ': ' + f'{x[1]:.2f}'.ljust(8), items))))


def main(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	gpt3.clear_staged_queries()

	date_schema = create_date_schema()
	name_schema = create_name_schema()
	url_schema = create_url_schema()

	# # for engine in ['ada', 'babbage', 'curie']:
	# for engine in ['ada', 'davinci']:
	for engine in ['ada', 'babbage', 'curie', 'davinci']:
		print('Processing model %s' % engine)
		run_schema_task(gpt3, engine, date_schema, max_tokens=20)
		run_schema_task(gpt3, engine, name_schema, max_tokens=20)
		run_schema_task(gpt3, engine, url_schema, max_tokens=150)
		print()
	gpt3.run_staged_queries()
	save_df(rows)

	default_generation_kwargs = {
		'do_sample': True, 
		# 'max_length': 15, 
		'top_k': 0, 
		# 'top_p': 0.95, 
		'temperature': 0.0001, 
		# 'num_return_sequences': 5, 
		'num_return_sequences': 1, 
		'stop': '\n',
	}
	for model_name in [
		'gpt2-xl',
		'gpt2-large', 
		'gpt2-medium', 
		'gpt2', 
	]:
	# for model_name in model_names:
		# if 'gpt2' in model_name:
		#     continue
		print('Evaluating model %s' % model_name)
		set_seed()
		cache_fname = f'cache_{model_name}.jsonl'
		cache = read_cache(cache_fname)
		# settings = {'generation_kwargs': default_generation_kwargs}
		# cache['generation_kwargs'] = settings['generation_kwargs']
		# runner = Runner(model_name=model_name, settings=settings, cache=cache)
		# # test_copycat_remove(runner)
		# run_task_suite(runner, cache, cache_fname)
		# import pdb; pdb.set_trace()
		# write_cache(cache, cache_fname)
		# eval_copycat(model_name)
		# eval_arithmetic(model_name)
		gpt = Runner(model_name=model_name, settings={'generation_kwargs': default_generation_kwargs}, cache=cache)
		run_schema_task(gpt, model_name, date_schema, max_tokens=20, temperature=default_generation_kwargs['temperature'])
		run_schema_task(gpt, model_name, name_schema, max_tokens=20, temperature=default_generation_kwargs['temperature'])
		run_schema_task(gpt, model_name, url_schema, max_tokens=150, temperature=default_generation_kwargs['temperature'])
		# gpt.model.num_parameters()
	save_df(rows)
	df = load_df(); print(len(df))
	df = df[df.num_examples == 3]
	generate_latex_plots(df)
	print('Wrote LaTeX plots')

def main2(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	gpt3.clear_staged_queries()

	date_schema = create_date_schema()

	engines = ['ada', 'babbage', 'curie', 'davinci']
	for engine in engines:
		kwargs = {}
		default_kwargs = {
			'temperature': 0, 
			'prefix': None, 
			'engine': engine, 
			'max_tokens': 20, 
			'staged': True, 
			'return_kwargs': True,
			'stop': '\n',
			'verbose': False,
			'logprobs': 25,
		}
		kwargs = {**default_kwargs, **kwargs}
		schema_type = date_schema
		# set_seed(0)
		# print(sample(schema_type, 0, 4, schema_type(*([0] * len(schema_type._fields)))))

		n_train = 3
		n_test = 1
		s = schema_type
		samples = [[sample(schema_type, 0, 4, schema_type(*([0] * len(schema_type._fields))), seed) for seed in range(n_train + n_test)] \
			for _ in range(1)]
		for sm in samples:
			print_samples(sm)
			evaluate(gpt3, sm[:n_train], sm[n_train:], additional_kwargs={}, **kwargs)
	gpt3.run_staged_queries()

	for engine, cur_data in zip(engines, data):
		print('Engine: %s' % engine)
		for items in zip(*cur_data):
			print('\t '.join(list(map(lambda x: x[0].replace("\n","nl").replace("<|endoftext|>","<eot>").ljust(5) + ': ' + f'{x[1]:.2f}'.ljust(8), items))))

def main3(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	gpt3.clear_staged_queries()

	date_schema = create_date_schema()
	name_schema = create_name_schema()
	url_schema = create_url_schema()

	# for engine in ['ada', 'babbage', 'curie', 'davinci']:
	# for engine in ['ada', 'davinci']:
	for engine in ['davinci']:
		print('Processing model %s' % engine)
		run_date_investigation(gpt3, engine, date_schema, max_tokens=20)
		# run_schema_task(gpt3, engine, name_schema, max_tokens=20)
		# run_schema_task(gpt3, engine, url_schema, max_tokens=150)
		print()
	gpt3.run_staged_queries()
	save_df()

def slides():
	date_schema = create_date_schema()
	s = date_schema
	content = date_schema(2020, 11, 13)
	sfi = 0
	tfi = 1
	src_form = s.forms['src_form'][sfi].function(content)
	tgt_form = s.forms['tgt_form'][tfi].function(content)
	sm = Sample(content, src_form, tgt_form)
	example = (sm.src_form, sm.tgt_form)
	print('Input: %s\nOutput: %s' % (example))

# Input: 2020-11-13
# Output: 11/13/2020

if __name__ == '__main__':
	# main(sys.argv)
	# main2(sys.argv)
	main3(sys.argv)


