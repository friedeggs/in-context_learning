
import sys, os
# import exrex
import json
from num2words import num2words
import numpy as np
from operator import add, sub, mul, truediv as div, neg
import pandas as pd
import random
import scipy
import scipy.special
from collections import defaultdict, OrderedDict
from termcolor import colored

from process import (
	GPT3,
	MockGPT3,
	read_cache,
)
from sample import (
	evaluate
)

from util import escape_ansi, set_seed

set_seed()

# CSV_PATH = 'results_formatting.csv'
CSV_PATH = 'results_sequence_mapping.csv'
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
]
keys_to_keep = [
	'engine',
	'temperature',
	'max_tokens',
	'staged',
	# 'prompt',
	'stop',
	'num_examples',
	# 'response',
	'rel',
	'x',
	'y',
	'pred',
	# 'balanced',
	# 'n',
	'score',
]

DEFAULT_KWARGS = {
	'temperature': 0, 
	'prefix': None, 
	# 'engine': engine, 
	'max_tokens': 50, # 20, 
	'staged': True, 
	'return_kwargs': True,
	'stop': '\n',
	'verbose': False,
	'logprobs': 100,
	# 'formatter': formatter,
}

def _run_helper(gpt3, engine, train_examples, test_examples, formatter=None, **kwargs):
	score = 0
	close = 0
	contains = 0
	total = 0
	pending = 0

	rows = []

	default_generation_kwargs = {
		'temperature': 0, 
		'prefix': None, 
		'engine': engine, 
		'max_tokens': 45, 
		# 'max_tokens': 50, # 45, 
		'staged': True, 
		'return_kwargs': True,
		'formatter': formatter,
		'stop': '\n',
		'verbose': True,
	}
	_kwargs = {**default_generation_kwargs, **kwargs}

	for idx, (x, y) in enumerate(test_examples):
		if idx == 0:
			cur_kwargs = _kwargs.copy()
			cur_kwargs['verbose'] = True
		else:
			cur_kwargs = _kwargs
		response, rel, kwargs = gpt3.few_shot(
			train_examples, 
			x=x, y=y, 
			**cur_kwargs,
		)
		rel = escape_ansi(rel)
		try:
			pred = response['choices'][0]['text'].lstrip().rstrip()
		except Exception:
			pred = None
		if rel == 'EQUALS':
			score += 1	
		elif rel == 'CLOSE':
			close += 1
		elif rel == 'CONTAINS':
			contains += 1
		if pred is not None:
			total += 1	
		else:
			pending += 1

		row = kwargs
		del row['prompt']
		# row['n'] = x.count('{')
		row['num_examples'] = len(train_examples)
		row['balanced'] = (y == 'balanced')
		row['x'] = x
		row['y'] = y
		row['pred'] = pred
		row['rel'] = rel
		row['score'] = (rel == 'EQUALS')
		rows.append(row)

	# print(colored('Engine: %s' % engine, 'magenta'))
	# print(colored('Score: %d/%d (%d close, %d contains); %d pending' % (score, total, close, contains, pending), 'magenta'))
	# print('')


	df = pd.DataFrame(rows) # , columns=column_names)
	if os.path.isfile(CSV_PATH):
		df_prev = pd.read_csv(CSV_PATH)
		df = pd.concat([df, df_prev], sort=False)
		df['is_duplicate'] = df[keys_for_comparison].duplicated()
		df = df[~df['is_duplicate']]
		# print(df['rel'].value_counts())
	df = df[keys_to_keep]
	df = df.dropna(subset=['pred','x'])
	df.to_csv(CSV_PATH)

def run_aba_abb(gpt3, engine, formatter=None):
	set_seed()
	n_train = 16
	n_test = 8
	train_syllables = ['de', 'di', 'je', 'ji', 'le', 'li', 'we', 'wi',]
	test_syllables = ['ba', 'po', 'ka', 'ko', 'ga', 'go',]

	train_examples_aba = []
	train_examples_abb = []
	for i in range(n_train):
		a, b = np.random.choice(train_syllables, 2, replace=False)
		train_examples_aba.append((f'{a} {b}', f'{a}'))
		train_examples_abb.append((f'{a} {b}', f'{b}'))

	test_examples_aba = []
	test_examples_abb = []
	for i in range(n_test):
		a, b = np.random.choice(test_syllables, 2, replace=False)
		test_examples_aba.append((f'{a} {b}', f'{a}'))
		test_examples_abb.append((f'{a} {b}', f'{b}'))

	_run_helper(gpt3, engine, train_examples_aba, test_examples_aba, formatter=formatter)
	_run_helper(gpt3, engine, train_examples_abb, test_examples_abb, formatter=formatter)

def run_aba_abb_comma_separated(gpt3, engine):
	set_seed()
	formatter = lambda tup: f'{tup[0]}, {tup[1]}'

	n_train = 16
	n_test = 8
	train_syllables = ['de', 'di', 'je', 'ji', 'le', 'li', 'we', 'wi',]
	test_syllables = ['ba', 'po', 'ka', 'ko', 'ga', 'go',]

	train_examples_aba = []
	train_examples_abb = []
	for i in range(n_train):
		a, b = np.random.choice(train_syllables, 2, replace=False)
		train_examples_aba.append((f'{a}, {b}', f'{a}'))
		train_examples_abb.append((f'{a}, {b}', f'{b}'))

	test_examples_aba = []
	test_examples_abb = []
	for i in range(n_test):
		a, b = np.random.choice(test_syllables, 2, replace=False)
		test_examples_aba.append((f'{a}, {b}', f'{a}'))
		test_examples_abb.append((f'{a}, {b}', f'{b}'))

	_run_helper(gpt3, engine, train_examples_aba, test_examples_aba, formatter=formatter)
	_run_helper(gpt3, engine, train_examples_abb, test_examples_abb, formatter=formatter)

def run_char_substitution(gpt3, engine, offset=None):
	set_seed()
	n_train = 50
	n_test = 3
	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'
	# offset = 1
	# offset = 2
	for k in range(5):
		ch1 = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
		if offset is not None:
			ch2 = chr((ord(ch1)-ord('a')+offset)%26+ord('a'))
		else:
			ch2 = ch1.upper()
		examples = []
		for i in range(n_train + n_test):
			word = ' '.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz') + [ch1] * 24, np.random.randint(5,11), replace=True))
			examples.append((word, word.replace(ch1,ch2)))

		# for x, y in examples[n_train:]:
		# 	response, rel, kwargs = gpt3.few_shot(
		# 		examples[:n_train], x=x, y=y, 
		# 		temperature=0, prefix=None, 
		# 		engine=engine, 
		# 		max_tokens=15, staged=True, 
		# 		return_kwargs=True,
		# 		formatter=formatter,
		# 		verbose=False,
		# 	)
		_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], formatter=formatter, max_tokens=15)


def run_char_substitution_comma_separated(gpt3, engine, offset=None):
	set_seed()
	n_train = 50
	n_test = 3
	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'
	# offset = 1
	# offset = 2
	score = 0
	close = 0
	total = 0
	pending = 0
	for k in range(5):
		ch1 = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
		if offset is not None:
			ch2 = chr((ord(ch1)-ord('a')+offset)%26+ord('a'))
		else:
			ch2 = ch1.upper()
		examples = []
		for i in range(n_train + n_test):
			word = ', '.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz') + [ch1] * 24, np.random.randint(5,11), replace=True))
			examples.append((word, word.replace(ch1,ch2)))

		_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], formatter=formatter, max_tokens=25)

	# 	for x, y in examples[n_train:]:
	# 		response, rel, kwargs = gpt3.few_shot(
	# 			examples[:n_train], x=x, y=y, 
	# 			temperature=0, prefix=None, 
	# 			engine=engine, 
	# 			max_tokens=25, staged=True, # NOTE 
	# 			return_kwargs=True,
	# 			formatter=formatter,
	# 			verbose=False,
	# 		)
	# 		rel = escape_ansi(rel)
	# 		try:
	# 			pred = response['choices'][0]['text'].lstrip().rstrip()
	# 		except Exception:
	# 			pred = None
	# 		if rel == 'EQUALS':
	# 			score += 1	
	# 		elif rel == 'CLOSE':
	# 			close += 1
	# 		if pred is not None:
	# 			total += 1	
	# 		else:
	# 			pending += 1

	# print(colored('Engine: %s' % engine, 'magenta'))
	# print(colored('Score: %d/%d (%d close); %d pending' % (score, total, close, pending), 'magenta'))
	# print('')

def run_char_substitution_star(gpt3, engine, n_train, separator=', ', offset=None):
	set_seed()
	n_test = 3
	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'
	# offset = 1
	# offset = 2
	score = 0
	close = 0
	total = 0
	pending = 0
	for k in range(5):
		ch1 = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
		ch2 = '*'
		examples = []
		for i in range(n_train + n_test):
			word = separator.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz') + [ch1] * 24, np.random.randint(5,11), replace=True))
			examples.append((word, word.replace(ch1,ch2)))

		_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], formatter=formatter, max_tokens=25)

	# 	for x, y in examples[n_train:]:
	# 		response, rel, kwargs = gpt3.few_shot(
	# 			examples[:n_train], x=x, y=y, 
	# 			temperature=0, prefix=None, 
	# 			engine=engine, 
	# 			max_tokens=25, staged=True, # NOTE 
	# 			return_kwargs=True,
	# 			formatter=formatter,
	# 			verbose=False,
	# 		)
	# 		rel = escape_ansi(rel)
	# 		try:
	# 			pred = response['choices'][0]['text'].lstrip().rstrip()
	# 		except Exception:
	# 			pred = None
	# 		if rel == 'EQUALS':
	# 			score += 1	
	# 		elif rel == 'CLOSE':
	# 			close += 1
	# 		if pred is not None:
	# 			total += 1	
	# 		else:
	# 			pending += 1

	# print(colored('Engine: %s' % engine, 'magenta'))
	# print(colored('Score: %d/%d (%d close); %d pending' % (score, total, close, pending), 'magenta'))
	# print('')

def run_remove_nonalphanumeric(gpt3, engine, n_train, separator=', ', offset=None):
	set_seed()
	n_test = 3
	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'
	# offset = 1
	# offset = 2
	score = 0
	close = 0
	total = 0
	pending = 0
	for k in range(5):
		ch1 = np.random.choice(list('!@#$%^&*()_+'))
		examples = []
		for i in range(n_train + n_test):
			chars = np.random.choice(list('abcdefghijklmnopqrstuvwxyz') + [ch1] * 24, np.random.randint(5,11), replace=True)
			chars2 = list(filter(lambda x: x != ch1, chars))
			word = separator.join(chars)
			word2 = separator.join(chars2)
			examples.append((word, word2))

		_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], formatter=formatter, max_tokens=25)

	# 	for x, y in examples[n_train:]:
	# 		response, rel, kwargs = gpt3.few_shot(
	# 			examples[:n_train], x=x, y=y, 
	# 			temperature=0, prefix=None, 
	# 			engine=engine, 
	# 			max_tokens=25, staged=True, # NOTE 
	# 			return_kwargs=True,
	# 			formatter=formatter,
	# 			verbose=False,
	# 		)
	# 		rel = escape_ansi(rel)
	# 		try:
	# 			pred = response['choices'][0]['text'].lstrip().rstrip()
	# 		except Exception:
	# 			pred = None
	# 		if rel == 'EQUALS':
	# 			score += 1	
	# 		elif rel == 'CLOSE':
	# 			close += 1
	# 		if pred is not None:
	# 			total += 1	
	# 		else:
	# 			pending += 1

	# print(colored('Engine: %s' % engine, 'magenta'))
	# print(colored('Score: %d/%d (%d close); %d pending' % (score, total, close, pending), 'magenta'))
	# print('')

def run_remove_nonalphanumeric_distinct(gpt3, engine, n_train, n_test, separator=', ', offset=None, max_tokens=25, n_kinds=5, **kwargs):
	# formatter = lambda tup: f'{tup[0]} -> {tup[1]}'
	formatter = None
	score = 0
	total = n_test * n_kinds
	ch1s = np.random.choice(list('!@#$%^&*()_+'), n_kinds, replace=False)
	for k, ch1 in enumerate(ch1s):
		set_seed(k)
		examples = []
		set_seed(ord(ch1))
		for i in range(n_train + n_test):
			chars = np.random.choice(list('abcdefghijklmnopqrstuvwxyz') + [ch1] * 24, np.random.randint(5,11), replace=False)
			chars2 = list(filter(lambda x: x != ch1, chars))
			word = separator.join(chars)
			word2 = separator.join(chars2)
			examples.append((word, word2))

		# _run_helper(gpt3, engine, examples[:n_train], examples[n_train:], formatter=formatter, max_tokens=25)
		kwargs = {
			'engine': engine, 
			'formatter': formatter,
			'max_tokens': max_tokens,
		}
		kwargs = {**DEFAULT_KWARGS, **kwargs}
		additional_kwargs = {
			'task': 'remove_nonalphanumeric_distinct',
			'schema_type': 'sequence_mapping',
		}
		score += evaluate(gpt3, 
			train_examples=examples[:n_train], 
			test_examples=examples[n_train:], 
			additional_kwargs=additional_kwargs, 
			**kwargs
		)
	print(colored('Engine: %s' % engine, 'magenta'))
	print(colored('Score: %d/%d' % (score, total), 'magenta'))
	print('')


def run_remove_nonalphanumeric_and_reverse(gpt3, engine, n_train, separator=', ', offset=None):
	set_seed()
	n_test = 3
	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'
	# offset = 1
	# offset = 2
	score = 0
	close = 0
	total = 0
	pending = 0
	for k in range(5):
		ch1 = np.random.choice(list('!@#$%^&*()_+'))
		examples = []
		for i in range(n_train + n_test):
			chars = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5, replace=False)) \
				+ [ch1] * np.random.randint(1,6)
			chars = np.random.permutation(chars)
			chars2 = list(reversed(list(filter(lambda x: x != ch1, chars))))
			word = separator.join(chars)
			word2 = separator.join(chars2)
			examples.append((word, word2))

		_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], formatter=formatter, max_tokens=25)

	# 	for x, y in examples[n_train:]:
	# 		response, rel, kwargs = gpt3.few_shot(
	# 			examples[:n_train], x=x, y=y, 
	# 			temperature=0, prefix=None, 
	# 			engine=engine, 
	# 			max_tokens=25, staged=True, # NOTE 
	# 			return_kwargs=True,
	# 			formatter=formatter,
	# 			verbose=False,
	# 		)
	# 		rel = escape_ansi(rel)
	# 		try:
	# 			pred = response['choices'][0]['text'].lstrip().rstrip()
	# 		except Exception:
	# 			pred = None
	# 		if rel == 'EQUALS':
	# 			score += 1	
	# 		elif rel == 'CLOSE':
	# 			close += 1
	# 		if pred is not None:
	# 			total += 1	
	# 		else:
	# 			pending += 1

	# print(colored('Engine: %s' % engine, 'magenta'))
	# print(colored('Score: %d/%d (%d close); %d pending' % (score, total, close, pending), 'magenta'))
	# print('')

def run_char_substitution_hyphenated(gpt3, engine, offset=None):
	set_seed()
	n_train = 50
	n_test = 3
	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'
	# offset = 1
	# offset = 2
	score = 0
	close = 0
	total = 0
	pending = 0
	for k in range(5):
		ch1 = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
		if offset is not None:
			ch2 = chr((ord(ch1)-ord('a')+offset)%26+ord('a'))
		else:
			ch2 = ch1.upper()
		examples = []
		for i in range(n_train + n_test):
			word = '-'.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz') + [ch1] * 24, np.random.randint(5,11), replace=True))
			examples.append((word, word.replace(ch1,ch2)))

		_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], formatter=formatter, max_tokens=25)

	# 	for x, y in examples[n_train:]:
	# 		response, rel, kwargs = gpt3.few_shot(
	# 			examples[:n_train], x=x, y=y, 
	# 			temperature=0, prefix=None, 
	# 			engine=engine, 
	# 			max_tokens=25, staged=True, # NOTE 
	# 			return_kwargs=True,
	# 			formatter=formatter,
	# 			verbose=False,
	# 		)
	# 		rel = escape_ansi(rel)
	# 		try:
	# 			pred = response['choices'][0]['text'].lstrip().rstrip()
	# 		except Exception:
	# 			pred = None
	# 		if rel == 'EQUALS':
	# 			score += 1	
	# 		elif rel == 'CLOSE':
	# 			close += 1
	# 		if pred is not None:
	# 			total += 1	
	# 		else:
	# 			pending += 1

	# print(colored('Score: %d/%d (%d close); %d pending' % (score, total, close, pending), 'magenta'))
	# print('')

def run_reverse_10(gpt3, engine):
	n_test = 2

	for n_words in range(5,10):
		print(f'Number of words: {n_words}')
		for n_train in range((n_words - 3)*5, (n_words + 3)*5, 5):
			set_seed()
			examples = []
			for i in range(n_train + n_test):
				words = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n_words, replace=False)
				examples.append((', '.join(words), ', '.join(reversed(words))))
			_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])
"""
5 10
6 15
7 25, 40
8 40
9 > 45
"""

def run_reverse_fixed_length(gpt3, engine, separator=', ', n=5, n_train=50):
	set_seed()
	n_test = 5

	examples = []
	for i in range(n_train + n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n, replace=False))
		examples.append((separator.join(words), separator.join(reversed(words))))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_reverse_6_7(gpt3, engine):
	set_seed()
	n_train = 50
	n_test = 5

	examples = []
	for i in range(n_train + n_test):
		words = [
			np.random.choice(list('abcdefghijklmnopqrstuvwxyz')) 
			for _ in range(np.random.randint(6,8))
		]
		# [''.join(np.random.choice(
		# 	list('abcdefghijklmnopqrstuvwxyz'), 3)
		# ) for _ in range(np.random.randint(5,8))]
		examples.append((', '.join(words), ', '.join(reversed(words))))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_reverse_more_lengths(gpt3, engine):
	set_seed()
	n_per = 5
	n_lengths = 12-2

	train_examples = []
	for i in range(2, 2 + n_lengths):
		set_seed(i)
		for j in range(n_per):
			words = [
				np.random.choice(list('abcdefghijklmnopqrstuvwxyz')) #, replace=False) 
				for _ in range(i) 
			]
			train_examples.append((', '.join(words), ', '.join(reversed(words))))
	train_examples = np.random.permutation(train_examples)

	test_examples = []
	for i in range(2, 2 + n_lengths):
		set_seed(10000 + i)
		for j in range(2):
			words = [
				np.random.choice(list('abcdefghijklmnopqrstuvwxyz')) #, replace=False) 
				for _ in range(i) 
			]
			test_examples.append((', '.join(words), ', '.join(reversed(words))))
	_run_helper(gpt3, engine, train_examples, test_examples)

def run_reverse_generalize(gpt3, engine):
	set_seed()
	n_train = 80
	n_test = 10

	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(4,6), replace=False))
		examples.append((', '.join(words), ', '.join(reversed(words))))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 6, replace=False))
		examples.append((', '.join(words), ', '.join(reversed(words))))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], x_label='Original', y_label='Reversed', max_tokens=20)

def run_reverse_generalize_numbers(gpt3, engine):
	set_seed()
	n_train = 80
	n_test = 10

	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('0123456789'), np.random.randint(4,6), replace=False)) # 4,6
		examples.append((', '.join(words), ', '.join(reversed(words))))
	for i in range(n_test):
		words = list(np.random.choice(list('0123456789'), 6, replace=False)) # 6
		examples.append((', '.join(words), ', '.join(reversed(words))))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], x_label='Original', y_label='Reversed', max_tokens=20)

def run_reverse_generalize_numbers_2(gpt3, engine):
	set_seed()
	n_train = 40 # 100 # 80
	n_test = 20

	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(3,5), replace=False)) # 2,5; 2,5; 
		examples.append((', '.join(words), ', '.join(reversed(words))))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(5,7), replace=False)) # 5,8; 5,7; 
		examples.append((', '.join(words), ', '.join(reversed(words))))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], x_label='Original', y_label='Reversed', max_tokens=20)

def run_transpose(gpt3, engine):
	set_seed()
	n_train = 50 # 100 # 80
	n_test = 10

	"""
	1,6
	6,10

	2,4
	4,6

	2,4
	2,4
	"""
	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 6, replace=False)) 
		transposed = [words[(i//2)*2+(i+1)%2] for i in range(len(words))]
		examples.append((', '.join(words), ', '.join(transposed)))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 8, replace=False)) 
		transposed = [words[(i//2)*2+(i+1)%2] for i in range(len(words))]
		examples.append((', '.join(words), ', '.join(transposed)))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], x_label='Input', y_label='Output', max_tokens=20)

def run_transpose_last(gpt3, engine):
	set_seed()
	n_train = 50 # 100 # 80
	n_test = 10

	"""
	2,6
	6,10

	6,8
	8,10
	"""
	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(6,8), replace=False)) 
		transposed = words[:-2] + [words[-1], words[-2]]
		examples.append((', '.join(words), ', '.join(transposed)))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(6,8), replace=False)) 
		transposed = words[:-2] + [words[-1], words[-2]]
		examples.append((', '.join(words), ', '.join(transposed)))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], x_label='Input', y_label='Output', max_tokens=20)

def run_transpose_last_fixed_length(gpt3, engine):
	set_seed()
	n_train = 50 # 100 # 80
	n_test = 10

	"""
	8
	"""
	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 8, replace=False)) 
		transposed = words[:-2] + [words[-1], words[-2]]
		examples.append((', '.join(words), ', '.join(transposed)))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 8, replace=False)) 
		transposed = words[:-2] + [words[-1], words[-2]]
		examples.append((', '.join(words), ', '.join(transposed)))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], x_label='Input', y_label='Output', max_tokens=20)

def run_transpose_first(gpt3, engine):
	set_seed()
	n_train = 50 # 100 # 80
	n_test = 10

	"""
	2,6
	6,10

	6,8
	8,10
	"""
	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(6,8), replace=False)) 
		transposed = [words[1], words[0]] + words[2:]
		examples.append((', '.join(words), ', '.join(transposed)))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(6,8), replace=False)) 
		transposed = [words[1], words[0]] + words[2:]
		examples.append((', '.join(words), ', '.join(transposed)))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], x_label='Input', y_label='Output', max_tokens=20)

def rearrange_2D_variable(gpt3, engine, min_size, max_size):
	set_seed()
	n_train = 50
	n_test = 10

	examples = []
	for i in range(n_train + n_test):
		R = np.random.randint(min_size, max_size)
		C = np.random.randint(min_size, max_size)
		words = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), (R, C), replace=False)
		x = '\n'.join(list(map(lambda _: ', '.join(_), words)))
		y = '\n'.join(list(map(lambda _: ', '.join(_), words.T)))
		examples.append(('\n' + x, y))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], stop=None, max_tokens=40, verbose=False)

def rearrange_2D_fixed(gpt3, engine, R, C, R1=None, C1=None, n_train=15, x_prefix='\n', y_prefix=''):
	set_seed()
	n_test = 10
	R1 = R1 or R
	C1 = C1 or C

	examples = []
	for i in range(n_train):
		words = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), (R, C), replace=False)
		x = '\n'.join(list(map(lambda _: ', '.join(_), words)))
		y = '\n'.join(list(map(lambda _: ', '.join(_), words.T)))
		examples.append((x_prefix + x, y_prefix + y))
	for i in range(n_test):
		words = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), (R1, C1), replace=False)
		x = '\n'.join(list(map(lambda _: ', '.join(_), words)))
		y = '\n'.join(list(map(lambda _: ', '.join(_), words.T)))
		examples.append((x_prefix + x, y_prefix + y))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], stop=None, max_tokens=50, verbose=False)

def run_permutation(gpt3, engine, n): 
	set_seed()
	n_train = 50
	n_test = 5

	for _ in range(5):
		order = np.random.permutation(range(n))
		examples = []
		for i in range(n_train + n_test):
			words = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n, replace=False)
			examples.append((', '.join(words), ', '.join(words[order])))
		_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_spelling(gpt3, engine): 
	set_seed()
	n_train = 50 # 5
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, ' '.join(list(word.upper()))))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_2(gpt3, engine): 
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
		examples.append((word, ' '.join(list(word.lower()))))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_and_interleave(gpt3, engine): 
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
		examples.append((word, '-'.join(list(word.upper()))))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, logprobs=100, formatter=formatter)

def run_spelling_and_interleave_reverse(gpt3, engine): 
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
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, logprobs=100, formatter=formatter)

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

def run_spelling_and_uppercase_all_but_first_two(gpt3, engine): 
	set_seed()
	n_train = 100
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	def func(word):
		return ' '.join(list(word[:2]) + list(word[2:].upper()))

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, func(word)))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_and_uppercase_alternate(gpt3, engine): 
	set_seed()
	n_train = 100
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	def func(word):
		lst = []
		for i, c in enumerate(list(word)):
			if i % 2 == 0:
				lst.append(c)
			else:
				lst.append(c.upper())
		return ' '.join(lst)

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, func(word)))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_and_char_substitution(gpt3, engine): 
	set_seed()
	n_train = 100
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)
	lines = list(filter(lambda x: 'a' in x, lines))

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, ' '.join(list(word.upper())).replace('A','a')))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_and_char_substitution_b(gpt3, engine): 
	set_seed()
	n_train = 100
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)
	lines = list(filter(lambda x: 'a' in x, lines))

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, ' '.join(list(word.upper())).replace('A','B')))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_and_char_substitution_ae(gpt3, engine): 
	set_seed()
	n_train = 100
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)
	lines = list(filter(lambda x: 'a' in x, lines))

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, ' '.join(list(word.upper())).replace('A','AE')))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_and_char_substitution_star(gpt3, engine): 
	set_seed()
	n_train = 100
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)
	lines = list(filter(lambda x: 'a' in x, lines))

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, ' '.join(list(word.upper())).replace('A','*')))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_and_char_substitution_vowels_to_stars(gpt3, engine): 
	set_seed()
	n_train = 100
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)
	lines = list(filter(lambda x: 'a' in x, lines))

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, ' '.join(list(word.upper()))
			.replace('A','*')
			.replace('E','*')
			.replace('I','*')
			.replace('O','*')
			# .replace('Y','*')
			.replace('U','*')))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def vowels_to_stars(gpt3, engine, n_train=100, separator=' '): 
	set_seed()
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)
	lines = list(filter(lambda x: 'a' in x, lines))

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((separator.join(list(word.upper())), separator.join(list(word.upper()))
			.replace('A','*')
			.replace('E','*')
			.replace('I','*')
			.replace('O','*')
			# .replace('Y','*')
			.replace('U','*')))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

def run_spelling_and_char_substitution_uv(gpt3, engine): 
	set_seed()
	n_train = 100
	n_test = 10

	with open('data/google-10000-english.txt') as f:
		lines = list(map(str.rstrip, f.readlines()))
	lines = np.random.permutation(lines)
	lines = list(filter(lambda x: 'u' in x, lines))

	formatter = lambda tup: f'{tup[0]} -> {tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		word = np.random.choice(lines)
		examples.append((word, ' '.join(list(word.upper())).replace('U','V')))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=50, formatter=formatter)

# def run_rearrange_sentence(gpt3, engine, n): 
# 	set_seed()
# 	n_train = 50
# 	n_test = 5

# 	for _ in range(5):
# 		order = np.random.permutation(range(n))
# 		examples = []
# 		for i in range(n_train + n_test):
# 			words = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n, replace=False)
# 			examples.append((', '.join(words), ', '.join(words[order])))
# 		_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_swap_first_last(gpt3, engine): 
	set_seed()
	n_train = 50
	n_test = 20

	examples = []
	for i in range(n_train + n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(5,10), replace=False))
		examples.append((', '.join(words), ', '.join([words[-1]] + words[1:-1] + [words[0]])))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_swap_first_last_generalize(gpt3, engine): 
	set_seed()
	n_train = 50
	n_test = 20

	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(3,7), replace=False))
		examples.append((', '.join(words), ', '.join([words[-1]] + words[1:-1] + [words[0]])))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(7,10), replace=False))
		examples.append((', '.join(words), ', '.join([words[-1]] + words[1:-1] + [words[0]])))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_swap_first_and_second_last(gpt3, engine): 
	set_seed()
	n_train = 50
	n_test = 10

	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(5,8), replace=False))
		examples.append((', '.join(words), ', '.join([words[-2]] + words[1:-2] + [words[0], words[-1]])))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(5,8), replace=False))
		examples.append((', '.join(words), ', '.join([words[-2]] + words[1:-2] + [words[0], words[-1]])))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_swap_first_and_second_last_fixed_length(gpt3, engine, length): 
	set_seed()
	n_train = 50
	n_test = 10

	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length, replace=False))
		examples.append((', '.join(words), ', '.join([words[-2]] + words[1:-2] + [words[0], words[-1]])))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length, replace=False))
		examples.append((', '.join(words), ', '.join([words[-2]] + words[1:-2] + [words[0], words[-1]])))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_swap_first_and_second_last_generalize(gpt3, engine): 
	set_seed()
	n_train = 50
	n_test = 10

	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(3,7), replace=False))
		examples.append((', '.join(words), ', '.join([words[-2]] + words[1:-2] + [words[0], words[-1]])))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(7,10), replace=False))
		examples.append((', '.join(words), ', '.join([words[-2]] + words[1:-2] + [words[0], words[-1]])))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_swap_first_last_longer(gpt3, engine): 
	set_seed()
	n_train = 30
	n_test = 20

	examples = []
	for i in range(n_train):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(7,10), replace=False))
		examples.append((', '.join(words), ', '.join([words[-1]] + words[1:-1] + [words[0]])))
	for i in range(n_test):
		words = list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), np.random.randint(7,10), replace=False))
		examples.append((', '.join(words), ', '.join([words[-1]] + words[1:-1] + [words[0]])))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def gen_brackets(n, lsym='(', rsym=')'):
	s = np.array(np.random.permutation([1,-1] * n))
	c = np.cumsum(s)
	brackets = ' '.join(list(map(lambda s: lsym if s > 0 else rsym, s)))
	balanced = (c >= 0).all()
	assert len(s) == 2 * n, brackets
	return (brackets, balanced)

def gen_balanced(n, lsym='(', rsym=')'):
	s = np.array(np.random.permutation([1,-1] * n))
	c = np.cumsum(s)
	x = np.insert(np.array(np.where(c == 0)) + 1, 0, 0)
	prefix = []
	suffix = []
	for i, j in zip(x[:-1],x[1:]):
		if s[i] > 0:
			prefix += list(s[i:j])
		else:
			prefix += [1]
			suffix += [-1] + list(-s[i+1:j-1])
	assert len(prefix + suffix) == 2 * n, prefix + suffix
	return ' '.join(list(map(lambda s: lsym if s > 0 else rsym, prefix + suffix)))

def run_balanced_brackets_0(gpt3, engine):
	set_seed()
	n_train = 25
	n_test = 25
	examples = []

	# n = np.random.randint(1,11)
	lengths = np.arange(4,6)
	# base distribution on Catalan numbers, not how many combinations there are total
	counts = np.array([scipy.special.comb(2*n, n)/(n+1) for n in lengths])
	probs = counts/counts.sum() 
	for i in range(n_train + n_test):
		n = np.random.choice(lengths, p=probs)
		if random.random() < .5-1./(n+1):
			examples.append((gen_balanced(n), 'balanced'))
		else:
			brackets, balanced = gen_brackets(n)
			examples.append((brackets, 'balanced' if balanced else 'unbalanced'))
	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:])

def run_balanced_brackets(gpt3, engine, bracket_symbols='()'):
	set_seed()
	n_train = 50
	n_test = 250

	# n = np.random.randint(1,11)
	lengths = np.arange(4,7)
	# >>> sum([14, 42])
	# 56
	# >>> sum([14, 42, 132])
	# 188
	# base distribution on Catalan numbers, not how many combinations there are total
	counts = np.array([scipy.special.comb(2*n, n)/(n+1) for n in lengths])
	probs = counts/counts.sum() 
	examples_balanced = {}
	examples_unbalanced = {}
	while len(examples_balanced) < (n_train+n_test)//2 or len(examples_unbalanced) < (n_train+n_test)//2:
		n = np.random.choice(lengths, p=probs)
		if random.random() < .5-1./(n+1):
			examples_balanced[gen_balanced(n, *list(bracket_symbols))] = 'balanced'
		else:
			brackets, balanced = gen_brackets(n, *list(bracket_symbols))
			if balanced:
				examples_balanced[brackets] = 'balanced'
			else:
				examples_unbalanced[brackets] = 'unbalanced'
	examples_balanced = list(examples_balanced.items())[:(n_train+n_test)//2]
	examples_unbalanced = list(examples_unbalanced.items())[:(n_train+n_test)//2]
	train_examples = examples_balanced[:n_train//2] + examples_unbalanced[:n_train//2]
	test_examples = examples_balanced[n_train//2:] + examples_unbalanced[n_train//2:]
	train_examples = np.random.permutation(train_examples)
	print(len(list(filter(lambda tup: tup[1] == 'balanced', train_examples)))) 
	print(len(list(filter(lambda tup: tup[1] != 'balanced', train_examples)))) 

	print(len(list(filter(lambda tup: tup[1] == 'balanced', test_examples)))) 
	print(len(list(filter(lambda tup: tup[1] != 'balanced', test_examples)))) 

	# lengths = np.arange(1,4)
	# # >>> sum([14, 42])
	# # 56
	# # >>> sum([14, 42, 132])
	# # 188
	# # base distribution on Catalan numbers, not how many combinations there are total
	# counts = np.array([scipy.special.comb(2*n, n)/(n+1) for n in lengths])
	# probs = counts/counts.sum() 
	# examples_short = {}
	# while len(examples_short) < 2*(1+2+5):
	# 	n = np.random.choice(lengths, p=probs)
	# 	if random.random() < .5-1./(n+1):
	# 		examples_short[gen_balanced(n, *list(bracket_symbols))] = 'balanced'
	# 	else:
	# 		brackets, balanced = gen_brackets(n, *list(bracket_symbols))
	# 		examples_short[brackets] = 'balanced' if balanced else 'unbalanced'
	# # while len(examples_short) < n_train + n_test:
	# 	n = np.random.choice(lengths, p=probs)
	# 	if random.random() < .5-1./(n+1):
	# 		examples_short[gen_balanced(n, *list(bracket_symbols))] = 'balanced'
	# 	else:
	# 		brackets, balanced = gen_brackets(n, *list(bracket_symbols))
	# 		examples_short[brackets] = 'balanced' if balanced else 'unbalanced'
	# examples_short = list(examples_short.items())
	# examples_short = np.random.permutation(examples_short)
	# examples = np.concatenate([examples, examples_short])

	_run_helper(gpt3, engine, train_examples, test_examples)

def run_balanced_brackets_4_5(gpt3, engine, bracket_symbols='()'):
	set_seed()
	n_train = 25
	n_test = 75

	# n = np.random.randint(1,11)
	lengths = np.arange(4,6)
	# >>> sum([14, 42])
	# 56
	# >>> sum([14, 42, 132])
	# 188
	# base distribution on Catalan numbers, not how many combinations there are total
	counts = np.array([scipy.special.comb(2*n, n)/(n+1) for n in lengths])
	probs = counts/counts.sum() 
	examples_balanced = {}
	examples_unbalanced = {}
	while len(examples_balanced) < (n_train+n_test)//2 or len(examples_unbalanced) < (n_train+n_test)//2:
		n = np.random.choice(lengths, p=probs)
		if random.random() < .5-1./(n+1):
			examples_balanced[gen_balanced(n, *list(bracket_symbols))] = 'balanced'
		else:
			brackets, balanced = gen_brackets(n, *list(bracket_symbols))
			if balanced:
				examples_balanced[brackets] = 'balanced'
			else:
				examples_unbalanced[brackets] = 'unbalanced'
	examples_balanced = list(examples_balanced.items())[:(n_train+n_test)//2]
	examples_unbalanced = list(examples_unbalanced.items())[:(n_train+n_test)//2]
	train_examples = examples_balanced[:n_train//2] + examples_unbalanced[:n_train//2]
	test_examples = examples_balanced[n_train//2:] + examples_unbalanced[n_train//2:]
	train_examples = np.random.permutation(train_examples)
	print(len(list(filter(lambda tup: tup[1] == 'balanced', train_examples)))) 
	print(len(list(filter(lambda tup: tup[1] != 'balanced', train_examples)))) 

	print(len(list(filter(lambda tup: tup[1] == 'balanced', test_examples)))) 
	print(len(list(filter(lambda tup: tup[1] != 'balanced', test_examples)))) 

	_run_helper(gpt3, engine, train_examples, test_examples)

def run_random_number_from_a_to_b(gpt3, engine, low=1, high=5):
	set_seed()
	n_train = 50
	n_test = 1

	formatter = lambda tup: f'Ex: {tup[1]}'

	counts = defaultdict(int)
	examples = []
	for i in range(n_train):
		x = np.random.randint(low, high)
		examples.append((None, x))
		counts[x] += 1
	for i in range(n_test):
		x = np.random.randint(low, high)
		examples.append((None, None))

	_run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=2, logprobs=100, formatter=formatter)
	counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
	# print(counts)
	print(json.dumps(counts, indent=4))

	# " 3": -1.2914898,
	# " 4": -1.313743,
	# " 2": -1.4056236,
	# " 1": -1.6714375,

	# " 4": -1.0392922,
	# " 1": -1.4164628,
	# " 2": -1.5644042,
	# " 3": -1.6542747,

# 36, 24, 24, 16

# " 4": 0.32
# " 1": 0.27
# " 2": 0.21
# " 3": 0.19

# " 4": 0.35
# " 1": 0.24
# " 2": 0.21
# " 3": 0.19

def run_random_bernoulli(gpt3, engine, p=.5):
	set_seed()
	n = 20

	formatter = lambda tup: f'Ex: {tup[1]}'

	counts = defaultdict(lambda: 1)
	examples = []
	prev = None
	for i in range(n):
		heads = random.random() < p
		x = 'True' if heads else 'False'
		examples.append((None, x))
		counts[x] += 1

		_run_helper(gpt3, engine, examples, [(None, None)], max_tokens=1, logprobs=100, formatter=formatter)
		# counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
		# # print(counts)
		# print(json.dumps(counts, indent=4))
		frac = 1. * (counts['True']-1) / (i+1)
		val = 1. * counts['True'] / (i+1+2)
		ch = ''
		if prev is not None:
			ch = '↓' if val < prev else '↑'
		prev = val
		print('%.2f %s %.2f' % (val, ch, frac))

def run_unnatural_addition(gpt3, engine, sep1=' + ', sep2=' = ', n_train=50, n_test=1, max_tokens=5, op=add, **kwargs):
	set_seed()

	formatter = lambda tup: f'{tup[0]}{sep2}{tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		a, b = np.random.randint(0, 100, 2)
		ans = op(a, b)
		examples.append(('%d%s%d' % (a, sep1, b), f'{ans}'))
	# _run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=max_tokens, formatter=formatter, **kwargs)
	kwargs = {
		'engine': engine, 
		'formatter': formatter,
		'max_tokens': max_tokens,
	}
	kwargs = {**DEFAULT_KWARGS, **kwargs}
	additional_kwargs = {
		'task': 'unnatural_addition',
		'schema_type': 'arithmetic',
		# 'a': a,
		# 'b': b,
		'sep1': sep1,
		'sep2': sep2,
	}
	evaluate(gpt3, 
		train_examples=examples[:n_train], 
		test_examples=examples[n_train:], 
		additional_kwargs=additional_kwargs, 
		**kwargs
	)

def run_arithmetic_in_words(gpt3, engine, sep1=' + ', sep2=' = ', n_train=50, n_test=1, max_tokens=10, op=add, **kwargs):
	set_seed()

	formatter = lambda tup: f'{tup[0]}{sep2}{tup[1]}'

	examples = []
	for i in range(n_train + n_test):
		a, b = np.random.randint(0, 100, 2)
		ans = op(a, b)
		a = num2words(a)
		b = num2words(b)
		ans = num2words(ans)
		examples.append((f'{a}{sep1}{b}', ans))

	# _run_helper(gpt3, engine, examples[:n_train], examples[n_train:], max_tokens=max_tokens, formatter=formatter, **kwargs)
	kwargs = {
		'engine': engine, 
		'formatter': formatter,
		'max_tokens': max_tokens,
	}
	kwargs = {**DEFAULT_KWARGS, **kwargs}
	additional_kwargs = {
		'task': 'arithmetic_in_words',
		'schema_type': 'arithmetic',
	}
	evaluate(gpt3, 
		train_examples=examples[:n_train], 
		test_examples=examples[n_train:], 
		additional_kwargs=additional_kwargs, 
		**kwargs
	)

"""
from formatting import load_df

df = load_df(); len(df)
df = df[df.engine == 'davinci']
df[df.score].size/df.size
df.score.sum(), len(df)
df[~df.score].groupby('n').size()
df[df.score].groupby('n').size() / df.groupby('n').size()
df[df.score].groupby('balanced').size() / df.groupby('balanced').size()
df[df.balanced].groupby('n').size() / df.groupby('n').size()
df[df.score].groupby('balanced').size(), df.groupby('balanced').size()
scipy.stats.binom_test(df.score.sum(), len(df))

import scipy
import scipy.stats
scipy.stats.binom_test(df.score.sum(), len(df))
df = load_df(); len(df)
df = df[df.engine == 'curie']
"""
def load_df(csv_path=CSV_PATH):
	df = pd.read_csv(csv_path)
	df = df[keys_to_keep]
	df = df.dropna(subset=['pred'])
	return df

def main(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	gpt3.clear_staged_queries()
	tasks = [
		# (run_reverse_10, [], {}),
		# (run_aba_abb, [], {}),
		# (run_aba_abb_comma_separated, [], {}),
		# (run_char_substitution, [], {}),
		# (run_char_substitution, [], {'offset': 1}),
		# (run_char_substitution, [], {'offset': 2}),
		# (run_char_substitution_comma_separated, [], {}),
		# (run_char_substitution_comma_separated, [], {'offset': 1}),
		# (run_char_substitution_comma_separated, [], {'offset': 2}),
		# (run_char_substitution_star, [], {'separator': ' ', 'n_train': 100}),
		# (run_char_substitution_star, [], {'separator': ', ', 'n_train': 50}),
		# (run_char_substitution_hyphenated, [], {}),
		# (run_char_substitution_hyphenated, [], {'offset': 1}),
		# (run_char_substitution_hyphenated, [], {'offset': 2}),
		# (run_balanced_brackets_4_5, [], {}),
		# (run_swap_first_last, [], {}),
		# (run_swap_first_last_generalize, [], {}),
		# (run_swap_first_last_longer, [], {}),
		# (run_swap_first_and_second_last_fixed_length, [6], {}),
		# (run_transpose_last, [], {}),
		# (run_transpose_last_fixed_length, [], {}),
		# (run_transpose_first, [], {}),
		# (rearrange_2D_fixed, [2, 3], {'n_train': 50, 'x_prefix': '\n', 'y_prefix': ''}),
		# (run_permutation, [5], {}),
		# (run_permutation, [6], {}),
		# (run_spelling, [], {}),
		# (run_spelling_2, [], {}),
		# (run_spelling_and_interleave, [], {}),
		# (run_spelling_and_interleave_reverse, [], {}),
		# (run_spelling_and_interleave_reverse_evaluate, [], {}),
		# (run_spelling_and_uppercase_all_but_first_two, [], {}),
		# (run_spelling_and_uppercase_alternate, [], {}),
		# (run_spelling_and_char_substitution, [], {}),
		# (run_spelling_and_char_substitution_b, [], {}),
		# (run_spelling_and_char_substitution_ae, [], {}),
		# (run_spelling_and_char_substitution_uv, [], {}),
		# (run_spelling_and_char_substitution_star, [], {}),
		# (run_spelling_and_char_substitution_vowels_to_stars, [], {}),
		# (vowels_to_stars, [], {}),
		# (vowels_to_stars, [], {'n_train': 50, 'separator': ', '}),
		# (run_remove_nonalphanumeric, [], {'n_train': 100, 'separator': ''}),
		# (run_remove_nonalphanumeric, [], {'n_train': 100, 'separator': ' '}),
		# (run_remove_nonalphanumeric_and_reverse, [], {'n_train': 100, 'separator': ' '}),
		# (run_reverse_fixed_length, [], {'n': 5, 'separator': ', ', 'n_train': 50}),
		# (run_reverse_fixed_length, [], {'n': 5, 'separator': ' ', 'n_train': 50}),
		# (run_reverse_fixed_length, [], {'n': 5, 'separator': ' ', 'n_train': 100}),
		# # davinci tasks
		# (run_swap_first_and_second_last, [], {}),
		# (run_swap_first_and_second_last_generalize, [], {}),
		# (run_swap_first_and_second_last_fixed_length, [8], {}),
		# (rearrange_2D_variable, [1, 4], {}),
		# (rearrange_2D_variable, [2, 4], {}),
		# (rearrange_2D_fixed, [4, 6], {'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [3, 6], {'n_train': 25, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [3, 5], {'n_train': 25, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [3, 4], {'n_train': 35, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [3, 3], {'n_train': 40, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [2, 6], {'n_train': 35, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [2, 6], {'n_train': 25, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [2, 7], {'n_train': 30, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [2, 8], {'n_train': 25, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [2, 4], {'C1': 8, 'n_train': 35, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [2, 4], {'C1': 6, 'n_train': 35, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [2, 6], {'C1': 8, 'n_train': 35, 'x_prefix': '\n', 'y_prefix': ''}),
		# (rearrange_2D_fixed, [6, 2], {'n_train': 35, 'x_prefix': '\n', 'y_prefix': ''}),
		# (run_random_number_from_a_to_b, [], {}),
		# # (run_random_number_from_a_to_b, [], {'low': , 'high': }),
		# (run_random_bernoulli, [], {}),
		# (run_unnatural_addition, [], {'sep1': ', ', 'sep2': ' -> '}),
		# (run_unnatural_addition, [], {'sep1': ', ', 'sep2': ', '}),
		# (run_unnatural_addition, [], {'sep1': ' # ', 'sep2': ' # '}),
		# (run_unnatural_addition, [], {'sep1': ' # ', 'sep2': ' # '}),
		# (run_unnatural_addition, [], {'logprobs': 100}),
		# (run_unnatural_addition, [], {'logprobs': 100, 'sep1': ' - '}),
		# # (run_unnatural_addition, [], {'logprobs': 100, 'n_train': 100, 'sep1': ' - '}),
		# (run_unnatural_addition, [], {'logprobs': 100, 'sep1': ' * '}),
		# (run_unnatural_addition, [], {'logprobs': 100, 'n_train': 100, 'n_test': 2, 'sep1': ' * '}),
		# (run_arithmetic_in_words, [], {'logprobs': 100, 'n_train': 100}),
		# (run_arithmetic_in_words, [], {'logprobs': 100, 'n_train': 100, 'sep1': ' - '}),
		# (run_arithmetic_in_words, [], {'logprobs': 100, 'n_train': 100, 'sep1': ', ', 'sep2': ', '}),
		# (run_arithmetic_in_words, [], {'logprobs': 100, 'n_train': 100, 'sep1': ', ', 'sep2': ' -> '}),
		# (run_unnatural_addition, [], {'logprobs': 100, 'n_test': 50, 'sep1': ' - ', 'sep2': ' = '}),
		# (run_arithmetic_in_words, [], {'logprobs': 100, 'n_train': 100, 'n_test': 50, 'sep1': ', ', 'sep2': ', '}),
		(run_remove_nonalphanumeric_distinct, [], {'logprobs': 100, 'n_train': 15, 'n_test': 5, 'n_kinds': 3}),
		(run_remove_nonalphanumeric_distinct, [], {'logprobs': 100, 'n_train': 100, 'n_test': 5, 'n_kinds': 3, 'separator': ' '}),
		(run_remove_nonalphanumeric_distinct, [], {'logprobs': 100, 'echo': True, 'n_train': 100, 'n_test': 5, 'n_kinds': 3, 'separator': ' '}),
	]
	# for engine in ['ada', 'babbage', 'curie', 'davinci']:
	# for engine in ['babbage', 'curie', 'davinci']:
	for func, args, kwargs in tasks:
		# for engine in ['ada', 'curie', 'davinci']:
		for engine in ['davinci']:
		# for engine in ['ada']: # , 'babbage', 'curie', 'davinci']:
		# for engine in ['ada', 'babbage', 'curie', 'davinci']:
			print(func.__name__, engine, args, kwargs)
			func(gpt3, engine, *args, **kwargs)

	print('This request will cost %d tokens' % gpt3.calculate_cost())
	gpt3.run_staged_queries()

	# FSTs

	# any fixed pattern 
	# delimited concatenation 
	# insensitivity 
	# fixed length 
	# rearrangement 

if __name__ == '__main__':
	main(sys.argv)
