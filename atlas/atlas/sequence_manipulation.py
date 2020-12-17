import sys, os
from collections import defaultdict, OrderedDict
import logging; log = logging.getLogger(__name__)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import numpy as np
from termcolor import colored
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from .api import (
	get_completion_s,
	get_ppl_s,
	get_top_logprobs_s,
	get_top_tokens_s,
)
from .content_lib import (
	random_distinct_alpha_chars,
	random_permutation,
	random_word_length_5,
)
from .dataset import (
	Dataset, FewShotDataset, FormattingDataset, FuncDataset, GPTDataset, IdentityDataset, IndexDataset, InputOutputDataset, IntDataset, ListDataset, NondeterministicDataset, ProductDataset, SumDataset
)
from .error_analysis import (
	add_neighbors,
	analyze_errors,
	get_form_primitives,
	get_value_dict,
	match_templates,
	filter_templates,
	print_templates,
)
from .form_lib import (
	space_separate,
)
from .gpt import GPT, get_cache
from .schema import (
	create_date_schema,
)
from .util import (
	count_tokens,
	make_immutable,
	permute,
)

DateSchema = create_date_schema()

TASK_DICT = defaultdict(int)
def register_task(func):
	global TASK_DICT
	name = func.__name__
	if name not in TASK_DICT:
		log.debug(f'Registering task {name}')
	TASK_DICT[name] += 1
	log.debug(f'{name}: {TASK_DICT[name]} calls')
	def wrapper(*args, **kwargs):
		return func(*args, **kwargs)
	return wrapper

class RandomDistinctAlphaCharsDataset(NondeterministicDataset):
	def __init__(self, n: int, **kwargs):
		self.n = n
		func = lambda idx: list(random_distinct_alpha_chars(self.n))
		super(RandomDistinctAlphaCharsDataset, self).__init__(func=func, **kwargs)

class RandomPermutationDataset(NondeterministicDataset):
	def __init__(self, n: int, **kwargs):
		self.n = n
		func = lambda idx: list(random_permutation(self.n))
		super(RandomPermutationDataset, self).__init__(func=func, **kwargs)

def run_task(argv, formatted, n_train, n_test, tag='', visualize: bool = True, 
		sample: Dataset = None, value_dict_func: Callable = None, index: int = -1):
	completion_kwargs = {
		'staged': True,
		'temperature': 0, 
		'engine': 'davinci', 
		'max_tokens': 20, 
		# 'staged': True, 
		'stop': '\n',
		'logprobs': 100,
		'echo': True,
	}
	mock = 'submit' not in argv
	cache = get_cache()
	gpt = GPT(cache, mock)

	x = FuncDataset(formatted, lambda _: _[index][0])
	y = FuncDataset(formatted, lambda _: _[index][1])
	prompt = InputOutputDataset(formatted)  # type: str
	n_tokens = FuncDataset(prompt, count_tokens)
	response = GPTDataset(prompt, gpt, completion_kwargs)
	response_prompt = SumDataset([response, prompt])
	pred = FuncDataset(response_prompt, lambda _: get_completion_s(_[0], completion_kwargs, _[1]))  # type: str
	correct = FuncDataset(SumDataset([pred, y]), lambda _: _[0] == _[1] if _[0] is not None else None)  # type: Optional[bool]
	ppl = FuncDataset(response_prompt, lambda _: get_ppl_s(_[0], completion_kwargs, _[1]))  # type: Optional[str]
	incorrect_indices = ListDataset([i for i, val in enumerate(correct) if val == False])  # type: Optional[float]
	top_logprobs = FuncDataset(response_prompt, lambda _: get_top_logprobs_s(_[0], completion_kwargs, _[1], completion_only=False))  # type: Optional[List[float]]
	log.info(prompt[0][-50:])
	# log.info([x for x in incorrect_indices])
	def analyze_templates(idx):
		# _sample = sample[idx]
		_pred = pred[idx]
		_x = x[idx]
		value_dict = value_dict_func(idx, sample, formatted)
		templates = match_templates(_pred, value_dict)
		min_length = min(map(len, templates))
		templates = list(filter(lambda _: len(_) == min_length, templates))
		print_templates(templates, None, _pred, _x)
		templates_by_name = list(map(lambda x1: list(map(lambda x2: list(map(lambda x3: x3[0], x2)), x1)), templates))
		return templates_by_name
	templates = FuncDataset(incorrect_indices, analyze_templates)
	rows = SumDataset([x, pred, y, correct, ppl,], keys=['x', 'pred', 'y', 'correct', 'ppl',])
	
	output_fname = f'outputs/results{tag}.csv'
	if False: # os.path.isfile(output_fname):
		df = pd.read_csv(output_fname)
	else:
		# rows = SumDataset(
		# 	[x, pred, y, correct,], 
		# 	keys=['x', 'pred', 'y', 'correct',])
		rows = SumDataset(
			# [x, pred, y, correct, ppl, top_logprobs,], 
			# keys=['x', 'pred', 'y', 'correct', 'ppl', 'top_logprobs',])
			[x, pred, y, correct, ppl, prompt, top_logprobs, response,], 
			keys=['x', 'pred', 'y', 'correct', 'ppl', 'prompt', 'top_logprobs', 'response',])
		# for _, i in zip(tqdm(range(len(rows))), range(len(rows))):
		# 	rows[i]
		# _rows = [r for _, r in zip(tqdm(range(len(rows))), rows)]
		_rows = [r for r in rows]
		# _token_counts = [_ for _ in n_tokens]
		# log.info('Token counts: %s' % str(_token_counts))
		# log.info('Max token count: %d' % max(_token_counts))
		# log.info('Total token count: %d' % sum(_token_counts))
		# log.info('Sample token count: %s' % n_tokens[0])
		# log.info(len(_rows))
		# # print(_rows)
		# # for _, r in zip(tqdm(_rows), _rows):
		for r in _rows:
			r['n_train'] = n_train

		for r in _rows[:5]:
			# print(r['prompt'])
			log.info({k: v for k, v in r.items() if k not in ['prompt', 'top_logprobs', 'response']})
		df = pd.DataFrame(_rows) # , columns=column_names)
		df.to_csv(output_fname)

	# df.fillna(value=pd.np.nan)
	if value_dict_func is not None and 'templates' not in df.keys():
		_templates = [None for _ in range(len(df))]
		for i, template in zip(incorrect_indices, templates):
			_templates[i] = template
		df = df.assign(templates=_templates)
		df.to_csv(output_fname)
	# if 'n_train' not in df.keys():
	# 	df['n_train'] = n_train
	# 	df.to_csv(output_fname)
	# dataset = response
	# for i, batch in enumerate(dataset):
	# 	if i >= 5:
	# 		break
	# 	print(batch)
	# log.info(df)
	log.info('Score: %.2f' % df.correct.mean())
	# log.info('Avg ppl: %.2f' % df.ppl.mean())

	if visualize and 'dates_unnatural_content' not in tag:
		try:
			_top_logprobs = np.exp(-np.stack(df.top_logprobs.values).mean(axis=0))
			# print(_top_logprobs[0].shape)
			# log.info(_top_logprobs.mean(axis=0).shape)
			# log.info(_top_logprobs)
			# log.info(df.top_logprobs)
			# n_tok_per = 2 + len(get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt']))
			# tokens = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'])
			# log.info(tokens)
			# toks = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'])
			tokens = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'], False)
			n_tok_per = tokens[1:].index('Input') + 1
			log.info(n_tok_per)
			# if 'dates' in tag:
			# 	import pdb; pdb.set_trace()
			for i in range(n_tok_per):
				_ = plt.figure(figsize=(15, 15))
				# log.info(tokens[i::n_tok_per][:25])
				plt.axhline(y=1., color='b', linestyle='-')
				if i == 0:
					ys = _top_logprobs[n_tok_per-1::n_tok_per]
					xs = range(1,1+len(ys))
				else:
					ys = _top_logprobs[(i-1)::n_tok_per]
					xs = range(len(ys))
				plt.scatter(x=xs, y=ys)
				bottom, top = plt.ylim()
				top = max(top, 10)
				plt.ylim(0, top)
				plt.title(' '.join(tokens[i::n_tok_per][:25]))
				plt.savefig(f'outputs/top_logprobs{tag}_{i}.png')
				# plt.savefig(f'outputs/top_logprobs_median{tag}_{i}.png')
				plt.clf()
				plt.close('all')
				# log.info(len(ys))  # n_train + 1
		except Exception as e:
			log.error(e)

	cost = gpt.calculate_cost()
	if cost:
		log.info('This request will cost %d tokens (including completion)' % cost)
		k = 'y'
		k = None
		if k == 'y':
			log.warn('Submitting queries without confirmation!')
		gpt.run_staged_queries(k)

def dates(argv, n_train=15, n_test=5):
	year = IntDataset(1970, 2020, offset=0)
	month = IntDataset(1, 12+1, offset=1)
	day = IntDataset(1, 28+1, offset=2)

	content_dataset = SumDataset([year, month, day])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda x: '/'.join(map(lambda n: f'{n:02d}', x)), 
		lambda x: '-'.join(map(lambda n: f'{n:02d}', x)),
		lambda x: '!'.join([''] + list(map(lambda n: f'{n:02d}', permute(x, [1,2,0]))) + ['']),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_dates_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_dates)

def dates_natural_format(argv, n_train=15, n_test=5):
	year = IntDataset(1970, 2020, offset=0)
	month = IntDataset(1, 12+1, offset=1)
	day = IntDataset(1, 28+1, offset=2)

	content_dataset = SumDataset([year, month, day])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda x: '/'.join(map(lambda n: f'{n:02d}', x)), 
		lambda x: '-'.join(map(lambda n: f'{n:02d}', x)),
		lambda x: '/'.join(map(lambda n: f'{n:02d}', permute(x, [1,2,0]))),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_dates_natural_format_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_dates)

def dates_unnatural_content(argv, n_train=15, n_test=5):
	year = IntDataset(10_000, 100_000, offset=0)
	month = IntDataset(10_000, 100_000, offset=1)
	day = IntDataset(10_000, 100_000, offset=2)

	content_dataset = SumDataset([year, month, day])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda x: '/'.join(map(lambda n: f'{n:02d}', x)), 
		lambda x: '-'.join(map(str, x)),
		lambda x: '!'.join([''] + list(map(str, permute(x, [1,2,0]))) + ['']),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_dates_unnatural_content_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_dates)

def reverse_natural_content(argv, n_train=80, n_test=5, n=5):
	content_dataset = NondeterministicDataset(func=lambda _: list(random_word_length_5()), offset=0)
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda _: space_separate(_), 
		# lambda _: space_separate(_[::-1]),
		lambda _: ', '.join(_), 
		lambda _: ', '.join(_[::-1]),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_reverse_natural_content_n-{n}_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_chars)

def reverse_to_natural_content(argv, n_train=80, n_test=5, n=5):
	content_dataset = NondeterministicDataset(func=lambda _: list(random_word_length_5())[::-1], offset=0)
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda _: space_separate(_), 
		# lambda _: space_separate(_[::-1]),
		lambda _: ', '.join(_), 
		lambda _: ', '.join(_[::-1]),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_reverse_to_natural_content_n-{n}_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_chars)

def reverse(argv, n_train=80, n_test=5, n=5):
	content_dataset = RandomDistinctAlphaCharsDataset(n, offset=0)
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda _: space_separate(_), 
		# lambda _: space_separate(_[::-1]),
		lambda _: ', '.join(_), 
		lambda _: ', '.join(_[::-1]),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_reverse_n-{n}_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_chars)

def addition_3_digit(argv, n_train=80, n_test=5, n=5):
	summand1 = IntDataset(100, 1_000, offset=0)
	summand2 = IntDataset(100, 1_000, offset=1)
	content_dataset = SumDataset([summand1, summand2])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda _: space_separate(_), 
		# lambda _: space_separate(_[::-1]),
		lambda _: f'{_[0]} + {_[1]}', 
		lambda _: str(sum(_)),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_addition_3_digit_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_numbers)

def get_value_dict_dates(idx, sample, formatted):
	_content = DateSchema(*sample[idx][-1])
	tgt_form = formatted.tgt_form
	return get_value_dict(_content, [tgt_form])

def get_value_dict_chars(idx, sample, formatted):
	_content = sample[idx][-1]
	tgt_form = formatted.tgt_form
	value_dict = OrderedDict({
		**{f'cp{i}': cp for i, cp in enumerate(_content)},
		**{'<alpha>': r'[a-z]',},
		**{'<*>': r'.',},
		**get_form_primitives(_content, [tgt_form]),
	})
	return value_dict

def get_value_dict_numbers(idx, sample, formatted):
	_content = sample[idx][-1]
	summand1, summand2 = _content
	_sum = summand1 + summand2
	# tgt_form = formatted.tgt_form
	value_dict = OrderedDict({
		# **{f'cp{i}': str(cp) for i, cp in enumerate(_content)},
		**{
			'summand1': summand1,
			'summand2': summand2,
			'sum': _sum,
		},
		**{'<n-digit>': r'\d+',},
		# **get_form_primitives(_content, [tgt_form]),
	})
	value_dict = add_neighbors_numeric(value_dict)
	return value_dict

def add_neighbors_numeric(base_value_dict):
	new_value_dict = {}

	for k, v in base_value_dict.items():
		new_value_dict[k] = str(v)

		if not isinstance(v, int):
			continue

		v_str = str(v)

		for i in range(len(v_str)):
			
			if v > 0:
				new_value_dict[f'{k}_minus-one'] = str(v-1)
			
			new_value_dict[f'{k}_plus-one'] = str(v+1)
			
			digit = int(v_str[i])

			if i < len(v_str)-1 and digit > 0: 
				new_value_dict[f'{k}_digit_{i}_minus-one'] = v_str[:i] + str(i-1) + v_str[i+1:]
			
			if i < len(v_str)-1 and digit < 9: 
				new_value_dict[f'{k}_digit_{i}_plus-one'] = v_str[:i] + str(i+1) + v_str[i+1:]
			
			new_value_dict[f'<{k}_digit_{i}_off>'] = v_str[:i] + '\d' + v_str[i+1:]

	base_value_dict = new_value_dict
	new_value_dict = {}
	for k, v in base_value_dict.items():
		if v != '':
			new_value_dict[k] = v
	return new_value_dict

def permutations(argv, n_train=100, n_test=5, n=5):
	content_dataset = RandomDistinctAlphaCharsDataset(n, offset=0)
	order_dataset = RandomPermutationDataset(n, offset=1)
	content = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	order = FuncDataset(order_dataset, funcs=[
		lambda x: [x for _ in range(n_train + 1)],
	])
	sample = SumDataset([content, order])
	sample = FuncDataset(sample, funcs=[
		lambda x: list(zip(*x)),
	])
	formatted = FormattingDataset(sample, 
		lambda pair: space_separate(pair[0]), 
		lambda pair: space_separate(permute(pair[0], pair[1])),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_permutations_n-{n}')

def random_char(argv, n_train=500, n_test=0):
	completion_kwargs = {
		'staged': True,
		'temperature': 0, 
		'engine': 'davinci', 
		'max_tokens': 20, 
		# 'staged': True, 
		'stop': '\n',
		'logprobs': 100,
		'echo': True,
	}
	mock = 'submit' not in argv
	cache = get_cache()
	gpt = GPT(cache, mock)

	content_dataset = RandomDistinctAlphaCharsDataset(1, offset=0)
	content = FewShotDataset(content_dataset, n_train=n_train, n_test=1)
	formatted = FuncDataset(content, lambda _: '\n'.join(map(lambda x: f'Input: {x[0]}', _)))
	prompt = formatted
	response = GPTDataset(prompt, gpt, completion_kwargs)
	response_prompt = SumDataset([response, prompt])
	top_logprobs = FuncDataset(response_prompt, lambda _: get_top_logprobs_s(_[0], completion_kwargs, _[1], completion_only=False))  # type: Optional[List[float]]
	dataset = formatted
	# for i, batch in enumerate(dataset):
	# 	if i >= 5:
	# 		break
	# 	print(batch)
	tag = '_random_char'
	_top_logprobs = np.exp(top_logprobs[0])
	# log.info(top_logprobs[0])
	log.info(_top_logprobs)
	tokens = get_top_tokens_s(response[0], completion_kwargs, prompt[0], False)
	n_tok_per = tokens[1:].index('Input') + 1
	log.info(n_tok_per)
	# if 'dates' in tag:
	# 	import pdb; pdb.set_trace()
	# for i in range(n_tok_per):
	i = 2
	fig = plt.figure(figsize=(15, 15))
	# log.info(tokens[i::n_tok_per][:25])
	plt.axhline(y=1./26, color='b', linestyle='-')
	if i == 0:
		ys = _top_logprobs[n_tok_per-1::n_tok_per]
		xs = range(1,1+len(ys))
	else:
		ys = _top_logprobs[(i-1)::n_tok_per]
		xs = range(len(ys))
	plt.scatter(x=xs, y=ys)
	plt.ylim(0, 1)
	plt.title(' '.join(tokens[i::n_tok_per][:25]))
	plt.savefig(f'outputs/top_logprobs{tag}_{i}.png')
	# plt.savefig(f'outputs/top_logprobs_median{tag}_{i}.png')
	plt.clf()
	plt.close('all')
	_logprobs = get_top_logprobs_s(response[0], completion_kwargs, prompt[0], completion_only=False, 
		keys=[' ' + chr(ord('a')+i) for i in range(26)])
	df = pd.DataFrame(_logprobs[i-1::n_tok_per])
	df = df.applymap(np.exp)
	log.info(df)
	log.info(df.describe())

def random_num(argv, low=1, high=5, n_train=500, n_test=0):
	completion_kwargs = {
		'staged': True,
		'temperature': 0, 
		'engine': 'davinci', 
		'max_tokens': 20, 
		# 'staged': True, 
		'stop': '\n',
		'logprobs': 100,
		'echo': True,
	}
	mock = 'submit' not in argv
	cache = get_cache()
	gpt = GPT(cache, mock)

	content_dataset = IntDataset(low, high, offset=0)
	content = FewShotDataset(content_dataset, n_train=n_train, n_test=1)
	formatted = FuncDataset(content, lambda _: '\n'.join(map(lambda x: f'Input: {x}', _)))
	prompt = formatted
	response = GPTDataset(prompt, gpt, completion_kwargs)
	response_prompt = SumDataset([response, prompt])
	top_logprobs = FuncDataset(response_prompt, lambda _: get_top_logprobs_s(_[0], completion_kwargs, _[1], completion_only=False))  # type: Optional[List[float]]
	dataset = formatted
	# for i, batch in enumerate(dataset):
	# 	if i >= 5:
	# 		break
	# 	print(batch)
	tag = f'_random_num_low-{low}_high-{high}'
	_top_logprobs = np.exp(top_logprobs[0])
	# log.info(top_logprobs[0])
	log.info(_top_logprobs)
	tokens = get_top_tokens_s(response[0], completion_kwargs, prompt[0], False)
	n_tok_per = tokens[1:].index('Input') + 1
	log.info(n_tok_per)
	# if 'dates' in tag:
	# 	import pdb; pdb.set_trace()
	# for i in range(n_tok_per):
	i = 2
	fig = plt.figure(figsize=(15, 15))
	# log.info(tokens[i::n_tok_per][:25])
	plt.axhline(y=1./(high-low), color='b', linestyle='-')
	if i == 0:
		ys = _top_logprobs[n_tok_per-1::n_tok_per]
		xs = range(1,1+len(ys))
	else:
		ys = _top_logprobs[(i-1)::n_tok_per]
		xs = range(len(ys))
	plt.scatter(x=xs, y=ys)
	plt.ylim(0, 1)
	plt.title(' '.join(tokens[i::n_tok_per][:25]))
	plt.savefig(f'outputs/top_logprobs{tag}_{i}.png')
	# plt.savefig(f'outputs/top_logprobs_median{tag}_{i}.png')
	plt.clf()
	plt.close('all')
	_logprobs = get_top_logprobs_s(response[0], completion_kwargs, prompt[0], completion_only=False, keys=[' 1', ' 2', ' 3', ' 4',])
	df = pd.DataFrame(_logprobs[i-1::n_tok_per])
	df = df.applymap(np.exp)
	log.info(df)
	log.info(df.describe())

@register_task
def calendar_2x2_exception(day_of_week, hour, meridiem_indicator):
	"""
	Args:
		day_of_week (str): Monday | ... | Friday | Saturday | Sunday
		hour (str): [8-12] 
		meridiem_indicator: (AM|PM)
	Returns:
		description (str): work morning | free evening | work evening | free morning | work noon | free noon
	Example(s):
		Monday 8 AM -> work morning
		Friday 12 AM -> free evening
	"""
	if day_of_week in ['Saturday', 'Sunday']:
		# available = 'free'
		available = 'weekend'
	else:
		# available = 'work'
		available = 'weekday'
	if hour == 12:
		if meridiem_indicator == 'PM':
			part_of_day = 'noon'
		else:
			part_of_day = 'evening'
	else:
		if meridiem_indicator == 'AM':
			part_of_day = 'morning'
		else:
			part_of_day = 'evening'
	# return f'{available} {part_of_day}'
	return (available, part_of_day)

@register_task
def calendar_2x2_exception_dummy(day_of_week, hour, meridiem_indicator):
	"""
	Args:
		day_of_week (str): $ | % | ^ | & | * | ( | )
		hour (str): [8-12] 
		meridiem_indicator: (@|#)
	Returns:
		description (str): A X | B Z | A Z | B X | A Y | B Y
	Example(s):
		$ 8 @ -> A X
		* 12 @ -> B Z
	"""
	if day_of_week in ['(', ')']:
		available = 'B'
	else:
		available = 'A'
	if hour == 12:
		if meridiem_indicator == '#':
			part_of_day = 'Y'
		else:
			part_of_day = 'Z'
	else:
		if meridiem_indicator == '@':
			part_of_day = 'X'
		else:
			part_of_day = 'Z'
	# return f'{available} {part_of_day}'
	return (available, part_of_day)

def setup_calendar_2x2_exception(argv, n_train=10, n_test=5, exclude_train_from_test=True):
	day_of_week = ListDataset(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
	hour = ListDataset(list(range(8, 12+1)))
	meridiem_indicator = ListDataset(['AM', 'PM'])
	dataset = ProductDataset([day_of_week, hour, meridiem_indicator])
	indexer_nd = RandomPermutationDataset(len(dataset), offset=0)
	# content = FuncDataset(indexer_nd, lambda _: [dataset[i] for i in _[:n_train+1]])  # hack
	content = FuncDataset(indexer_nd, lambda permutation: [dataset[i] for i in permutation])
	# content.get_train = lambda self, idx: self.get_split('train', idx)[:n_]
	identity = IdentityDataset()
	content_single = FuncDataset(identity, lambda idx: content[idx//n_train][idx%n_train])
	content_single.max_splits = 1  # bit hacky
	sample = FewShotDataset(content_single, n_train=n_train, n_test=n_test, exclude_train_from_test=exclude_train_from_test)
	formatted = FormattingDataset(sample, 
		lambda _: space_separate(list(map(str, _))), 
		lambda _: space_separate(calendar_2x2_exception(*_)),
		map=True,
	)
	# for i in range(1):
	# 	log.info(formatted[i])
		# log.info(sample[i])
		# log.info(len(sample[i]))
		# log.info(len(set(make_immutable(sample[i]))))
	# 	# log.info(content_single[i])
	# log.info(content[0])
	# log.info(content[1])
	run_task(argv, formatted, n_train, n_test, f'_calendar_2x2_exception_n_train-{n_train}')

def setup_calendar_2x2_exception_dummy(argv, n_train=10, n_test=5, exclude_train_from_test=True):
	day_of_week = ListDataset(list('$%^&*()'))
	hour = ListDataset(list(range(8, 12+1)))
	meridiem_indicator = ListDataset(['@', '#'])
	dataset = ProductDataset([day_of_week, hour, meridiem_indicator])
	indexer_nd = RandomPermutationDataset(len(dataset), offset=0)
	# content = FuncDataset(indexer_nd, lambda _: [dataset[i] for i in _[:n_train+1]])  # hack
	content = FuncDataset(indexer_nd, lambda permutation: [dataset[i] for i in permutation])
	# content.get_train = lambda self, idx: self.get_split('train', idx)[:n_]
	identity = IdentityDataset()
	content_single = FuncDataset(identity, lambda idx: content[idx//n_train][idx%n_train])
	content_single.max_splits = 1  # bit hacky
	sample = FewShotDataset(content_single, n_train=n_train, n_test=n_test, exclude_train_from_test=exclude_train_from_test)
	formatted = FormattingDataset(sample, 
		lambda _: space_separate(list(map(str, _))), 
		lambda _: space_separate(calendar_2x2_exception_dummy(*_)),
		map=True,
	)
	for i in range(1):
		log.info(formatted[i])
		# log.info(sample[i])
		# log.info(len(sample[i]))
		# log.info(len(set(make_immutable(sample[i]))))
	# 	# log.info(content_single[i])
	# log.info(content[0])
	# log.info(content[1])
	run_task(argv, formatted, n_train, n_test, f'_calendar_2x2_exception_dummy_n_train-{n_train}')

# def unnatural_addition_pt_2(summand1, operator, summand2):
# 	"""
# 	Args:
# 		summand1, summand2 (int): [-10,10]
# 	"""
# 	pass

# # boolean gates
# # natural
# concat
# order
# unary ops
# - classify, map
# binary ops
# - logical operators
# # products
# 	AND
# 	OR
# 	XOR
# 	yes no
# 	True False
# 	1 0
# 	+ -
# 	@ #
# 	# :) :(
# small PPL

def quick_tests():
	pass 


