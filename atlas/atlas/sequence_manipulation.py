
import logging; log = logging.getLogger(__name__)
import matplotlib
matplotlib.use('tkAgg')
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
)
from .dataset import (
	Dataset, FewShotDataset, FormattingDataset, FuncDataset, GPTDataset, IndexDataset, InputOutputDataset, IntDataset, ListDataset, NondeterministicDataset, SumDataset
)
# from .error_analysis import (
# 	analyze_errors,
# 	get_value_dict,
# 	match_templates,
# 	filter_templates,
# 	print_templates,
# )
from .form_lib import (
	space_separate,
)
from .gpt import GPT, read_cache
from .util import (
	count_tokens,
	permute,
)

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

def run_task(argv, formatted, n_train, n_test, tag=''):
	completion_kwargs = {
		'temperature': 0, 
		'engine': 'davinci', 
		'max_tokens': 20, 
		# 'staged': True, 
		'stop': '\n',
		'logprobs': 100,
		'echo': True,
	}
	mock = 'submit' not in argv
	cache = read_cache()
	gpt = GPT(cache, mock)

	x = FuncDataset(formatted, lambda _: _[-1][0])
	y = FuncDataset(formatted, lambda _: _[-1][1])
	prompt = InputOutputDataset(formatted)  # type: str
	n_tokens = FuncDataset(prompt, count_tokens)
	response = GPTDataset(prompt, gpt, completion_kwargs)
	response_prompt = SumDataset([response, prompt])
	pred = FuncDataset(response_prompt, lambda _: get_completion_s(_[0], completion_kwargs, _[1]))  # type: str
	correct = FuncDataset(SumDataset([pred, y]), lambda _: _[0] == _[1] if _[0] is not None else None)  # type: Optional[bool]
	ppl = FuncDataset(response_prompt, lambda _: get_ppl_s(_[0], completion_kwargs, _[1]))  # type: Optional[str]
	incorrect_indices = ListDataset([i for i, val in enumerate(correct) if val == False])  # type: Optional[float]
	top_logprobs = FuncDataset(response_prompt, lambda _: get_top_logprobs_s(_[0], completion_kwargs, _[1], completion_only=False))  # type: Optional[List[float]]
	# log.info([x for x in incorrect_indices])
	# def analyze_templates(idx):
	# 	_sample = sample[idx]
	# 	_pred = pred[idx]
	# 	_x = x[idx]
	# 	tgt_form_func = lambda _: permute(_, _sample[-1][1])
	# 	value_dict = get_value_dict(TODO, [tgt_form_func])
	# 	templates = match_templates(_pred, value_dict)
	# 	print_templates(templates, None, _pred, _x)
	# 	templates_by_name = list(map(lambda x1: list(map(lambda x2: list(map(lambda x3: x3[0], x2)), x1)), templates))
	# 	return templates_by_name
	# templates = FuncDataset(incorrect_indices, analyze_templates)
	# rows = SumDataset([x, pred, y, correct, ppl,], keys=['x', 'pred', 'y', 'correct', 'ppl',])
	rows = SumDataset(
		[x, pred, y, correct, ppl, prompt, top_logprobs, response,], 
		keys=['x', 'pred', 'y', 'correct', 'ppl', 'prompt', 'top_logprobs', 'response',])
	# for _, i in zip(tqdm(range(len(rows))), range(len(rows))):
	# 	rows[i]
	# _rows = [r for _, r in zip(tqdm(range(len(rows))), rows)]
	_rows = [r for r in rows]
	log.info(max([_ for _ in n_tokens]))
	df = pd.DataFrame(_rows) # , columns=column_names)
	# df.fillna(value=pd.np.nan)
	# _templates = [None for _ in range(len(_rows))]
	# for i, template in zip(incorrect_indices, templates):
	# 	_templates[i] = template
	# df = df.assign(templates=_templates)
	# dataset = response
	# for i, batch in enumerate(dataset):
	# 	if i >= 5:
	# 		break
	# 	print(batch)
	# log.info(len(_rows))
	# # print(_rows)
	# # for _, r in zip(tqdm(_rows), _rows):
	# for r in _rows:
	# 	# print(r['prompt'])
	# 	log.info({k: v for k, v in r.items() if k not in ['prompt', 'top_logprobs', 'response']})
	log.info(df.correct.mean())
	log.info(df.ppl.mean())
	# try:
	# 	_top_logprobs = np.exp(-np.stack(df.top_logprobs.values).mean(axis=0))
	# 	# print(_top_logprobs[0].shape)
	# 	# log.info(_top_logprobs.mean(axis=0).shape)
	# 	# log.info(_top_logprobs)
	# 	# log.info(df.top_logprobs)
	# 	# n_tok_per = 2 + len(get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt']))
	# 	# tokens = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'])
	# 	# log.info(tokens)
	# 	# toks = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'])
	# 	tokens = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'], False)
	# 	n_tok_per = tokens[1:].index('Input') + 1
	# 	log.info(n_tok_per)
	# 	# if 'dates' in tag:
	# 	# 	import pdb; pdb.set_trace()
	# 	for i in range(n_tok_per):
	# 		fig = plt.figure(figsize=(15, 15))
	# 		# log.info(tokens[i::n_tok_per][:25])
	# 		plt.axhline(y=1., color='b', linestyle='-')
	# 		if i == 0:
	# 			ys = _top_logprobs[n_tok_per-1::n_tok_per]
	# 			xs = range(1,1+len(ys))
	# 		else:
	# 			ys = _top_logprobs[(i-1)::n_tok_per]
	# 			xs = range(len(ys))
	# 		plt.scatter(x=xs, y=ys)
	# 		bottom, top = plt.ylim()
	# 		top = max(top, 10)
	# 		plt.ylim(0, top)
	# 		plt.title(' '.join(tokens[i::n_tok_per][:25]))
	# 		plt.savefig(f'outputs/top_logprobs{tag}_{i}.png')
	# 		# plt.savefig(f'outputs/top_logprobs_median{tag}_{i}.png')
	# 		plt.clf()
	# 		plt.close('all')
	# 		# log.info(len(ys))  # n_train + 1
	# except Exception as e:
	# 	pass

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
	run_task(argv, formatted, n_train, n_test, '_dates')

def reverse(argv, n_train=100, n_test=5, n=5):
	content_dataset = RandomDistinctAlphaCharsDataset(n, offset=0)
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		lambda _: space_separate(_), 
		lambda _: space_separate(_[::-1]),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_reverse_n-{n}')

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
		'temperature': 0, 
		'engine': 'davinci', 
		'max_tokens': 20, 
		# 'staged': True, 
		'stop': '\n',
		'logprobs': 100,
		'echo': True,
	}
	mock = 'submit' not in argv
	cache = read_cache()
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
		'temperature': 0, 
		'engine': 'davinci', 
		'max_tokens': 20, 
		# 'staged': True, 
		'stop': '\n',
		'logprobs': 100,
		'echo': True,
	}
	mock = 'submit' not in argv
	cache = read_cache()
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

