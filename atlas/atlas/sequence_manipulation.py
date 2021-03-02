import sys, os
from collections import defaultdict, OrderedDict
import itertools
import json
import logging; log = logging.getLogger(__name__)
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
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
	get_completion_logprobs_s,
)
from .content_lib import (
	random_distinct_alpha_chars,
	random_distinct_chars,
	random_permutation,
	random_word_length_5,
)
from .dataset import (
	Dataset, FewShotDataset, FormattingDataset, FuncDataset, GPTDataset, IdentityDataset, IndexDataset, InputOutputDataset, IntDataset, ListDataset, NondeterministicDataset, ProductDataset, ConcatDataset
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
from .gpt import GPT3, get_cache
from .schema import (
	create_date_schema,
)
from .util import (
	count_tokens,
	make_immutable,
	permute,
	set_seed,
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

def run_task(argv, formatted, n_train, n_test, tag='', visualize: bool = False, 
		sample: Dataset = None, value_dict_func: Callable = None, index: int = -1,
		# io_format_args: Dict = {}, cache_args: List[Any] = []):
		io_format_args: Dict = {'include_y': True}, cache_args: List[Any] = [],
		# io_format_args: Dict = {'include_y': True, 'suffix': '\n'}, cache_args: List[Any] = [],
		engine: str = 'davinci',
		offset=0,
		max_n_train=None,
		datasets={},
		n_skip=1,
		):
	completion_kwargs = {
		'staged': True,
		# 'staged': False,
		'temperature': 0, 
		'engine': engine, 
		'max_tokens': 0, 
		# 'max_tokens': 20, 
		# 'staged': True, 
		'stop': '\n',
		'logprobs': 100,
		'echo': True,
	}

	completion_kwargs2 = {
		'staged': True,
		# 'staged': False,
		'temperature': 0, 
		'engine': engine, 
		# 'max_tokens': 0, 
		'max_tokens': 20, 
		# 'staged': True, 
		'stop': '\n',
		'logprobs': 100,
		# 'echo': True,
	}

	mock = 'submit' not in argv
	cache = get_cache(*cache_args)
	gpt = GPT3(cache, mock)

	x = FuncDataset(formatted, lambda _: _[index][0])
	y = FuncDataset(formatted, lambda _: _[index][1] + '\n')
	prompt = InputOutputDataset(formatted, **io_format_args)  # type: str
	n_tokens = FuncDataset(prompt, count_tokens)
	response = GPTDataset(prompt, gpt, completion_kwargs)
	# response_prompt = ConcatDataset([response, prompt])
	# pred = FuncDataset(response_prompt, lambda _: get_completion_s(_[0], completion_kwargs, _[1]))  # type: str
	# pred = FuncDataset(response_prompt, 
	#   lambda _: get_completion_logprobs_s(_[0], completion_kwargs, _[1]))  # type: str
	# correct = FuncDataset(ConcatDataset([pred, y]), lambda _: _[0] == _[1] if _[0] is not None else None)  # type: Optional[bool]
	# ppl = FuncDataset(response_prompt, lambda _: get_ppl_s(_[0], completion_kwargs, _[1]))  # type: Optional[str]
	# top_logprobs = FuncDataset(response_prompt, lambda _: get_top_logprobs_s(_[0], completion_kwargs, _[1], completion_only=False))  # type: Optional[List[float]]
	# log.info(prompt[0][-50:])
	# log.info([x for x in incorrect_indices])

	prompt_x = InputOutputDataset(formatted, **{**io_format_args, **{'include_y': False}})
	# pred2 = FuncDataset(ConcatDataset([response, prompt_x]), 
	#   lambda _: get_completion_logprobs_s(_[0], completion_kwargs, _[1]))  # type: str
	# correct2 = FuncDataset(ConcatDataset([pred2, y]), lambda _: _[0] == _[1] if _[0] is not None else None)  # type: Optional[bool]
	# log.info(prompt[0])
	# # log.info(prompt[120])
	responses = []
	for _, i in zip(tqdm(range(len(response))), range(len(response))):
		responses.append(response[i])
		# c = correct2[i]
		# if not c:
		#   log.info(f'Incorrect: pred={pred2[i]}, y={y[i]}, idx={i}')
			# sys.exit()
	# responses = [response[i] for i in range(len(response))]
	response_frozen = ListDataset(responses)
	# response_prompt.datasets[0] = response_frozen
	response_prompt2 = ConcatDataset([response_frozen, prompt_x])

	output_fname = f'outputs/results{tag}.csv'

	if os.path.isfile(output_fname):
		df = pd.read_csv(output_fname)
	else:
		df = None
	# for n_train2 in range(0, n_train+1): # range(1, 2): # 0, n_train+1): # 0, n_train+1): # n_train+5, 5):
	max_n_train = max_n_train or n_train+1
	# max_n_train = 5+5
	log.info(max_n_train)
	# if engine != 'davinci':
	# 	max_n_train = offset
	for n_train2 in range(offset, max_n_train, n_skip): # range(1, 2): # 0, n_train+1): # 0, n_train+1): # n_train+5, 5):
		if n_train2 == 0:
			continue
		partial = FuncDataset(sample, lambda _: _[:n_train2+1])
		formatted.dataset = partial
		formatted.getitem.cache_clear()
		prompt.getitem.cache_clear()
		response.getitem.cache_clear()
		# response_prompt.getitem.cache_clear()
		# pred.getitem.cache_clear()
		# correct.getitem.cache_clear()
		# prompt_x.getitem.cache_clear()
		x.getitem.cache_clear()
		y.getitem.cache_clear()
		response_prompt2.getitem.cache_clear()
		# FuncDataset.getitem.cache_clear()
		# InputOutputDataset.getitem.cache_clear()
		# formatted.__getitem__.cache_clear()
		prompt_x.__getitem__.cache_clear()
		# y.__getitem__.cache_clear()
		# response_prompt2.__getitem__.cache_clear()
		# FuncDataset.__getitem__.cache_clear()
		# InputOutputDataset.__getitem__.cache_clear()
		# log.info(prompt_x[0])
		Dataset.__getitem__.cache_clear()
		# log.info(prompt[0])
		# log.info(x[0])
		# log.info(y[0])
		# log.info(pred[0])
		# log.info(correct[0])
		# pred2 = FuncDataset(response_prompt2, 
		# 	lambda _: get_completion_logprobs_s(_[0], completion_kwargs, _[1], strip_whitespace=['left']))  # type: str
		if n_train2 < n_train:
			pred2 = FuncDataset(response_prompt2, 
				lambda _: get_completion_logprobs_s(_[0], completion_kwargs, _[1], strip_whitespace=['left']))  # type: str
		else:
			pred2 = FuncDataset(response_prompt2, 
				lambda _: (get_completion_logprobs_s(_[0], completion_kwargs, _[1], strip_whitespace=['left']) or '') + '\n')  # type: str
		# # log.info(response_frozen[0]['choices'][0]['text'])
		# correct2 = FuncDataset(ConcatDataset([pred2, y]), lambda _: _[0] == _[1] if _[0] is not None else None)  # type: Optional[bool]
		correct2 = FuncDataset(ConcatDataset([pred2, y]), lambda _: _[0] == _[1] if _[0].strip() else None)  # type: Optional[bool]
		ppl2 = FuncDataset(response_prompt2, lambda _: get_ppl_s(_[0], completion_kwargs, _[1]))  # type: Optional[str]
		# log.info(prompt_x[0])
		# log.info(pred2[0])
		# log.info(x[0])
		# log.info(y[0])
		# log.info(correct2[0])
		# log.info(ppl2[0])

		# _correct = [correct[i] for i in range(len(correct)) if correct[i] is not None]
		# if len(_correct) > 0:
		#   score = sum(_correct)
		#   # for i in range(len(correct)):
		#   #   # log.info((prompt_x[i],pred2[i],y[i],correct[i]))
		#   #   log.info((pred2[i],y[i],correct[i]))
		#   log.info(f'Score: {score}/{len(_correct)} = {100.*score/len(_correct):.2f}%; n_train={n_train2}')

		_correct2 = [correct2[i] for i in range(len(correct2)) if correct2[i] is not None]
		if len(_correct2) > 0:
			score = sum(_correct2)
			# for i in range(len(correct2)):
			#   # log.info((prompt_x[i],pred2[i],y[i],correct2[i]))
			#   log.info((pred2[i],y[i],correct2[i]))
			log.info(f'Score: {score}/{len(_correct2)} = {100.*score/len(_correct2):.2f}%; n_train={n_train2}')

		incorrect_indices = ListDataset([i for i, val in enumerate(correct2) if val == False])  # type: Optional[float]
		# incorrect_indices = ListDataset([i for i, (val, p) in enumerate(zip(correct2, pred2)) if val == False and p.startswith('-')])  # type: Optional[float]
		log.info(len(incorrect_indices))

		if engine == 'davinci':
			response2 = GPTDataset(prompt_x, gpt, completion_kwargs2)
			pred3 = FuncDataset(response2, lambda _: get_completion_s(_, completion_kwargs2))  # type: str
			_pred3 = [pred3[idx] for idx in incorrect_indices]
			log.info(len(list(filter(lambda _: _ is not None, _pred3))))


		# # log.info('\n'.join(filter(lambda _: _ is not None, _pred3)))
		# for idx in incorrect_indices:
		# 	# Get the true completion
		# 	# pred3 = FuncDataset(response_prompt2, 
		# 	# 	lambda _: get_completion_logprobs_s(_[0], completion_kwargs2, _[1], strip_whitespace=['left']))  # type: str
		# 	# log.info(response_frozen[0]['choices'][0]['text'])
		# 	# correct3 = FuncDataset(ConcatDataset([pred3, y]), lambda _: _[0] == _[1] if _[0] is not None else None)  # type: Optional[bool]
		# 	log.info(prompt_x[idx])
		# 	log.info(x[idx])
		# 	log.info(pred2[idx])
		# 	log.info(pred3[idx])
		# 	log.info(y[idx])
		# 	break

		# # print(datasets.keys())
		# # print(len(datasets) > 0)
		# if len(datasets) > 0:
		# 	# perturb
		# 	year = datasets['year']
		# 	month = datasets['month']
		# 	day = datasets['day']
		# 	content_dataset = datasets['content_dataset']
		# 	new_year = IntDataset(1970, 2020, offset=3)
		# 	new_month = IntDataset(1, 12+1, offset=4)
		# 	new_day = IntDataset(1, 28+1, offset=5)

		# 	new_content_dataset = ConcatDataset([new_year, month, day])

		# 	# new_prompt_x = InputOutputDataset(formatted, **{**io_format_args, **{'include_y': False}})
		# 	def perturb_sample(sample, new_sample):
		# 		# log.info(sample[-2:])
		# 		# log.info(new_sample[-2:])
		# 		return np.concatenate([
		# 			sample[:-1],
		# 			np.expand_dims(new_sample[-1], 0)
		# 		], axis=0)
		# 	# new_content_dataset = FuncDataset(
		# 	# 	ConcatDataset([content_dataset, new_content_dataset]), 
		# 	# 	lambda _: perturb_content(_)
		# 	# )  # type: str
		# 	new_sample = FewShotDataset(new_content_dataset, n_train=n_train, n_test=n_test)
		# 	partial = FuncDataset(sample, lambda _: _[:n_train2+1])
		# 	new_partial = FuncDataset(new_sample, lambda _: _[:n_train2+1])
		# 	perturbed_sample = FuncDataset(
		# 		ConcatDataset([partial, new_partial]), 
		# 		lambda _: perturb_sample(*_)
		# 	)  # type: str

		# 	# sample.dataset = new_content_dataset
		# 	# sample.getitem.cache_clear()
		# 	formatted.dataset = perturbed_sample
		# 	formatted.getitem.cache_clear()
		# 	prompt.getitem.cache_clear()
		# 	response.getitem.cache_clear()
		# 	x.getitem.cache_clear()
		# 	y.getitem.cache_clear()
		# 	response_prompt2.getitem.cache_clear()
		# 	response2.getitem.cache_clear()
		# 	pred3.getitem.cache_clear()
		# 	prompt_x.getitem.cache_clear()
		# 	Dataset.__getitem__.cache_clear()
		# 	# log.info(sample[idx])
		# 	# log.info(new_sample[idx])
		# 	# log.info(perturbed_sample[idx])
		# 	# log.info(formatted[idx])
		# 	log.info(prompt_x[idx])
		# 	# log.info(pred2[idx])
		# 	log.info(x[idx])
		# 	log.info(y[idx])
		# 	# log.info(correct2[idx])
		# 	# log.info(ppl2[idx])

		# 	response2 = GPTDataset(prompt_x, gpt, completion_kwargs2)
		# 	pred3 = FuncDataset(response2, lambda _: get_completion_s(_, completion_kwargs2))  # type: str
		# 	_pred3 = [pred3[idx] for idx in incorrect_indices]

		def analyze_templates(idx):
		  # _sample = sample[idx]
		  _pred = pred3[idx].rstrip()
		  _x = x[idx]
		  value_dict = value_dict_func(idx, partial, formatted)
		  templates = match_templates(_pred, value_dict)
		  min_length = min(map(len, templates))
		  if 'unnatural_addition_2_digit' in tag:
			  min_length = 1
		  templates = list(filter(lambda _: len(_) == min_length, templates))
		  if idx < 10:
			  print(json.dumps(value_dict, indent=4))
			  print_templates(templates, None, _pred, _x)
		  templates_by_name = list(map(lambda x1: list(map(lambda x2: list(map(lambda x3: x3[0], x2)), x1)), templates))
		  return templates_by_name
		templates = FuncDataset(incorrect_indices, analyze_templates)
		
		cur_rows = ConcatDataset([x, pred2, y, correct2, ppl2,], keys=['x', 'pred', 'y', 'correct', 'ppl',])
		_rows = [r for r in cur_rows]
		df_cur = pd.DataFrame(_rows) # , columns=column_names)
		df_cur['n_train'] = n_train2
		df_cur['engine'] = engine
		df_cur.fillna(value=pd.np.nan)

		actual_preds = [pred2[_] for _ in range(len(pred2))]
		if engine == 'davinci':
			for idx in incorrect_indices:
				actual_preds[idx] = pred3[idx]
			df_cur = df_cur.assign(pred=actual_preds)

		if value_dict_func is not None: # and 'templates' not in df_cur.keys():
		  _templates = [None for _ in range(len(df_cur))]
		  if engine == 'davinci':
		    for i, template in zip(incorrect_indices, templates):
		      _templates[i] = template
		  df_cur = df_cur.assign(templates=_templates)

		if df is None:
		  df = df_cur
		else:
		  df = pd.concat([df, df_cur], sort=False)
		  keys_for_comparison = ['x', 'pred', 'y', 'correct', 'ppl', 'engine',]
		  # df['is_duplicate'] = df[keys_for_comparison].duplicated()
		  # df = df[~df['is_duplicate']]
		df.to_csv(output_fname)
		
	formatted.dataset = sample

	# if False: # os.path.isfile(output_fname):
	#   df = pd.read_csv(output_fname)
	# else:
	#   # rows = ConcatDataset(
	#   #   [x, pred, y, correct,], 
	#   #   keys=['x', 'pred', 'y', 'correct',])
	#   rows = ConcatDataset(
	#       # [x, pred, y, correct, ppl, top_logprobs,], 
	#       # keys=['x', 'pred', 'y', 'correct', 'ppl', 'top_logprobs',])
	#       [x, pred, y, correct, ppl, prompt, top_logprobs, response,], 
	#       keys=['x', 'pred', 'y', 'correct', 'ppl', 'prompt', 'top_logprobs', 'response',])
	#   # for _, i in zip(tqdm(range(len(rows))), range(len(rows))):
	#   #   rows[i]
	#   # _rows = [r for _, r in zip(tqdm(range(len(rows))), rows)]
	#   _rows = [r for r in rows]
	#   # _token_counts = [_ for _ in n_tokens]
	#   # log.info('Token counts: %s' % str(_token_counts))
	#   # log.info('Max token count: %d' % max(_token_counts))
	#   # log.info('Total token count: %d' % sum(_token_counts))
	#   # log.info('Sample token count: %s' % n_tokens[0])
	#   # log.info(len(_rows))
	#   # # print(_rows)
	#   # # for _, r in zip(tqdm(_rows), _rows):
	#   for r in _rows:
	#       r['n_train'] = n_train

	#   for r in _rows[:5]:
	#       # print(r['prompt'])
	#       log.info({k: v for k, v in r.items() if k not in ['prompt', 'top_logprobs', 'response']})
	#   df = pd.DataFrame(_rows) # , columns=column_names)
	#   df.to_csv(output_fname)

	# df.fillna(value=pd.np.nan)
	# if value_dict_func is not None and 'templates' not in df.keys():
	#   _templates = [None for _ in range(len(df))]
	#   for i, template in zip(incorrect_indices, templates):
	#       _templates[i] = template
	#   df = df.assign(templates=_templates)
	#   df.to_csv(output_fname)
	# if 'n_train' not in df.keys():
	#   df['n_train'] = n_train
	#   df.to_csv(output_fname)
	# dataset = response
	# for i, batch in enumerate(dataset):
	#   if i >= 5:
	#       break
	#   print(batch)
	# try:
	#   log.info(df)
	#   log.info('Score: %.2f' % df.correct.mean())
	#   log.info('Avg ppl: %.2f' % df.ppl.mean())
	# except Exception as e:
	#   log.warn(e)

	# if visualize and 'dates_unnatural_content' not in tag:
	# 	try:
	# 		_top_logprobs = np.exp(-np.stack(df.top_logprobs.values).mean(axis=0))
	# 		# print(_top_logprobs[0].shape)
	# 		# log.info(_top_logprobs.mean(axis=0).shape)
	# 		# log.info(_top_logprobs)
	# 		# log.info(df.top_logprobs)
	# 		# n_tok_per = 2 + len(get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt']))
	# 		# tokens = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'])
	# 		# log.info(tokens)
	# 		# toks = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'])
	# 		tokens = get_top_tokens_s(_rows[0]['response'], completion_kwargs, _rows[0]['prompt'], False)
	# 		n_tok_per = tokens[1:].index('Input') + 1
	# 		log.info(n_tok_per)
	# 		# if 'dates' in tag:
	# 		#   import pdb; pdb.set_trace()
	# 		for i in range(n_tok_per):
	# 			_ = plt.figure(figsize=(15, 15))
	# 			# log.info(tokens[i::n_tok_per][:25])
	# 			plt.axhline(y=1., color='b', linestyle='-')
	# 			if i == 0:
	# 				ys = _top_logprobs[n_tok_per-1::n_tok_per]
	# 				xs = range(1,1+len(ys))
	# 			else:
	# 				ys = _top_logprobs[(i-1)::n_tok_per]
	# 				xs = range(len(ys))
	# 			plt.scatter(x=xs, y=ys)
	# 			bottom, top = plt.ylim()
	# 			top = max(top, 10)
	# 			plt.ylim(0, top)
	# 			plt.title(' '.join(tokens[i::n_tok_per][:25]))
	# 			plt.savefig(f'outputs/top_logprobs{tag}_{i}.png')
	# 			# plt.savefig(f'outputs/top_logprobs_median{tag}_{i}.png')
	# 			plt.clf()
	# 			plt.close('all')
	# 			# log.info(len(ys))  # n_train + 1
	# 	except Exception as e:
	# 		log.error(e)

	cost = gpt.calculate_cost()
	if cost:
		log.info('This request will cost %d tokens (including completion)' % cost)
		k = 'y'
		k = None
		if k == 'y':
		  log.warn('Submitting queries without confirmation!')
		gpt.run_staged_queries(k)

def dates(argv, n_train=15, n_test=5, engine='davinci'):
	year = IntDataset(1970, 2020, offset=0)
	month = IntDataset(1, 12+1, offset=1)
	day = IntDataset(1, 28+1, offset=2)

	content_dataset = ConcatDataset([year, month, day])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda x: '/'.join(map(lambda n: f'{n:02d}', x)), 
		lambda x: '-'.join(map(lambda n: f'{n:02d}', x)),
		lambda x: '!'.join([''] + list(map(lambda n: f'{n:02d}', permute(x, [1,2,0]))) + ['']),
		map=True,
	)
	cache_args = ['cache_GPT3_dates.jsonl']
	datasets = {
		'year': year,
		'month': month,
		'day': day,
		'content_dataset': content_dataset,
	}
	run_task(argv, formatted, n_train, n_test, f'_dates_n_train-{n_train}', datasets=datasets, sample=sample, value_dict_func=get_value_dict_dates, cache_args=cache_args, engine=engine)

def dates_natural_format(argv, n_train=15, n_test=5):
	year = IntDataset(1970, 2020, offset=0)
	month = IntDataset(1, 12+1, offset=1)
	day = IntDataset(1, 28+1, offset=2)

	content_dataset = ConcatDataset([year, month, day])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda x: '/'.join(map(lambda n: f'{n:02d}', x)), 
		lambda x: '-'.join(map(lambda n: f'{n:02d}', x)),
		lambda x: '/'.join(map(lambda n: f'{n:02d}', permute(x, [1,2,0]))),
		map=True,
	)
	cache_args = ['cache_GPT3_dates.jsonl']
	run_task(argv, formatted, n_train, n_test, f'_dates_natural_format_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_dates, cache_args=cache_args)

def dates_unnatural_content(argv, n_train=15, n_test=5, n_skip=1):
	year = IntDataset(10_000, 100_000, offset=0)
	month = IntDataset(10_000, 100_000, offset=1)
	day = IntDataset(10_000, 100_000, offset=2)

	content_dataset = ConcatDataset([year, month, day])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda x: '/'.join(map(lambda n: f'{n:02d}', x)), 
		lambda x: '-'.join(map(str, x)),
		lambda x: '!'.join([''] + list(map(str, permute(x, [1,2,0]))) + ['']),
		map=True,
	)
	cache_args = ['cache_GPT3_dates.jsonl']
	datasets = {
		'year': year,
		'month': month,
		'day': day,
		'content_dataset': content_dataset,
	}
	run_task(argv, formatted, n_train, n_test, f'_dates_unnatural_content_n_train-{n_train}', n_skip=n_skip, datasets=datasets, sample=sample, value_dict_func=get_value_dict_dates, cache_args=cache_args)

def dates_unnatural_content_3_digit(argv, n_train=15, n_test=5):
	year = IntDataset(100, 1_000, offset=0)
	month = IntDataset(100, 1_000, offset=1)
	day = IntDataset(100, 1_000, offset=2)

	content_dataset = ConcatDataset([year, month, day])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda x: '/'.join(map(lambda n: f'{n:02d}', x)), 
		lambda x: '-'.join(map(str, x)),
		lambda x: '!'.join([''] + list(map(str, permute(x, [1,2,0]))) + ['']),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'dates_unnatural_content_3_digit_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_dates)

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
	content_dataset = ConcatDataset([summand1, summand2])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda _: space_separate(_), 
		# lambda _: space_separate(_[::-1]),
		lambda _: f'{_[0]} + {_[1]}', 
		lambda _: str(sum(_)),
		map=True,
	)
	run_task(argv, formatted, n_train, n_test, f'_addition_3_digit_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_numbers)

def unnatural_addition_2_digit(argv, n_train=80, n_test=5, n=5, sep1=' + ', sep2=' = ', prefix=None, offset=0, max_n_train=None):
	summand1 = IntDataset(10, 100, offset=0)
	summand2 = IntDataset(10, 100, offset=1)
	content_dataset = ConcatDataset([summand1, summand2])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda _: space_separate(_), 
		# lambda _: space_separate(_[::-1]),
		lambda _: f'{_[0]}{sep1}{_[1]}', 
		lambda _: str(sum(_)),
		map=True,
	)
	io_format_args = {
		'x_label': '',
		'y_label': '',
		'intra_separator': '',
		'x_y_separator': sep2,
		'include_y': True,
		'prefix': prefix,
	}
	cache_args = ['cache_GPT3_unnatural_addition_lps-1.jsonl']
	if n_train > 300:
		cache_args = ['cache_GPT3_unnatural_addition_long.jsonl']
	# cache_args = []
	if offset:
		cache_args = [cache_args[0].replace('_lps-1.jsonl',f'_offset-{offset}_lps-1.jsonl')]
	if max_n_train is not None:
		cache_args = [cache_args[0].replace('_lps-1.jsonl',f'_max_n_train-{max_n_train}_lps-1.jsonl')]
	run_task(argv, formatted, n_train, n_test, f'_unnatural_addition_2_digit_n_train-{n_train}', max_n_train=max_n_train, offset=offset, sample=sample, value_dict_func=get_value_dict_numbers, io_format_args=io_format_args, cache_args=cache_args)

def unnatural_addition_2_digit_qa(argv, n_train=80, n_test=5, n=5, sep1=' + ', sep2=' = ', prefix=None):
	summand1 = IntDataset(10, 100, offset=0)
	summand2 = IntDataset(10, 100, offset=1)
	content_dataset = ConcatDataset([summand1, summand2])
	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample, 
		# lambda _: space_separate(_), 
		# lambda _: space_separate(_[::-1]),
		lambda _: f'{_[0]}{sep1}{_[1]} = ?', 
		lambda _: str(sum(_)),
		map=True,
	)
	io_format_args = {
		'x_label': 'Q',
		'y_label': 'A',
		'intra_separator': ': ',
		'x_y_separator': '\n',
		'include_y': True,
		'prefix': prefix,
	}
	cache_args = ['cache_GPT3_unnatural_addition_qa.jsonl']
	# cache_args = []
	run_task(argv, formatted, n_train, n_test, f'_unnatural_addition_2_digit_qa_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_numbers, io_format_args=io_format_args, cache_args=cache_args)

def identity(argv, n_train=5, n_test=5, n=5, sep=', ', prefix=None, engine='davinci'):
	_content = []
	cntr = 0
	for comb_idx, comb in enumerate(itertools.combinations('abcdefgh', n)):
		for perm in itertools.permutations(comb):
			set_seed(cntr)
			c = np.concatenate([
				random_distinct_chars((1, n)) for _ in range(n_train)] + 
				[np.array(perm).reshape(1, n)
			])
			_content.append(c)
			cntr += 1
	sample = ListDataset(_content)
	# log.info(sample[0])
	# content_dataset = NondeterministicDataset(func=lambda _: list(random_distinct_chars(n)))
	# sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
	formatted = FormattingDataset(sample,  
		# lambda _: space_separate(_), 
		# lambda _: space_separate(_[::-1]),
		lambda _: sep.join(_), 
		lambda _: sep.join(_),
		map=True,
	)
	io_format_args = {
		# 'x_label': '',
		# 'y_label': '',
		# 'intra_separator': '',
		# 'x_y_separator': sep2,
		'include_y': True,
		# 'prefix': prefix,
	}
	cache_args = ['cache_GPT3_identity.jsonl']
	run_task(argv, formatted, n_train, n_test, f'_identity_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_chars, io_format_args=io_format_args, cache_args=cache_args, engine=engine)

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
		**{'Input': 'Input'},
		**{'Output': 'Output'},
	})
	return value_dict

def get_value_dict_numbers(idx, sample, formatted=None):
	_content = sample[idx][-1]
	summand1, summand2 = _content
	_sum = summand1 + summand2
	_diff = summand1 - summand2
	# tgt_form = formatted.tgt_form
	value_dict = OrderedDict({
		# **{f'cp{i}': str(cp) for i, cp in enumerate(_content)},
		**{
			'summand1': summand1,
			'summand2': summand2,
			'sum': _sum,
			'diff': _diff,
			'neg': '-',
			'\$': '\$',
			'zero': '0',
		},
		**{'<other>': r'\d+',},
		# **get_form_primitives(_content, [tgt_form]),
		**{'Input': 'Input'},
		**{'Output': 'Output'},
	})
	value_dict = add_neighbors_numeric(value_dict)
	return value_dict

def add_neighbors_numeric(base_value_dict):
	new_value_dict = {}

	for k, v in base_value_dict.items():
		new_value_dict[k] = str(v)

		if not isinstance(v, int):
			continue
		
		# if v > 0:
		# new_value_dict[f'{k}_minus-one'] = str(v-1)
		
		# new_value_dict[f'{k}_plus-one'] = str(v+1)

		v_str = str(v)

		for i in range(len(v_str)):
			if v_str[i] == '-':
				continue
			
			digit = int(v_str[i])
			ni = len(v_str)-1-i

			# if i < len(v_str)-1 and digit > 0: 
			#   new_value_dict[f'{k}_digit_{ni}_minus-one'] = v_str[:i] + str(digit-1) + v_str[i+1:]
			
			# if i < len(v_str)-1 and digit < 9: 
			#   new_value_dict[f'{k}_digit_{ni}_plus-one'] = v_str[:i] + str(digit+1) + v_str[i+1:]
			
			new_value_dict[f'<{k}_digit_{ni}_off>'] = v_str[:i] + '\d' + v_str[i+1:]

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
	sample = ConcatDataset([content, order])
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
	gpt = GPT3(cache, mock)

	content_dataset = RandomDistinctAlphaCharsDataset(1, offset=0)
	content = FewShotDataset(content_dataset, n_train=n_train, n_test=1)
	formatted = FuncDataset(content, lambda _: '\n'.join(map(lambda x: f'Input: {x[0]}', _)))
	prompt = formatted
	response = GPTDataset(prompt, gpt, completion_kwargs)
	response_prompt = ConcatDataset([response, prompt])
	top_logprobs = FuncDataset(response_prompt, lambda _: get_top_logprobs_s(_[0], completion_kwargs, _[1], completion_only=False))  # type: Optional[List[float]]
	dataset = formatted
	# for i, batch in enumerate(dataset):
	#   if i >= 5:
	#       break
	#   print(batch)
	tag = '_random_char'
	_top_logprobs = np.exp(top_logprobs[0])
	# log.info(top_logprobs[0])
	log.info(_top_logprobs)
	tokens = get_top_tokens_s(response[0], completion_kwargs, prompt[0], False)
	n_tok_per = tokens[1:].index('Input') + 1
	log.info(n_tok_per)
	# if 'dates' in tag:
	#   import pdb; pdb.set_trace()
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
	gpt = GPT3(cache, mock)

	content_dataset = IntDataset(low, high, offset=0)
	content = FewShotDataset(content_dataset, n_train=n_train, n_test=1)
	formatted = FuncDataset(content, lambda _: '\n'.join(map(lambda x: f'Input: {x}', _)))
	prompt = formatted
	response = GPTDataset(prompt, gpt, completion_kwargs)
	response_prompt = ConcatDataset([response, prompt])
	top_logprobs = FuncDataset(response_prompt, lambda _: get_top_logprobs_s(_[0], completion_kwargs, _[1], completion_only=False))  # type: Optional[List[float]]
	dataset = formatted
	# for i, batch in enumerate(dataset):
	#   if i >= 5:
	#       break
	#   print(batch)
	tag = f'_random_num_low-{low}_high-{high}'
	_top_logprobs = np.exp(top_logprobs[0])
	# log.info(top_logprobs[0])
	log.info(_top_logprobs)
	tokens = get_top_tokens_s(response[0], completion_kwargs, prompt[0], False)
	n_tok_per = tokens[1:].index('Input') + 1
	log.info(n_tok_per)
	# if 'dates' in tag:
	#   import pdb; pdb.set_trace()
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
	#   log.info(formatted[i])
		# log.info(sample[i])
		# log.info(len(sample[i]))
		# log.info(len(set(make_immutable(sample[i]))))
	#   # log.info(content_single[i])
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
	#   # log.info(content_single[i])
	# log.info(content[0])
	# log.info(content[1])
	run_task(argv, formatted, n_train, n_test, f'_calendar_2x2_exception_dummy_n_train-{n_train}')

# def unnatural_addition_pt_2(summand1, operator, summand2):
#   """
#   Args:
#       summand1, summand2 (int): [-10,10]
#   """
#   pass

# # boolean gates
# # natural
# concat
# order
# unary ops
# - classify, map
# binary ops
# - logical operators
# # products
#   AND
#   OR
#   XOR
#   yes no
#   True False
#   1 0
#   + -
#   @ #
#   # :) :(
# small PPL

def quick_tests():
	pass 


