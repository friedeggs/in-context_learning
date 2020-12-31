import sys, os
import abc
from copy import deepcopy
import functools
import itertools
import logging; log = logging.getLogger(__name__)
from collections import namedtuple
import numpy as np
import random
from termcolor import colored
from tqdm import tqdm
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .content_lib import (
	random_distinct_chars,
	random_permutation,
	random_word_length_5,
)
from .form_lib import (
	comma_separate,
	space_separate,
	io_format, 
)
from .gpt import get_cache
from .gpt2 import GPT2
from .util import (
	make_immutable, 
	is_iterable, 
	MAX_INT, 
	set_seed,
	plot,
	moving_average,
)

def vanilla_metropolis_hastings(t: int, x, T: Callable, Teval: Optional[Callable], p: Callable, return_prob=False):
	xprime = T(x)
	# if random.random() < p(xprime) * Teval(x, xprime) / (p(x) * Teval(xprime, x)):
	p_xprime = p(xprime)
	p_x = p(x)
	# p_xprime = 1 - p(xprime.fmt())
	# p_x = 1 - p(x.fmt())
	# log.info(x.fmt())
	# log.info(xprime.data[:,0,:])
	# log.info((t, p_xprime, p_x, min(1, p_xprime / p_x)))
	# if random.random() < p(xprime) / (p(x)):
	transition_ratio = Teval(x, xprime)
	prob = min(1, p_xprime / p_x * transition_ratio)
	if np.log(random.random()) <= np.log(prob):
		x = xprime
		p_x = p_xprime
	result = [x, p_x]
	if return_prob:
		result += [prob]
	return result

class Sample:
	def __init__(self, data, fmt_func):
		self.data = data
		self.fmt_func = fmt_func

	def fmt(self):
		return self.fmt_func(self.data)

class Task(abc.ABC):
	def __init__(self):
		pass 

	@abc.abstractmethod
	def sample(self):
		raise NotImplementedError 

	@abc.abstractmethod
	def perturb(self):
		raise NotImplementedError 

class Copy(Task):
	def __init__(self, n, n_train):
		self.n = n 
		self.n_train = n_train 

	def sample(self):
		return Sample(
			np.tile(np.concatenate([random_distinct_chars((1, 1, self.n)) for _ in range(self.n_train)]), (1,2,1)), 
			lambda _: io_format(_, transform=comma_separate, include_y=True))

	def perturb(self, x):
		x = deepcopy(x)
		# x.data[np.random.choice(x.data.shape[0]),:,np.random.choice(x.data.shape[-1])] = random_distinct_chars(1)[0]
		# return x
		# val = random_distinct_chars(1)[0]
		idx0 = np.random.choice(x.data.shape[0])
		idx1 = np.random.choice(x.data.shape[-1])
		cur_val = x.data[idx0,0]
		choices = [ch for ch in list('abcdefgh') if ch not in cur_val]
		# x.data[idx0,:,idx1] = np.random.choice(choices, 1, replace=False)[0]
		val = np.random.choice(choices, 1)[0]
		x.data[idx0,:,idx1] = val
		return x

	def perturb_eval(self, x, xprime):
		# return (1./self.n_train) * (1./self.n) * (1./26)
		# raise NotImplementedError
		return 1. # TODO 

def run_random_search(argv, n_train, T=140, engine='gpt2'):
	copy = Copy(5, n_train)

	# q1 = pass

	# completion_kwargs = {
	# 	# 'staged': True,
	# 	'temperature': 0, 
	# 	'engine': 'gpt2', 
	# 	'max_tokens': 10, 
	# 	'stop': ['\n','Output'],
	# 	'logprobs': 1,
	# 	# 'echo': True,
	# }

	completion_kwargs = {
		# 'staged': True,
		'temperature': 0, 
		'engine': engine, 
		'max_tokens': 0, 
		# 'stop': ['\n','Output'],
		'logprobs': 1,
		'echo': True,
	}

	prompt_fn = lambda _: io_format(_, transform=comma_separate)

	cache = get_cache('cache_gpt2.jsonl')
	mock = 'submit' not in argv
	gpt2 = GPT2(cache, mock)
	# p_failure = lambda _: 1. - 1./gpt2.ppl(_, completion_kwargs)
	p_failure = lambda _: 1. - 1./gpt2.ppl(_.fmt() + '\n', completion_kwargs, prefix=prompt_fn(_.data))
	samples = []
	probs = []
	unique_samples = set()
	for _, t in zip(tqdm(range(T)), range(T)):
		sample = copy.sample()
		p_x = p_failure(sample)
		samples.append(sample)
		probs.append(p_x)
		unique_samples.add(sample.fmt())
	log.info(len(unique_samples))
	vals = probs
	xs = np.arange(len(vals))
	ys = np.cumsum(vals)/np.arange(1,len(vals)+1)
	# plot(np.arange(len(ys)),ys,f'../outputs/random_search_n_train-{copy.n_train}.png', scatter_kwargs={'c':'r'})
	# log.info(ys[-1])
	log.info(f'Random search estimate: {ys[-1]*100.:.2f}%')

	completion_kwargs = {
		# 'staged': True,
		'temperature': 0, 
		# 'engine': 'gpt2-medium', 
		'engine': engine, 
		'max_tokens': 10, 
		'stop': ['\n','Output'],
		'logprobs': 1,
		# 'echo': True,
	}
	fmt_func = lambda _: io_format(_, transform=comma_separate, include_y=False)
	results = []
	for _, sample in zip(tqdm(samples), samples):
		prompt = fmt_func(sample.data)
		completion = gpt2.complete(prompt, completion_kwargs, return_response=False)
		y = ' ' + comma_separate(list(map(str, list(sample.data[-1][0])))) + '\n'
		# log.info(sample.fmt())
		# if completion != y:
		# 	log.info((completion, y)) #, completion == y))
		# else:
		results.append((prompt, completion, y))
	score = 0
	results = list(set(results))
	for _, completion, y in results:
		if completion == y:
			score += 1
	log.info(f'Random search score: {score}/{len(results)} = {100.*score/len(results):.2f}%')
	return xs, ys

def run_smarter_search(argv, n_train, T=1500, engine='gpt2'):
	copy = Copy(5, n_train)
	xt = copy.sample()
	# log.info(x0.fmt())

	# q1 = pass

	completion_kwargs = {
		# 'staged': True,
		'temperature': 0, 
		'engine': engine, 
		'max_tokens': 0, 
		# 'stop': ['\n','Output'],
		'logprobs': 1,
		'echo': True,
	}

	prompt_fn = lambda _: io_format(_, transform=comma_separate)

	cache = get_cache(f'cache_{engine}.jsonl')
	mock = 'submit' not in argv
	gpt2 = GPT2(cache, mock)
	p_failure = lambda _: 1. - 1./gpt2.ppl(_.fmt() + '\n', completion_kwargs, prefix=prompt_fn(_.data))
	# log.info(prompt_fn(xt.data))
	samples = []
	probs = []
	unique_prompts = set()
	unique_samples = []
	for _, t in zip(tqdm(range(T)), range(T)):
	# for t in range(T):
		while xt.fmt() in unique_prompts:
			xt, p = vanilla_metropolis_hastings(
				t,
				xt, 
				copy.perturb,
				copy.perturb_eval,
				p_failure,
			)
			samples.append(xt)
			probs.append(p)
		unique_prompts.add(xt.fmt())
		unique_samples.append(xt)
		# log.info(colored(xt.data[:,0,:], 'green'))
		# log.info(xt.fmt() + colored(gpt2.complete(xt.fmt(), completion_kwargs, False), 'magenta'))
	# samples = np.array(samples)
	# print(np.unique(samples, axis=0))
	# _samples = np.vstack(set(tuple(row.data[0,0,:].tolist()) for row in samples))
	# log.info(_samples)
	# log.info(len(samples))
	log.info(len(unique_samples))
	n_burn_in = 100 # 100 # 1000 # 100 # 0 # 1500
	n_skip = 5 # 10 # 5 # 1 # 10
	vals = probs[n_burn_in:][::n_skip]
	xs = np.arange(len(probs))
	xs = xs[n_burn_in:][::n_skip]
	ys = np.cumsum(vals)/np.arange(1,len(vals)+1)
	try:
		log.info(ys[-1])
	except Exception as e:
		pass

	completion_kwargs = {
		# 'staged': True,
		'temperature': 0, 
		# 'engine': 'gpt2-medium', 
		'engine': 'gpt2-large', 
		'max_tokens': 10, 
		'stop': ['\n','Output'],
		'logprobs': 1,
		# 'echo': True,
	}
	fmt_func = lambda _: io_format(_, transform=comma_separate, include_y=False)
	results = []
	# samples = list(set(samples))
	for _, sample in zip(tqdm(unique_samples), unique_samples):
		new_row = np.tile(random_distinct_chars((1, 1, copy.n)), (1,2,1))
		data = np.concatenate([new_row, sample.data])
		prompt = fmt_func(data)
		completion = gpt2.complete(prompt, completion_kwargs, return_response=False)
		y = ' ' + comma_separate(list(map(str, list(sample.data[-1][0])))) + '\n'
		# log.info(sample.fmt())
		# if completion != y:
		# 	log.info((completion, y)) #, completion == y))
		# else:
		results.append((prompt, completion, y))
	score = 0
	results = list(set(results))
	for _, completion, y in results[:T]:
		if completion == y:
			score += 1
	log.info(f'Score: {score}/{len(results)} = {100.*score/len(results):.2f}%')
	return xs, ys
	# return probs
	# plot(np.arange(len(ys)),ys,f'../outputs/vanilla_monte_carlo_n_train-{copy.n_train}.png')

def plot_searches(argv, n_train, T=1500, engine='gpt2'):
	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pyplot as plt
	output_file = f'../outputs/compare_search_n_train-{n_train}_engine-{engine}.png'
	xs = []
	ys = []
	for i in range(1):
		set_seed(i)
		x, y = run_smarter_search(argv, n_train, T=T, engine=engine)
		xs.append(x)
		ys.append(y)
	set_seed()
	x2, y2 = run_random_search(argv, n_train, T=T, engine=engine) # len(y))
	sz = 15
	_ = plt.figure(figsize=(sz, sz))
	for x, y, c in zip(xs, ys, list('rgbkc')):
		plt.scatter(x=x, y=y, c=c)
		plt.axhline(y=y[-1], color=c, linestyle='-')
	plt.scatter(x=x2, y=y2, c='m')
	plt.axhline(y=y2[-1], color='m', linestyle='-')
	mid = np.round(y2[-1]/0.1)*0.1
	plt.ylim(max(0,mid-.1), min(1,mid+.1))
	plt.savefig(output_file)

def run_chained_search_wrong(argv, n_train, T=2000):
	prompt_fn = lambda _: io_format(_, transform=comma_separate)
	mock = 'submit' not in argv
	engines = ['gpt2', 'gpt2-medium', 'gpt2-large']
	chains = {engine: {} for engine in engines}
	cache = {engine: get_cache(f'cache_{engine}.jsonl') for engine in engines}
	gpt2 = {engine: GPT2(cache[engine], mock) for engine in engines}
	completion_kwargs = {engine: {
		# 'staged': True,
		'temperature': 0, 
		'engine': engine, 
		'max_tokens': 0, 
		# 'stop': ['\n','Output'],
		'logprobs': 1,
		'echo': True,
	} for engine in engines}
	p_failure = {
		engine: lambda _: 1. - 1./gpt2[engine].ppl(_.fmt() + '\n', completion_kwargs[engine], prefix=prompt_fn(_.data))
		for engine in engines
	}

	copy = Copy(5, n_train)

	n_burn_in = 200 # 100 # 1000 # 100 # 0 # 1500
	n_skip = 5 # 10 # 5 # 1 # 10

	# perturb_funcs = {engine: None for engine in engines}
	transition_ratios = {engine: None for engine in engines}

	for engine_idx, engine in enumerate(engines):
		set_seed()
		xt = copy.sample()

		# def perturb(x):
		# 	if engine_idx == 0:
		# 		return copy.perturb(x)
		# 	xt = chains[engine]['xt']
		# 	for i in range(n_skip):
		# 		xt, p, p_sample = run_vanilla_metropolis_hastings(xt)
		# 	chains[engine]['xt'] = xt

		def transition_ratio(engine_idx):
			def func(x, xprime):
				"""Compute T(x|xprime) / T(xprime|x).
				"""
				if engine_idx == 0:
					return copy.perturb_eval(x, xprime)
				prev_engine = engines[engine_idx - 1]
				p_x = p_failure[engine](x)
				p_xprime = p_failure[engine](xprime)
				# log.info(engine_idx)
				# log.info(prev_engine)
				t_ratio = transition_ratios[prev_engine](x, xprime)
				# return min(1, p_xprime / p_x * t_ratio)
				return p_xprime / p_x * t_ratio
			return func

		# perturb_funcs[engine] = perturb
		transition_ratios[engine] = transition_ratio(engine_idx)

		# for _, t in zip(tqdm(range(n_burn_in),desc=f'burn_in {engine}'), range(n_burn_in)):
		# 	xt, p, p_sample = vanilla_metropolis_hastings(
		# 		t,
		# 		xt, 
		# 		copy.perturb,
		# 		transition_ratio,
		# 		p_failure[engine],
		# 		return_prob=True,
		# 	)
		chains[engine]['xt'] = xt

	engine = 'gpt2-large'
	xt = chains[engine]['xt']
	samples = []
	probs = []
	for _, t in zip(tqdm(range(T)), range(T)):
	# for t in range(T):
		xt, p, p_sample = vanilla_metropolis_hastings(
			t,
			xt, 
			copy.perturb,
			transition_ratios[engine],
			p_failure[engine],
			return_prob=True,
		)
		samples.append(xt)
		probs.append(p)

	vals = probs[n_burn_in:][::n_skip]
	xs = np.arange(len(probs))
	xs = xs[n_burn_in:][::n_skip]
	ys = np.cumsum(vals)/np.arange(1,len(vals)+1)
	log.info(ys[-1])
	return xs, ys

def run_chained_search(argv, n_train, T=2000, engine='gpt2-large'):
	prompt_fn = lambda _: io_format(_, transform=comma_separate)
	mock = 'submit' not in argv
	engines = ['gpt2', 'gpt2-medium', 'gpt2-large']
	chains = {engine: {} for engine in engines}
	cache = {engine: get_cache(f'cache_{engine}.jsonl') for engine in engines}
	gpt2 = {engine: GPT2(cache[engine], mock) for engine in engines}
	completion_kwargs = {engine: {
		# 'staged': True,
		'temperature': 0, 
		'engine': engine, 
		'max_tokens': 0, 
		# 'stop': ['\n','Output'],
		'logprobs': 1,
		'echo': True,
	} for engine in engines}
	p_failure = {
		idx: lambda _: 1. - 1./gpt2[engine].ppl(_.fmt() + '\n', completion_kwargs[engine], prefix=prompt_fn(_.data))
		for idx in range(len(engines))
	}
	engine_idx = engines.index(engine)
	p_func = p_failure[engine_idx] / p_failure[engine_idx - 1]

	copy = Copy(5, n_train)

	n_burn_in = 200 # 100 # 1000 # 100 # 0 # 1500
	n_skip = 5 # 10 # 5 # 1 # 10

	# perturb_funcs = {engine: None for engine in engines}
	transition_ratios = {engine: None for engine in engines}

	for engine_idx, engine in enumerate(engines):
		set_seed()
		xt = copy.sample()

		# def perturb(x):
		# 	if engine_idx == 0:
		# 		return copy.perturb(x)
		# 	xt = chains[engine]['xt']
		# 	for i in range(n_skip):
		# 		xt, p, p_sample = run_vanilla_metropolis_hastings(xt)
		# 	chains[engine]['xt'] = xt

		def transition_ratio(engine_idx):
			def func(x, xprime):
				"""Compute T(x|xprime) / T(xprime|x).
				"""
				if engine_idx == 0:
					return copy.perturb_eval(x, xprime)
				prev_engine = engines[engine_idx - 1]
				p_x = p_failure[engine](x)
				p_xprime = p_failure[engine](xprime)
				# log.info(engine_idx)
				# log.info(prev_engine)
				t_ratio = transition_ratios[prev_engine](x, xprime)
				# return min(1, p_xprime / p_x * t_ratio)
				return p_xprime / p_x * t_ratio
			return func

		# perturb_funcs[engine] = perturb
		transition_ratios[engine] = transition_ratio(engine_idx)

		for _, t in zip(tqdm(range(n_burn_in),desc=f'burn_in {engine}'), range(n_burn_in)):
			xt, p, p_sample = vanilla_metropolis_hastings(
				t,
				xt, 
				copy.perturb,
				transition_ratio,
				p_failure[engine],
				return_prob=True,
			)
		chains[engine]['xt'] = xt

	engine = 'gpt2-large'
	xt = chains[engine]['xt']
	samples = []
	probs = []
	for _, t in zip(tqdm(range(T)), range(T)):
	# for t in range(T):
		xt, p, p_sample = vanilla_metropolis_hastings(
			t,
			xt, 
			copy.perturb,
			transition_ratios[engine],
			p_failure[engine],
			return_prob=True,
		)
		samples.append(xt)
		probs.append(p)

	vals = probs[n_burn_in:][::n_skip]
	xs = np.arange(len(probs))
	xs = xs[n_burn_in:][::n_skip]
	ys = np.cumsum(vals)/np.arange(1,len(vals)+1)
	log.info(ys[-1])
	return xs, ys

def run_importance_sampling(argv, n_train, T=1500, source_engine='gpt2', target_engine='gpt2-large'):
	copy = Copy(5, n_train)
	xt = copy.sample()
	# log.info(x0.fmt())

	# q1 = pass

	completion_kwargs = {
		# 'staged': True,
		'temperature': 0, 
		'engine': source_engine, 
		'max_tokens': 0, 
		# 'stop': ['\n','Output'],
		'logprobs': 1,
		'echo': True,
	}

	prompt_fn = lambda _: io_format(_, transform=comma_separate)

	cache = get_cache('cache_gpt2.jsonl')
	mock = 'submit' not in argv
	gpt2 = GPT2(cache, mock)
	# p_failure = lambda _: 1. - 1./gpt2.ppl(_, completion_kwargs)
	p_failure = lambda _: 1. - 1./gpt2.ppl(_.fmt() + '\n', completion_kwargs, prefix=prompt_fn(_.data))
	samples = []
	probs = []
	unique_prompts = set()
	for _, t in zip(tqdm(range(T)), range(T)):
		sample = copy.sample()
		while sample.fmt() in unique_prompts:
			sample = copy.sample()
		p_x = p_failure(sample)
		samples.append(sample)
		probs.append(p_x)
		unique_prompts.add(sample.fmt())
	# log.info(len(unique_prompts))
	vals = probs
	xs = np.arange(len(vals))
	ys = np.cumsum(vals)/np.arange(1,len(vals)+1)
	# plot(np.arange(len(ys)),ys,f'../outputs/random_search_n_train-{copy.n_train}.png', scatter_kwargs={'c':'r'})
	log.info(f'{source_engine} estimate: {ys[-1]*100.:.2f}%')

	completion_kwargs = {
		# 'staged': True,
		'temperature': 0, 
		'engine': target_engine, 
		'max_tokens': 0, 
		# 'stop': ['\n','Output'],
		'logprobs': 1,
		'echo': True,
	}
	completion_kwargs_binary = {
		# 'staged': True,
		'temperature': 0, 
		# 'engine': 'gpt2-medium', 
		'engine': target_engine, 
		'max_tokens': 10, 
		'stop': ['\n','Output'],
		'logprobs': 1,
		# 'echo': True,
	}
	p_failure = lambda _: 1. - 1./gpt2.ppl(_.fmt() + '\n', completion_kwargs, prefix=prompt_fn(_.data))
	fmt_func = lambda _: io_format(_, transform=comma_separate, include_y=False)
	alpha = .5
	Talg = 0
	S = 0.
	S_score = 0.
	Z = (np.array(probs) ** alpha).mean()
	# log.info(Z)
	# log.info(probs)
	# results = []
	score = 0
	for _, sample, prob in zip(tqdm(samples), samples, probs):
		accept = random.random() < prob ** alpha
		if not accept:
			continue
		Talg += 1
		p = p_failure(sample)
		S += 1. * p / prob ** alpha
		# log.info(p)
		# log.info(1. * p / prob ** alpha)
		prompt = fmt_func(sample.data)
		completion = gpt2.complete(prompt, completion_kwargs_binary, return_response=False)
		y = ' ' + comma_separate(list(map(str, list(sample.data[-1][0])))) + '\n'
		# results.append((prompt, completion, y))
		S_score += 1. * (completion == y) / prob ** alpha
		score += (completion == y)
	estimate = Z * S / Talg
	score_estimate = Z * S_score / Talg
	log.info('Number of evaluations: %d' % Talg)
	log.info(f'Importance sampling estimate: {estimate*100.:.2f}%')
	log.info(f'Importance sampling score: {score}/{Talg} (= {100.*score/Talg:.2f}%) -> {score_estimate*100.:.2f}%')
	# log.info(estimate)
	# 	new_row = np.tile(random_distinct_chars((1, 1, copy.n)), (1,2,1))
	# 	data = np.concatenate([new_row, sample.data])
	# 	prompt = fmt_func(data)
	# 	completion = gpt2.complete(prompt, completion_kwargs, return_response=False)
	# 	y = ' ' + comma_separate(list(map(str, list(sample.data[-1][0])))) + '\n'
	# 	# log.info(sample.fmt())
	# 	# if completion != y:
	# 	# 	log.info((completion, y)) #, completion == y))
	# 	# else:
	# 	results.append((prompt, completion, y))
	# score = 0
	# results = list(set(results))
	# for _, completion, y in results[:T]:
	# 	if completion == y:
	# 		score += 1
	# log.info(f'Score: {score}/{len(results)} = {100.*score/len(results):.2f}%')
	# return xs, ys
	# return estimate
	# completion_kwargs = {
	# 	# 'staged': True,
	# 	'temperature': 0, 
	# 	# 'engine': 'gpt2-medium', 
	# 	'engine': 'gpt2-large', 
	# 	'max_tokens': 10, 
	# 	'stop': ['\n','Output'],
	# 	'logprobs': 1,
	# 	# 'echo': True,
	# }
	# fmt_func = lambda _: io_format(_, transform=comma_separate, include_y=False)
	# results = []
	# for _, sample in zip(tqdm(samples), samples):
	# 	prompt = fmt_func(sample.data)
	# 	completion = gpt2.complete(prompt, completion_kwargs, return_response=False)
	# 	y = ' ' + comma_separate(list(map(str, list(sample.data[-1][0])))) + '\n'
	# 	# log.info(sample.fmt())
	# 	# if completion != y:
	# 	# 	log.info((completion, y)) #, completion == y))
	# 	# else:
	# 	results.append((prompt, completion, y))
	# score = 0
	# results = list(set(results))
	# for _, completion, y in results:
	# 	if completion == y:
	# 		score += 1
	# log.info(f'Score: {score}/{len(results)} = {100.*score/len(results):.2f}%')
	return xs, ys

