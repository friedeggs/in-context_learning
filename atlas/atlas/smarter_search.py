import sys, os
import abc
import copy
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
	random_distinct_alpha_chars,
	random_permutation,
	random_word_length_5,
)
from .form_lib import (
	comma_separate,
	space_separate,
	io_format, 
)
from .gpt import read_cache
from .gpt2 import GPT2
from .util import (
	make_immutable, 
	is_iterable, 
	MAX_INT, 
	set_seed,
	plot,
	moving_average,
)

def random_distinct_chars(n):
	return np.random.choice(list('abcdefgh'), n, replace=False)

def vanilla_metropolis_hastings(t: int, x, T: Callable, Teval: Optional[Callable], p: Callable):
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
	if np.log(random.random()) <= np.log(min(1, p_xprime / p_x)):
		x = xprime
		p_x = p_xprime
	return x, p_x

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
		x = copy.deepcopy(x)
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
		raise NotImplementedError

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

	cache = read_cache('cache_gpt2.jsonl')
	mock = 'submit' not in argv
	gpt2 = GPT2(cache, mock)
	# p_failure = lambda _: 1. - 1./gpt2.ppl(_, completion_kwargs)
	p_failure = lambda _: 1. - 1./gpt2.ppl(_.fmt() + '\n', completion_kwargs, prefix=prompt_fn(_.data))
	samples = []
	probs = []
	for _, t in zip(tqdm(range(T)), range(T)):
		sample = copy.sample()
		p_x = p_failure(sample)
		samples.append(sample)
		probs.append(p_x)
	vals = probs
	xs = np.arange(len(vals))
	ys = np.cumsum(vals)/np.arange(1,len(vals)+1)
	plot(np.arange(len(ys)),ys,f'../outputs/random_search_n_train-{copy.n_train}.png', scatter_kwargs={'c':'r'})
	log.info(ys[-1])
	return xs, ys

def run_smarter_search(argv, n_train, T=1500, engine='gpt2'):
	copy = Copy(5, n_train)
	xt = copy.sample()
	# log.info(x0.fmt())

	# q1 = pass

	# completion_kwargs = {
	# 	# 'staged': True,
	# 	'temperature': 0, 
	#	# 'engine': 'gpt2-medium', 
	# 	'engine': engine, 
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

	cache = read_cache(f'cache_{engine}.jsonl')
	mock = 'submit' not in argv
	gpt2 = GPT2(cache, mock)
	p_failure = lambda _: 1. - 1./gpt2.ppl(_.fmt() + '\n', completion_kwargs, prefix=prompt_fn(_.data))
	# log.info(prompt_fn(xt.data))
	samples = [xt]
	probs = [p_failure(xt)]
	for _, t in zip(tqdm(range(T)), range(T)):
	# for t in range(T):
		xt, p = vanilla_metropolis_hastings(
			t,
			xt, 
			copy.perturb,
			copy.perturb_eval,
			p_failure,
		)
		samples.append(xt)
		probs.append(p)
		# log.info(colored(xt.data[:,0,:], 'green'))
		# log.info(xt.fmt() + colored(gpt2.complete(xt.fmt(), completion_kwargs, False), 'magenta'))
	# samples = np.array(samples)
	# print(np.unique(samples, axis=0))
	samples = np.vstack(set(tuple(row.data[0,0,:].tolist()) for row in samples))
	# log.info(samples)
	log.info(len(samples))
	n_burn_in = 200 # 100 # 1000 # 100 # 0 # 1500
	n_skip = 5 # 10 # 5 # 1 # 10
	vals = probs[n_burn_in:][::n_skip]
	xs = np.arange(len(probs))
	xs = xs[n_burn_in:][::n_skip]
	ys = np.cumsum(vals)/np.arange(1,len(vals)+1)
	log.info(ys[-1])
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


