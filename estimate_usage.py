
import sys, os
from collections import OrderedDict
# import exrex
import numpy as np
import pandas as pd
import random
import scipy
import scipy.special
from termcolor import colored

from text_histogram import histogram

from datetime import datetime

from process import (
	GPT3,
	MockGPT3,
	read_cache,
)

from util import escape_ansi, set_seed

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(s):
	return len(tokenizer.encode(s))

if __name__ == '__main__':
	# GPT = GPT3 if 'submit' in argv else MockGPT3
	# print('Using ' + GPT.__name__)
	cache_fname = f'cache_GPT3.jsonl'
	cache = read_cache(cache_fname)
	usage = OrderedDict()
	
	total = 0
	for k, v in cache.items():
		if k == '__filename__': continue
		# import pdb; pdb.set_trace()
		prompt = dict(k)['prompt']
		completion = ''.join([_['text'] for _ in v['choices']])
		s = prompt + completion
		n_tokens = count_tokens(s)
		date = datetime.fromtimestamp(v['created'])
		kk = (date.year, date.month, date.day)
		if kk not in usage:
			usage[kk] = 0
		usage[kk] += n_tokens
		# total += count_tokens(s)
		# if 'P-H-A-N-T-O-M' in s:
		# 	print(count_tokens(s))
	print(total)
	cumu = 0
	# last_date = 
	for k, v in usage.items():
		cumu += v
		print(k, v, cumu)

	# histogram(arr)
