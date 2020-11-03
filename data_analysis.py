import sys, os
from collections import OrderedDict
from enum import Enum, IntEnum
import exrex 
import json
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pdb
import random
import re
from typing import Any, Callable, Dict, List, Tuple, Optional
from termcolor import colored
import traceback

from process import read_cache

CSV_PATH = '/sailhome/rongf/results.csv'

def cache_to_csv(cache):
	column_names = [
		# API 
		'engine',
		'prompt',
		'max_tokens',
		'temperature',
		'top_p',
		'n',
		'stream',
		'logprobs',
		'stop',
		# ---
		'response', # Output string 
	]
	rows_dict = {k: [] for k in column_names}
	for k,v in cache.items():
		if k == '__filename__':
			continue 
		for kk in column_names:
			rows_dict[kk].append(k.get(kk, None))
		rows_dict['response'][-1] = v

	df = pd.DataFrame(rows_dict, columns=column_names)
	df.to_csv(CSV_PATH)

def plot(xs, ys, fname):
	sz = 10
	fig = plt.figure(figsize=(sz,sz))
	# xs, ys 
	plt.axis('equal')
	plt.axis('off')
	plt.scatter(xs, ys) # c='b') # s=5)
	plt.savefig(fname)

def plot_TODO():
	df = pd.read_csv(CSV_PATH)
	xs = None
	# model.num_parameters()
	ys = None 

def print_num_parameters():
	from transformers import AutoModelForCausalLM, AutoTokenizer
	num_parameters = {}
	for model_name in [
		'gpt2-xl',
		'gpt2-large', 
		'gpt2-medium', 
		'gpt2', 
	]:
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
		num_parameters[model_name] = model.num_parameters()
	print(num_parameters)

if __name__ == '__main__':
	# cache_fname = '/u/pliang/results/cache.jsonl'
	# cache = read_cache('cache.jsonl')
	# cache_to_csv(cache)
	print_num_parameters()

