from collections import OrderedDict
import gzip
import inspect
import json
import logging; log = logging.getLogger(__name__)
import multiprocessing
import numpy as np
import os
import random
import re
import sys
from termcolor import colored
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple, Optional

MAX_INT = np.iinfo(np.int64).max

tokenizer = None
autotokenizer = None
vocab = None

def fix(n):
	"""For array indexing"""
	return None if n == 0 else n

def load_model():
	from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
	global tokenizer, autotokenizer, vocab
	model_id = 'gpt2'
	if tokenizer is None:
		tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
	if autotokenizer is None:
		autotokenizer = AutoTokenizer.from_pretrained(model_id)
	if vocab is None:
		vocab = tokenizer.get_vocab()

def set_seed(seed: int = 0):
	seed = seed % 2147483647 # seed must be <= 2**32-1 # largest prime under 2**31
	random.seed(seed)
	np.random.seed(seed)
	# tf.random.set_seed(seed)
	# torch.manual_seed(seed)

def write_to_file(filename, s):
	with open(filename, 'w') as f:
		f.write(s)

def load_file(filename):
	if filename.endswith('.txt'):
		with open(filename) as f:
			lines = list(map(str.rstrip, f.readlines()))
			return lines
	if filename.endswith('.json'):
		return json.loads(filename)
	if filename.endswith('.json.gz'):
		with gzip.open(filename, 'rt') as f:
			return json.load(f)
	raise Exception(f'File name {filename} ends with unrecognized extension.')

def save_file(filename, obj):
	if filename.endswith('.txt'):
		with open(filename, 'w') as f:
			for line in obj:
				f.write(line + '\n')
	elif filename.endswith('.json'):
		return json.dump(obj, filename)
	raise Exception(f'File name {filename} ends with unrecognized extension.')

def insert_random_spaces(s, p=[.8,.15,.05]):
	ss = s.split(' ')
	spaces = [' ' * np.random.choice(range(3), p=p) for i in range(len(ss)+1)]
	return ''.join([el for tup in zip(spaces, [''] + list(' ' * len(ss[:-1])) + [''], ss + ['']) for el in tup])

def escape_ansi(line): # Source: https://stackoverflow.com/a/38662876
	if not line:
		return line
	ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
	return ansi_escape.sub('', line)

def flatten(lst_of_lsts):
	return [el for lst in lst_of_lsts for el in lst]

def repeat(lst, n):
	"""
	>>> repeat(list('abc'), 2)
	['a', 'a', 'b', 'b', 'c', 'c']
	"""
	return [x for x in lst for _ in range(n)]

def permute(lst, order):
	return [lst[i] for i in order]

def is_iterable(maybe_lst):
	if isinstance(maybe_lst, str):
		return False
	try:
		_ = iter(maybe_lst)
	except TypeError:
		return False
	else:
		return True

def to_tuple(x):
	if is_iterable(x):
		return tuple(map(to_tuple, x))
	return x

def get_type(funcname, attrname):
	return inspect.signature(funcname).parameters[attrname].annotation

def make_immutable(obj):
	if isinstance(obj, dict):
		return tuple(sorted(({k: make_immutable(v) for k, v in obj.items()}).items()))
	elif isinstance(obj, (list, tuple)):
		return tuple([make_immutable(x) for x in obj])
	return obj

def logsumexp(lst):
	"""
	>>> np.exp(reduce_func([np.log(.5), np.log(.3)]))
	0.8
	"""
	return np.log(sum(map(np.exp, lst)))

def count_tokens(s):
	load_model()
	return len(tokenizer.encode(s))

def show_tokenization(s, delimiter='|'):
	load_model()
	if not isinstance(s, list):
		s = [s]
	texts = []
	for _s in s:
		token_ids = autotokenizer.encode(_s)
		tokens = autotokenizer.convert_ids_to_tokens(token_ids)
		if delimiter not in autotokenizer.byte_decoder:
			delimiter = chr(ord(' ') + 256)
		text = delimiter.join(tokens)
		text = bytearray([autotokenizer.byte_decoder[c] for c in text]).decode("utf-8", errors=autotokenizer.errors)
		texts.append(text)
	return texts

def get_tokenization(s):
	load_model()
	token_ids = autotokenizer.encode(s)
	tokens = autotokenizer.convert_ids_to_tokens(token_ids)
	tokens = [bytearray([autotokenizer.byte_decoder[c] for c in tok]).decode("utf-8", errors=autotokenizer.errors) for tok in tokens]
	return tokens

def run_parallel(func, dataset, N_PARALLEL=8):
	with multiprocessing.Pool(N_PARALLEL) as p:
		list(tqdm(p.imap(func, dataset), total=len(dataset)))

def plot_logprobs(logprobs):
	xs = range(len(logprobs))
	ys = logprobs


