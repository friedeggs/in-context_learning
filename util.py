from collections import OrderedDict
import json
import numpy as np
import os
import random
import re
import sys
from typing import Any, Callable, Dict, List, Tuple, Optional
from termcolor import colored
from tqdm import tqdm


def set_seed(seed: int = 0):
	random.seed(seed)
	np.random.seed(seed)
	# tf.random.set_seed(seed)
	# torch.manual_seed(seed)

def load_file(filename):
	if filename.endswith('.txt'):
		with open(filename) as f:
			lines = list(map(str.rstrip, f.readlines()))
			return lines
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