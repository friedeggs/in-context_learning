import sys, os
from collections import defaultdict, OrderedDict
import itertools
import logging; log = logging.getLogger(__name__)
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import scipy
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
	evaluate_s,
)
from .content_lib import (
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
	comma_separate,
	io_format,
	space_separate,
	str_separate, 
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
from .sequence_manipulation import (
	get_value_dict_chars,
	run_task,
)

# def reverse_natural_content(argv, n_train=80, n_test=5, n=5):
# 	content_dataset = NondeterministicDataset(func=lambda _: list(random_word_length_5()), offset=0)
# 	sample = FewShotDataset(content_dataset, n_train=n_train, n_test=n_test)
# 	formatted = FormattingDataset(sample, 
# 		# lambda _: space_separate(_), 
# 		# lambda _: space_separate(_[::-1]),
# 		lambda _: ', '.join(_), 
# 		lambda _: ', '.join(_[::-1]),
# 		map=True,
# 	)
# 	run_task(argv, formatted, n_train, n_test, f'_reverse_natural_content_n-{n}_n_train-{n_train}', sample=sample, value_dict_func=get_value_dict_chars)

def run_queries(argv):
	completion_kwargs = {
		'staged': True,
		# 'staged': True, 
		'temperature': 0, 
		'engine': 'davinci', 
		'max_tokens': 0, # 20,
		'stop': '\n',
		'logprobs': 100,
		'echo': True,
	}
	mock = 'submit' not in argv
	cache = get_cache('cache_gpt3_identity.jsonl')
	gpt = GPT3(cache, mock)

	n_chars = 5
	total = int(scipy.special.comb(8, n_chars) * math.factorial(n_chars))
	# token cost is 24 * (n_train + 1) - 1
	# 23, 47, 71, 95
	transform_func = str_separate(', ')
	fmt_func = lambda _: io_format(_, transform=transform_func, include_y=True)
	# fmt_func = lambda _: io_format(_, x_label='', y_label='', intra_separator='', transform=transform_func, include_y=True)
	# fmt_func_x = lambda _: io_format(_, x_label='', y_label='', intra_separator='', transform=transform_func, include_y=False)
	
	for n_train in [3]: # range(10,11): # 6):
		# for comb in itertools.combinations('abcd', 3):
		pbar = tqdm(total=total)
		cntr = 0
		score = 0
		for comb_idx, comb in enumerate(itertools.combinations('abcdefgh', n_chars)):
			for perm in itertools.permutations(comb):
				set_seed(cntr)
				content = np.tile(np.concatenate(
					[random_distinct_chars((1, 1, n_chars))
					 for _ in range(n_train)] + [np.array(perm).reshape(1, 1, n_chars)]), (1,2,1))
				completion = None
				if cntr % 100 == 0:
					prompt = fmt_func(content)
					# prompt_x = fmt_func_x(content)
					y = transform_func(perm)
					prompt_x = prompt[:-(len(y)+1)] # + '\n'
					response = gpt.complete(prompt, completion_kwargs)
					completion = get_completion_logprobs_s(response, completion_kwargs, prompt_x)
					correct = (completion == y) # evaluate_s(response, completion_kwargs, y, prompt_x)
				if cntr == 0:
					log.info(prompt)
					log.info(prompt_x.rstrip())
					log.info(completion)
					log.info(correct)
				cntr += 1
				# log.info((completion, n_train))
				# log.info(response['choices'][0]['text'])
				if completion is not None:
					correct = (completion == y)
					score += correct
					if not correct:
						log.info(f'Incorrect; {score}/{cntr} (n_train={n_train})')
				pbar.update(1)
				# break
			# 	if cntr >= 50:
			# 		break
			# if comb_idx >= 1:
			# 	break
		# break
		if cntr:
			log.info(f'Score: {score}/{cntr} = {100.*score/cntr:.2f} (n_train={n_train})')

	cost = gpt.calculate_cost()
	if cost:
		log.info('This request will cost %d tokens (including completion)' % cost)
		# k = 'y'
		k = None
		if k == 'y':
			log.warn('Submitting queries without confirmation!')
		staged = gpt.get_staged_queries()
		while k not in list('ynqc'):
			k = input(f"Submit {len(staged)} staged request(s) to the server? [y/n/q/c] ")
		if k not in list('yc'):
			return
		gpt.run_staged_queries(k)



