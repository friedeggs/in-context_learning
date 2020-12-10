import sys, os 
import copy
from collections import OrderedDict, namedtuple
from itertools import product as cartesian_product
# import exrex
import json
import numpy as np
import pandas as pd
import random
import re
from termcolor import colored
import traceback

try:
	import exrex
except ImportError:
	class Dummy:
		def __init__(self):
			pass
	exrex = Dummy()
	exrex.getone = lambda x: 'XXX'

from data.model_info import model_info
from error_analysis import (
	analyze_errors,
	get_value_dict,
	match_templates,
	filter_templates,
	print_templates,
)
# from formatting import (
# 	run_spelling_and_interleave_reverse_evaluate,
# )
from process import (
	GPT3, MockGPT3,
	read_cache, 
)
from process_transformers import Runner
from sample import (
	create_date_schema,
	create_name_schema,
	create_url_schema,
	run_schema_task,
	run_date_investigation,
	run_perplexity_investigation,
	run_spelling_and_interleave_reverse_evaluate,
	run_perplexity_investigation_sampled_train,
	load_df,
	save_df,
	print_logprobs,
)
import util
from util import set_seed

# change options to max_tokens = max length of expected output, unset stop

def main(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_{GPT.__name__}_2.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	gpt3.clear_staged_queries()

	date_schema = create_date_schema()
	name_schema = create_name_schema()
	url_schema = create_url_schema()

	# # engines = ['ada']
	# engines = ['davinci']
	# # engines = ['ada', 'babbage', 'curie', 'davinci']
	# for engine in engines:
	# 	print('Processing model %s' % engine)
	# 	# run_perplexity_investigation(gpt3, engine, date_schema, max_tokens=20)
	# 	# run_spelling_and_interleave_reverse_evaluate(gpt3, engine)
	# 	# run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=5, n_test=1000, max_tokens=20)
	# 	# run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=10, n_test=100, max_tokens=20)
	# 	# run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=50, n_test=10, max_tokens=20)
	# 	# run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=10, n_test=1000, max_tokens=20)

	# 	run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=3, n_test=1000, max_tokens=20)
	# 	run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=5, n_test=1000, max_tokens=20)
	# 	run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=10, n_test=100, max_tokens=20)
	# 	# run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=15, n_test=1000, max_tokens=20)
	# 	# run_date_investigation(gpt3, engine, date_schema, max_tokens=20)
	# 	# run_schema_task(gpt3, engine, name_schema, max_tokens=20)
	# 	# run_schema_task(gpt3, engine, url_schema, max_tokens=150)
	# 	print()

	# engines = ['ada', 'babbage', 'curie', 'davinci']
	# engines = ['curie']
	# for engine in engines:
	# 	run_perplexity_investigation_sampled_train(gpt3, engine, date_schema, n_train=15, n_test=1000, max_tokens=20)
	# 	save_df()

	# run_perplexity_investigation_sampled_train(gpt3, 'ada', date_schema, n_train=100, n_test=100, max_tokens=20)
	# run_perplexity_investigation_sampled_train(gpt3, 'babbage', date_schema, n_train=15, n_test=1000, max_tokens=20)
	# run_perplexity_investigation_sampled_train(gpt3, 'curie', date_schema, n_train=5, n_test=1000, max_tokens=20)
	# run_perplexity_investigation_sampled_train(gpt3, 'curie', date_schema, n_train=10, n_test=1000, max_tokens=20)
	run_perplexity_investigation_sampled_train(gpt3, 'davinci', date_schema, n_train=10, n_test=1000, max_tokens=20)

	save_df()
	print('This request will cost %d tokens' % gpt3.calculate_cost())
	gpt3.run_staged_queries()
	save_df()
	# df = load_df()
	# analyze_errors(df[(df.engine == 'davinci') & (df.num_examples == 5) & (df.rel != 'EQUALS')])

	# print_logprobs(engines)

if __name__ == '__main__':
	main(sys.argv)
