import sys, os 
import exrex
# import matplotlib
# matplotlib.use('tkAgg')
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import re
from termcolor import colored
import traceback

from process import (
	GPT3, MockGPT3,
	read_cache, 
	set_seed
)

# from data_language import DataGenerator 
from syntax_language import Formatter 

def escape_ansi(line): # Source: https://stackoverflow.com/a/38662876
	if not line:
		return line
	ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
	return ansi_escape.sub('', line)

CSV_PATH = 'results_1.csv'
keys_for_comparison = [
	'engine',
	'temperature',
	'max_tokens',
	# 'staged',
	# 'prompt',
	'stop',
	# 'di',
	# 'dj',
	# 'td1',
	# 'td2',
	# 'fi',
	# 'fj',
	# 'tf1',
	# 'tf2',
	'num_examples',
	'num_transfer_examples',
	'test_name',
	# 'response',
	# 'rel',
	'x',
	'y',
]
keys_to_keep = [
	'engine',
	'temperature',
	'max_tokens',
	'staged',
	# 'prompt',
	'stop',
	'di',
	'dj',
	'td1',
	'td2',
	'fi',
	'fj',
	'tf1',
	'tf2',
	'num_examples',
	'num_transfer_examples',
	'test_name',
	# 'response',
	'rel',
	'x',
	'y',
	'pred',
]

# 4 x 2 x 2 x 5 x 4

# Data: 

# - normal phone numbers 
# - uncommon length numbers
# - capital letters
# - mixed capital, lowercase letters and numbers
### - arbitrary punctuation and alphanumeric characters
sem_data = [lambda: [exrex.getone('[0-9]{%d}' % x) for x in [3, 3, 4]]]
syn_data = [
	# lambda: [exrex.getone('[0-9]{%d}' % random.randint(2, 5)) for _ in range(random.randint(3, 5))],
	# lambda: [exrex.getone('[A-Z]{%d}' % random.randint(2, 5)) for _ in range(random.randint(3, 5))],
	lambda: [exrex.getone('[A-Za-z0-9]{%d}' % random.randint(2, 5)) for _ in range(random.randint(3, 5))],
]

sem_fmt = [Formatter('SPLIT(1, MAP(PAREN("(", ")")), LIST(JOIN("-")), JOIN(" "))')] # (415) 342-4000
syn_fmt = [
	# Formatter('JOIN("::")'),
	Formatter('SPLIT(1, MAP(ID), LIST([MAP(PAREN("(", ")")), JOIN("::")]), JOIN("::"))'),
	Formatter('JOIN(" #-# ")'),
	# Formatter('[MAP(PAREN("(", ")")), JOIN("::")]'),
]

# na_phone_numbers = ['[0-9]{3}', '[0-9]{3}', '[0-9]{4}']
# fake_data = ['[A-Z]{6}']
# fmt_1 = lambda x: '(%s) ' % x[0] + '-'.join(x[1:])
# fmt_2 = '%s %s %s'
# fmt_2b = '%s::(%s)::(%s)::(%s)'
# fmt_3 = 'Area code: %s, Central office prefix: %s, Line number: %s'
# fmt_4 = '[%s]-(%s)-<%s>'
# fmt_4b = '[%s]-(%s)-<%s>-\{%s\}'
# fmt_5 = '<[%s<->%s :~:%s (%s)'
# fmt_6 = '+1-(%s)-%s-%s' # todo check


# column_names = [
# 	# API 
# 	'engine',
# 	'prompt',
# 	'max_tokens',
# 	'temperature',
# 	'top_p',
# 	'n',
# 	'stream',
# 	'logprobs',
# 	'stop',
# 	# ---
# 	'response', # Output string 
# ]

# 4 models * 4 d * 2 f * 3 tests
def evaluate(gpt3, engine, n_train=2, n_test=3):
	global score, total, pending
	prefix = 'Process the input:'
	data_funcs = [(x, True) for x in sem_data] + [(x, False) for x in syn_data]
	fmt_funcs = [(x, True) for x in sem_fmt] + [(x, False) for x in syn_fmt]
	score = 0
	total = 0
	pending = 0
	rows = []

	for di, (d1, td1) in enumerate(data_funcs):
		for dj, (d2, td2) in enumerate(data_funcs):

			set_seed(10000 * di + dj)
			train_data = [d1() for i in range(n_train)]
			test_data = [d2() for i in range(n_test)]
			set_seed(1_000_000 + 10000 * di + dj)
			extra_train_data = [d1()]
			extra_test_data = [d2() for i in range(2)]

			for fi, (f1, tf1) in enumerate(fmt_funcs):
				for fj, (f2, tf2) in enumerate(fmt_funcs):
					if f1 == f2:
						continue
					if 'PAREN("(", ")"' not in f1.definition:
						continue
					train_examples = [(f1.format(d), f2.format(d)) for d in train_data]
					test_examples = [(f1.format(d), f2.format(d)) for d in test_data]
					_f1 = Formatter(f1.definition.replace('PAREN("(", ")"', 'PAREN("[", "]"'))
					_f2 = Formatter(f2.definition.replace('PAREN("(", ")"', 'PAREN("[", "]"'))
					extra_train_examples = [(_f1.format(d), _f2.format(d)) for d in extra_train_data]
					extra_test_examples = [(_f1.format(d), _f2.format(d)) for d in extra_test_data]
					# print(len(train_examples))
					# import pdb; pdb.set_trace()

					def _evaluate_helper(_train_examples, _x, _y, prefix, engine):
						global score, total, rows, pending
						response, rel, kwargs = gpt3.few_shot(_train_examples, x=_x, y=_y, temperature=0, prefix=prefix, engine=engine, max_tokens=50, staged=True, return_kwargs=True)
						rel = escape_ansi(rel)
						try:
							pred = response['choices'][0]['text'].lstrip().rstrip()
						except Exception:
							pred = None			
						if rel == 'EQUALS':
							score += 1
						if pred is not None:
							total += 1	
						else:
							pending += 1	

						row = kwargs
						del row['prompt']
						row['di'] = di
						row['dj'] = dj
						row['td1'] = td1
						row['td2'] = td2
						row['fi'] = fi
						row['fj'] = fj
						row['tf1'] = tf1
						row['tf2'] = tf2
						row['num_examples'] = len(_train_examples)
						row['num_transfer_examples'] = np.nan
						row['test_name'] = 'normal'
						# row['response'] = response
						row['x'] = x
						row['y'] = y
						row['pred'] = pred
						row['rel'] = rel
						return row

					for x, y in test_examples:
						row = _evaluate_helper(train_examples, x, y, prefix, engine)
						row['num_transfer_examples'] = np.nan
						row['test_name'] = 'normal'
						rows.append(row)

					# without example
					for x, y in extra_test_examples:
						row = _evaluate_helper(train_examples, x, y, prefix, engine)
						row['num_transfer_examples'] = 0
						row['test_name'] = 'transfer'
						rows.append(row)

					# with example
					for x, y in extra_test_examples:
						row = _evaluate_helper(train_examples + extra_train_examples, x, y, prefix, engine)
						row['num_transfer_examples'] = 1
						row['test_name'] = 'transfer'
						rows.append(row)

					print()

	print(colored('Engine: %s' % engine, 'magenta'))
	print(colored('Score: %d/%d; %d pending' % (score, total, pending), 'magenta'))
	print('')

	df = pd.DataFrame(rows) # , columns=column_names)
	if os.path.isfile(CSV_PATH):
		df_prev = pd.read_csv(CSV_PATH)
		df = pd.concat([df, df_prev], sort=False)
		df['is_duplicate'] = df[keys_for_comparison].duplicated()
		df = df[~df['is_duplicate']]
		# print(df['rel'].value_counts())
	df = df[keys_to_keep]
	df.to_csv(CSV_PATH)

# for model in model_names:
# 	pass 

"""
from synthetic_syntax import load_df
df = load_df()
df[(df.engine == 'davinci') & (df.rel != 'EQUALS')]
df[(df.engine == 'davinci') & (df.rel == 'EQUALS') & (df.test_name == 'transfer')]
df[(df.engine == 'davinci') & (df.rel != 'EQUALS')][['di','dj','fi','fj','td1','td2','tf1','tf2']]
df[(df.engine == 'davinci') & (df.rel == 'EQUALS') & (df.test_name == 'transfer')][['di','dj','fi','fj','td1','td2','tf1','tf2']]
df[(df.engine == 'davinci') & (df.test_name != 'transfer')]
df[(df.engine == 'davinci') & (df.test_name != 'transfer')]['rel'].value_counts()
df[(df.rel == 'EQUALS') & (df.num_transfer_examples == 1)]['engine'].value_counts()
df[(df.rel == 'EQUALS') & (df.test_name != 'transfer')]['engine'].value_counts()
df[(df.rel != 'EQUALS') & (df.test_name != 'transfer')]['engine'].value_counts()
df[(df.rel == 'EQUALS') & (df.test_name == 'transfer')]['engine'].value_counts()
df[(df.engine == 'davinci') & (df.rel != 'EQUALS') & (df.test_name == 'transfer')]
df[(df.engine == 'davinci') & (df.test_name == 'transfer')]
df[(df.engine == 'ada') & (df.test_name == 'transfer')]
df[(df.engine == 'ada') & (df.test_name == 'transfer')].sort_values(['rel'])[['x','y','pred','rel']]
df[(df.engine == 'davinci') & (df.test_name == 'transfer')].sort_values(['rel'])[['x','y','pred','rel']]
df[(df.engine == 'davinci') & (df.test_name == 'transfer')].sort_values(['rel'])[['x','y','pred','rel', 'num_transfer_examples']]
# df[(df.di == 0) & (df.dj == 1) & (df.rel == 'EQUALS') & (df.test_name != 'transfer')]['engine'].value_counts()
for engine in ['ada', 'babbage', 'curie', 'davinci']:
	print(df[(df.engine == engine) & (df.rel != 'EQUALS') & (df.test_name != 'transfer')].groupby(['di','dj']).apply(len))
df[(df.engine == 'davinci') & (df.rel != 'EQUALS') & (df.test_name != 'transfer')]
for engine in ['ada', 'babbage', 'curie', 'davinci']:
	print(df[(df.engine == engine) & (df.rel != 'EQUALS') & (df.test_name != 'transfer')].groupby(['fi','fj']).apply(len))
len(df[(df.engine == 'davinci') & (df.test_name == 'transfer')])

df[(df.rel == 'EQUALS') & (df.tf1 | df.tf2)]['engine'].value_counts()
df[(df.rel == 'EQUALS') & (~df.tf1 & ~df.tf2)]['engine'].value_counts()
df[(df.rel == 'EQUALS') & (df.fj == 2)]['engine'].value_counts()
df[(df.rel == 'EQUALS') & (df.engine == 'davinci')].groupby(['fi', 'fj']).size()
df[(df.rel == 'EQUALS') & (df.engine == 'curie')].groupby(['fi', 'fj']).size()
"""
def load_df():
	df = pd.read_csv(CSV_PATH)
	df = df[keys_to_keep]
	return df

def main(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	gpt3.clear_staged_queries()
	# for engine in ['curie']:
	for engine in ['ada', 'babbage', 'curie', 'davinci']:
		evaluate(gpt3, engine=engine)
	gpt3.run_staged_queries()

if __name__ == '__main__':
	main(sys.argv)





