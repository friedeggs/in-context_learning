import sys, os 
import exrex
import hashlib
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

def hash_str(s):
	return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8

CSV_PATH = 'results_urls.csv'
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

with open('data/urls2.txt') as f:
	real_urls = f.readlines()
	real_urls = list(map(lambda l: l.strip().split('\t'), real_urls))
# print(real_urls[:5])
words = [el for url in real_urls for el in url[1].split(' ')] # TODO or use common words from some vocab
words = list(filter(lambda x: x.isalpha(), words)) 
# print(words[:50])
set_seed()
strs = list(filter(lambda x: 10 < len(x), [exrex.getone('.*')[:30] for _ in range(1000)]))

# 4 x 2 x 2 x 5 x 4

# Data: 
# URL, TEXT
sem_data = [lambda: real_urls[np.random.choice(len(real_urls))]]
syn_data = [
	lambda: [
		np.random.choice(strs),
		' '.join([np.random.choice(words) for _ in range(random.randint(2, 15))])
	],
	# lambda: [
	# 	np.random.choice(strs),
	# 	exrex.getone('[A-Za-z]{%d}' % random.randint(2, 50))
	# ],
	lambda: [
		np.random.choice(strs),
		np.random.choice(strs),
	],
]

sem_fmt = [
	# ('URL TEXT',
	# 	Formatter('JOIN(" ")')),
	('<a href="URL">TEXT</a>',
		Formatter('SPLIT(1, MAP(PAREN("<a href=\\"", "\\">")), MAP(PAREN("", "</a>")), JOIN(""))')),
	# (r'\\href{URL}{TEXT}',
	# 	Formatter('SPLIT(1, MAP(PAREN("\\href{", "}")), MAP(PAREN("{", "}")), JOIN(""))')),
	# (r'\\url{URL}',
	# 	Formatter(r'SPLIT(1, MAP(PAREN("\\url{", "}")), NULL, JOIN(" "))')),
	('[TEXT](URL)',
		Formatter('[REVERSE, SPLIT(1, MAP(PAREN("[", "]")), MAP(PAREN("(", ")")), JOIN(""))]')),
]

syn_fmt = [
	('<a href="TEXT">URL</a>',
		Formatter('[REVERSE, SPLIT(1, MAP(PAREN("<a href=\\"", "\\">")), MAP(PAREN("", "</a>")), JOIN(""))]')),
	# ('(TEXT)[URL]',
	# 	Formatter('[REVERSE, SPLIT(1, MAP(PAREN("[", "]")), MAP(PAREN("(", ")")), JOIN(""))]')),
	# ('(URL)[TEXT]',
	# 	Formatter('SPLIT(1, MAP(PAREN("[", "]")), MAP(PAREN("(", ")")), JOIN(""))')),
	('[URL](TEXT)',
		Formatter('SPLIT(1, MAP(PAREN("[", "]")), MAP(PAREN("(", ")")), JOIN(""))')),
]

"""
Experiments:
Tasks:
- Reverse
	- to the uncommon format
- Reformat
	- when the input is unnatural  
"""

# 3 x 3 x 4 x 3 x 3

def evaluate(gpt3, engine, n_train=15, n_test=1):
	global score, close, total, pending
	prefix = 'Process the input:'
	data_funcs = [(x, True) for x in sem_data] + [(x, False) for x in syn_data]
	fmt_funcs = [(x, True) for x in sem_fmt] + [(x, False) for x in syn_fmt]
	score = 0
	close = 0
	total = 0
	pending = 0
	rows = []

	for di, (d1, td1) in enumerate(data_funcs):
		for dj, (d2, td2) in enumerate(data_funcs):
			# if not (~td1 and td2):
			# 	continue
			# if di != dj:
			# 	continue

			set_seed(10_000 * di + dj)
			train_data = [d1() for i in range(n_train)]
			test_data = [d2() for i in range(n_test)]
			set_seed(1_000_000 + 10000 * di + dj)
			# extra_train_data = [d1()]
			# extra_test_data = [d2() for i in range(2)]

			for fi, ((fname1, f1), tf1) in enumerate(fmt_funcs):
				for fj, ((fname2, f2), tf2) in enumerate(fmt_funcs):
					if f1 == f2:
						continue
					# if fi != 0:
					# 	continue
					if not (tf1 & ~tf2):
						continue
					train_examples = [(f1.format(d), f2.format(d)) for d in train_data]
					test_examples = [(f1.format(d), f2.format(d)) for d in test_data]

					def _evaluate_helper(_train_examples, _x, _y, prefix, engine):
						global score, close, total, rows, pending
						response, rel, kwargs = gpt3.few_shot(_train_examples, x=_x, y=_y, temperature=0, prefix=prefix, engine=engine, max_tokens=100, staged=True, return_kwargs=True)
						rel = escape_ansi(rel)
						try:
							pred = response['choices'][0]['text'].lstrip().rstrip()
						except Exception:
							pred = None	
						if rel == 'EQUALS':
							score += 1	
						elif rel == 'CLOSE':
							close += 1
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

					# print()

	print(colored('Engine: %s' % engine, 'magenta'))
	print(colored('Score: %d/%d (%d close); %d pending' % (score, total, close, pending), 'magenta'))
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


"""
from synthetic_syntax import load_df
df = load_df()
df[(df.rel == 'EQUALS') & (df.tf1 | df.tf2)]['engine'].value_counts()

df[(df.rel == 'EQUALS') & ~df.tf1 & df.tf2]['engine'].value_counts() / df[~df.tf1 & df.tf2]['engine'].value_counts()
df[(df.rel == 'EQUALS') & ~df.tf1 & df.tf2]['engine'].value_counts() / df[~df.tf1 & df.tf2]['engine'].value_counts()

df[(df.tf1 | df.tf2)]['engine'].value_counts()
df.engine.value_counts()
df[(df.engine == 'ada')]
df[(df.rel == 'EQUALS')]['engine'].value_counts()
df[(df.rel != 'NOT EQUALS')]['engine'].value_counts()

condition = (~df.tf1 & df.tf2)
condition = ((df.di == 0) & (df.dj == 0) & df.tf1 & df.tf2)
condition = ((df.di == 1) & (df.dj == 1) & df.tf1 & df.tf2)


df[(df.engine == 'davinci') & (df.num_examples == 5) & (df.fi != 0) & (df.fj != 0)]


condition = (df.tf1 & df.tf2)
	(ada        0.555556
	babbage    0.555556
	curie      0.722222
	davinci    0.888889
	Name: engine, dtype: float64, ada        18
	babbage    18
	curie      18
	davinci    18
condition = (~df.tf1 & df.tf2)
	(ada        0.527778
	babbage    0.611111
	curie      0.722222
	davinci    0.916667
	Name: engine, dtype: float64, ada        36
	curie      36
	davinci    36
	babbage    36
condition = (df.tf1 & ~df.tf2)
	(ada        0.305556
	babbage    0.555556
	curie      0.611111
	davinci    0.666667
	Name: engine, dtype: float64, ada        36
	curie      36
	davinci    36
	babbage    36
condition = (~df.tf1 & ~df.tf2)
	(ada        0.333333
	babbage    0.555556
	curie      0.500000
	davinci    0.777778
	Name: engine, dtype: float64, ada        18
	babbage    18
	curie      18
	davinci    18


condition = (df.td1 & df.td2 & df.tf1 & ~df.tf2)
condition = (df.td1 & df.td2)
	(ada        0.666667
	babbage    0.916667
	curie      0.750000
	davinci    0.833333
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12
condition = (~df.td1 & df.td2)
	(ada        0.333333
	babbage    0.583333
	curie      0.583333
	davinci    0.708333
	Name: engine, dtype: float64, ada        24
	curie      24
	davinci    24
	babbage    24
condition = (df.td1 & ~df.td2)
	(ada             NaN
	babbage    0.250000
	curie      0.375000
	davinci    0.541667
	Name: engine, dtype: float64, ada        24
	curie      24
	davinci    24
	babbage    24
condition = (~df.td1 & ~df.td2) 
	(ada        0.625000
	babbage    0.645833
	curie      0.791667
	davinci    0.979167
	Name: engine, dtype: float64, ada        48
	curie      48
	davinci    48
	babbage    48

condition = ((df.di == 1) & df.td2)
	(ada        0.416667
	babbage    0.666667
	curie      0.583333
	davinci    0.750000
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12
condition = (df.td1 & (df.dj == 1)) # spread of ability 
	(ada             NaN
	babbage    0.166667
	curie      0.500000
	davinci    0.583333
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12
condition = ((df.di == 1) & (df.dj == 1)) # ada weirdly good at this 
	(ada        1.000000
	babbage    0.083333
	curie      0.666667
	davinci    1.000000
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12

babbage sometimes better than curie 
condition = ((df.di == 2) & (df.dj == 1)) not enough data? babbage does better 
	(ada        0.333333
	babbage    0.750000
	curie      0.666667
	davinci    0.916667
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12
condition = ((df.di == 1) & (df.dj == 2)) # all do quite well 
	(ada        0.750000
	babbage    0.916667
	curie      1.000000
	davinci    1.000000
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12
condition = ((df.di == 1) & (df.dj == 2) & df.tf1 & df.tf2) # all do perfect (2 data points)

condition = ((df.di == 2) & df.td2)
	(ada        0.250000
	babbage    0.500000
	curie      0.583333
	davinci    0.666667
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12
condition = (df.td1 & (df.dj == 2))
	(ada       NaN
	babbage    0.333333
	curie      0.250000
	davinci    0.500000
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12
condition = ((df.di == 2) & (df.dj == 2))
	(ada        0.416667
	babbage    0.833333
	curie      0.833333
	davinci    1.000000
	Name: engine, dtype: float64, ada        12
	babbage    12
	curie      12
	davinci    12
condition = (df.di == df.dj)
	(ada        0.694444
	babbage    0.611111
	curie      0.750000
	davinci    0.944444
	Name: engine, dtype: float64, ada        36
	curie      36
	davinci    36
	babbage    36
df[(df.rel == 'EQUALS') & condition]['engine'].value_counts() / df[condition]['engine'].value_counts(), df[condition]['engine'].value_counts()
df[(df.rel != 'NOT EQUALS') & condition]['engine'].value_counts() / df[condition]['engine'].value_counts(), df[condition]['engine'].value_counts()

df[condition & (df.engine == 'ada')][['x', 'pred', 'y', 'rel']]

df[(df.rel == 'NOT EQUALS') & condition & (df.engine == 'davinci')][['pred', 'y', 'tf1', 'tf2']]
df[(df.rel == 'NOT EQUALS') & condition].groupby([['tf1', 'tf2']]).size()

df[(df.rel == 'NOT EQUALS')].groupby(['di', 'dj', 'tf1', 'tf2']).size()
df[(df.rel == 'NOT EQUALS')].groupby(['di', 'tf1', 'tf2']).size()

df[condition & (df.rel == 'NOT EQUALS') & (df.engine == 'ada')].groupby(['di', 'tf1', 'tf2']).size()


df[(df.rel == 'NOT EQUALS')]

"""

"""
condition = (df.engine == 'ada') & (df.num_examples == 2) & (df.di == df.dj)
condition = (df.engine == 'davinci') & (df.num_examples == 5)
df[condition & (df.rel != 'NOT EQUALS')].groupby(['di', 'tf1', 'tf2']).size(), df[condition].groupby(['di', 'tf1', 'tf2']).size()
df[condition & (df.rel != 'NOT EQUALS')].groupby(['di', 'fi', 'fj']).size()

df[~(condition & df.fi == 0)]
"""
def print_stats(df):
	pass

def load_df(csv_path=CSV_PATH):
	df = pd.read_csv(csv_path)
	df = df[keys_to_keep]
	df = df.dropna(subset=['pred'])
	return df

def main(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	gpt3.clear_staged_queries()
	# for engine in ['ada', 'babbage', 'curie']:
	for engine in ['davinci']:
	# for engine in ['ada', 'babbage', 'curie', 'davinci']:
		evaluate(gpt3, engine=engine)
	gpt3.run_staged_queries()

if __name__ == '__main__':
	main(sys.argv)

# Engine: ada, n = 2
# Score: 20/108 (26 close); 0 pending
# Engine: ada, n = 5
# Score: 38/108 (21 close); 0 pending
# Engine: ada, n = 10

# Engine: babbage, n = 2
# Score: 38/108 (24 close); 0 pending
# Engine: curie, n = 2
# Score: 38/108 (32 close); 0 pending

## tf1 & ~tf2
# Engine: ada, n = 2
# Score: 5/36 (6 close); 0 pending
# Engine: ada, n = 5
# Score: 12/36 (6 close); 0 pending
# Engine: ada, n = 10
# Score: 12/36 (6 close); 0 pending
# Engine: ada, n = 15
# Score: 13/36 (8 close); 0 pending


# Engine: babbage, n = 2
# Score: 12/36 (8 close); 0 pending
# Engine: babbage, n = 5
# Score: 16/36 (4 close); 0 pending
# Engine: babbage, n = 10
# 
# Engine: babbage, n = 15
# Score: 22/36 (3 close); 0 pending

# Engine: curie, n = 2
# Score: 12/36 (10 close); 0 pending
# Engine: curie, n = 5
# Score: 18/36 (7 close); 0 pending
# Engine: curie, n = 10
# Score: 25/36 (3 close); 0 pending
# Engine: curie, n = 15
# Score: 23/36 (7 close); 0 pending
# Engine: curie, n = 18
# Score: 26/36 (1 close); 0 pending

# Engine: davinci, n = 2
# Score: 14/36 (10 close); 0 pending
# Engine: davinci, n = 10
# Score: 24/36 (5 close); 0 pending
# Engine: davinci, n = 15
# Score: 25/36 (2 close); 0 pending

## tf1 & ~tf2
# n = 2: 5+6/36
# n = 5: 12+6/36
# n = 10: 12+6/36
# n = 15: 13+8/36


# n = 2: 12+8/36
# n = 5: 16+4/36
# n = 10: 
# n = 15: 22+3/36

# n = 2: 12+10/36
# n = 5: 18+7/36
# n = 10: 25+3/36
# n = 15: 23+7/36
# n = 18: 26+1/36

# Engine: davinci
# n = 2:	14+10/36
# n = 10:	24+5/36
# n = 15:	25+2/36



