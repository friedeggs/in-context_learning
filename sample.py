import sys, os 
from collections import OrderedDict, namedtuple
from itertools import product as cartesian_product
# import exrex
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
from process import (
	GPT3, MockGPT3,
	read_cache, 
)
from process_transformers import Runner
import util
from util import set_seed

Content = namedtuple('Content', 'function unnaturalness')
Form = namedtuple('Form', 'function unnaturalness')
FormPair = namedtuple('FormPair', 'src_form tgt_form')
Sample = namedtuple('Sample', 'content src_form tgt_form')

CSV_PATH = 'results_sample.csv'
rows = []
keys_for_comparison = [
	'engine',
	'temperature',
	'max_tokens',
	# 'staged',
	# 'prompt',
	'stop',
	'num_examples',
	# 'response',
	# 'rel',
	'x',
	'y',
]
keys_to_keep = [
	'engine',
	'temperature',
	'max_tokens',
	# 'staged',
	# 'prompt',
	'stop',
	'num_examples',
	# 'response',
	'rel',
	'x',
	'y',
	'pred',
	# 'balanced',
	# 'n',
	# 'score',
	'src_form',
	'tgt_form',
	'content',
	'src_form_un',
	'tgt_form_un',
	'content_un',
	'schema_type',
	'test_idx',
]

def create_date_schema():
	contents = {
		'year':
			OrderedDict({
				0: Content(lambda: np.random.randint(1970, 2020), 0),
				1: Content(lambda: np.random.randint(2030, 2040), 1),
				2: Content(lambda: np.random.randint(1, 10_000), 2),
			}),
		'month': 
			OrderedDict({
				0: Content(lambda: np.random.randint(1, 12+1), 0),
				1: Content(lambda: np.random.randint(40, 1000), 1),
			}),
		'day': 
			OrderedDict({
				0: Content(lambda: np.random.randint(1, 28+1), 0),
				1: Content(lambda: np.random.randint(40, 1000), 1),
			}),
	}

	date_forms = OrderedDict({
		0: Form(lambda _: f'{_.year}-{_.month:02d}-{_.day:02d}', 0),
		1: Form(lambda _: f'{_.month:02d}/{_.day:02d}/{_.year}', 0),
		2: Form(lambda _: f'{_.month} {_.day} {_.year}', 1),
		3: Form(lambda _: '{' + f'"month": {_.month}, "day": {_.day}, "year": {_.year}' + '}', 1),
		4: Form(lambda _: f'!{_.month}!{_.day}!{_.year}!', 2),
	})
	forms = {
		'src_form': date_forms,
		'tgt_form': date_forms,
	}

	DateSchema = namedtuple('DateSchema', 'year month day')
	DateSchema.contents = contents
	DateSchema.forms = forms
	return DateSchema

def create_name_schema():
	first_names = util.load_file('data/common_first_names.txt')
	last_names = util.load_file('data/common_last_names.txt')

	contents = {
		'firstName':
			OrderedDict({
				0: Content(lambda: np.random.choice(first_names), 0),
				1: Content(lambda: exrex.getone('[a-z]{%d}' % np.random.randint(5, 10)).capitalize(), 1),
			}),
		'lastName': 
			OrderedDict({
				0: Content(lambda: np.random.choice(last_names), 0),
				1: Content(lambda: exrex.getone('[a-z]{%d}' % np.random.randint(5, 10)).capitalize(), 1),
			}),
	}

	name_forms = OrderedDict({
		0: Form(lambda _: f'{_.firstName} {_.lastName}', 0),
		1: Form(lambda _: f'{_.lastName}, {_.firstName}', 0),
		2: Form(lambda _: '{' + f'"firstName": {_.firstName}, "lastName": {_.lastName}' + '}', 1),
		3: Form(lambda _: f'!{_.firstName}!{_.lastName}!', 1),
	})
	forms = {
		'src_form': name_forms,
		'tgt_form': name_forms,
	}

	NameSchema = namedtuple('NameSchema', 'firstName lastName')
	NameSchema.contents = contents
	NameSchema.forms = forms
	return NameSchema

def create_url_schema():
	with open('data/urls2.txt') as f:
		real_urls = f.readlines()
		real_urls = list(map(lambda l: l.strip().split('\t'), real_urls))
	# print(real_urls[:5])
	words = [el for url in real_urls for el in url[1].split(' ')] # TODO or use common words from some vocabulary
	words = list(filter(lambda x: x.isalpha(), words)) 
	# print(words[:50])
	# set_seed()
	# strs = list(filter(lambda x: 10 < len(x), [exrex.getone('.*')[:30] for _ in range(1000)]))

	contents = {
		'url':
			OrderedDict({
				0: Content(lambda: real_urls[np.random.choice(len(real_urls))][0], 0),
				1: Content(lambda: 'http://' + exrex.getone('[a-zA-Z0-9]{%d}' % np.random.randint(5, 10)) + '.com', 1),
			}),
		'text': 
			OrderedDict({
				0: Content(lambda: real_urls[np.random.choice(len(real_urls))][1], 0),
				1: Content(lambda: ' '.join([np.random.choice(words) for _ in range(random.randint(2, 15))]), 1),
			}),
	}

	url_forms = OrderedDict({
		0: Form(lambda _: f'<a href="{_.url}">{_.text}</a>', 0),
		1: Form(lambda _: f'[{_.text}]({_.url})', 0),
		2: Form(lambda _: '{' + f'"url": {_.url}, "text": {_.text}' + '}', 1),
		3: Form(lambda _: f'!{_.url}!{_.text}!', 1),
	})
	forms = {
		'src_form': url_forms,
		'tgt_form': url_forms,
	}

	UrlSchema = namedtuple('UrlSchema', 'url text')
	UrlSchema.contents = contents
	UrlSchema.forms = forms
	return UrlSchema

def sample(schema_type, src_form_idx: int, tgt_form_idx: int, content_idx, seed=None):
	s = schema_type
	ci = content_idx
	sfi = src_form_idx
	tfi = tgt_form_idx
	if seed is not None:
		set_seed(seed)
	content = s(**{f: s.contents[f][getattr(ci, f)].function() for f in s._fields})
	src_form = s.forms['src_form'][sfi].function(content)
	tgt_form = s.forms['tgt_form'][tfi].function(content)
	return Sample(content, src_form, tgt_form)

def _exactly_k_unnatural(s, idxs, attr, fields, k=1):
	cnt = 0
	for f, idx in zip(fields, idxs):
		if getattr(s, attr)[f][idx].unnaturalness > 0:
			cnt += 1
			if cnt > k:
				return False
	return cnt == k

def exactly_k_unnatural(schema_type, attr, k=1):
	s = schema_type
	if attr == 'contents':
		nt = schema_type
	elif attr == 'forms':
		nt = FormPair
	else:
		raise Exception(f'Unrecognized attribute {attr}')
	lists = [getattr(s, attr)[f].keys() for f in nt._fields]
	all_possibilities = cartesian_product(*lists)
	possibilities = list(filter(lambda _: _exactly_k_unnatural(s, _, attr, nt._fields, k), all_possibilities))
	if attr == 'contents':
		possibilities = [s(**{f: p for f, p in zip(nt._fields, poss)}) for poss in possibilities]
	elif attr == 'forms':
		possibilities = list(filter(lambda _: _[0] != _[1], possibilities))
		possibilities = [nt(**{f: p for f, p in zip(nt._fields, poss)}) for poss in possibilities]
	return possibilities

def print_possibilities(s, possibilities, attr):
	for p in possibilities:
		print({f: getattr(s, attr)[f][v].unnaturalness for f, v in p._asdict().items()})

def print_samples(samples):
	for _ in samples:
		print(f'{_.src_form} \t-> \t{_.tgt_form}')

def evaluate(gpt3, train_samples, test_samples, additional_kwargs={}, **kwargs):
	global rows
	train_examples = [(_.src_form, _.tgt_form) for _ in train_samples]
	test_examples = [(_.src_form, _.tgt_form) for _ in test_samples]
	for idx, (x, y) in enumerate(test_examples):
		response, rel, _kwargs = gpt3.few_shot(
			train_examples, 
			x=x, y=y, 
			**kwargs,
		)
		rel = util.escape_ansi(rel)
		try:
			pred = response['choices'][0]['text'].lstrip().rstrip()
		except Exception:
			try:
				pred = response[0]
			except Exception:
				pred = None
		row = {**_kwargs, **additional_kwargs}
		del row['prompt']
		row['num_examples'] = len(train_examples)
		row['x'] = x
		row['y'] = y
		row['pred'] = pred
		row['rel'] = rel
		row['test_idx'] = idx
		rows.append(row)

def run_schema_task(gpt3, engine, schema_type, **kwargs):
	default_kwargs = {
		'temperature': 0, 
		'prefix': None, 
		'engine': engine, 
		'max_tokens': 20, 
		'staged': True, 
		'return_kwargs': True,
		'stop': '\n',
		'verbose': False,
	}
	kwargs = {**default_kwargs, **kwargs}

	set_seed(0)
	print(sample(schema_type, 0, 1, schema_type(*([0] * len(schema_type._fields)))))

	n_train = 3
	n_test = 5

	poss_fc = list(cartesian_product(
		exactly_k_unnatural(schema_type, 'forms'), 
		exactly_k_unnatural(schema_type, 'contents', 0))) \
		+ list(cartesian_product(
		exactly_k_unnatural(schema_type, 'forms', 0), 
		exactly_k_unnatural(schema_type, 'contents'))) \
		+ list(cartesian_product(
		exactly_k_unnatural(schema_type, 'forms', 0), 
		exactly_k_unnatural(schema_type, 'contents', 0)))

	# df = load_df()
	# df = df[(df.rel != 'EQUALS') & (df.num_examples == 5)]
	# poss_f = [FormPair(x, y) for x, y in list(df[['src_form', 'tgt_form']].values)]
	# poss_c = list(map(eval, df['content'].values))
	# poss_fc = list(zip(poss_f, poss_c))
	print('Number of possible form and content combinations to process: %d' % len(poss_fc))

	s = schema_type
	samples = [[sample(schema_type, pf.src_form, pf.tgt_form, pc, seed) for seed in range(n_train + n_test)] \
		for pf, pc in poss_fc]
	for sm, (pf, pc) in zip(samples, poss_fc):
		# print_samples(sm)
		# print()
		content_un = False
		for field in s._fields:
			if s.contents[field][getattr(pc, field)].unnaturalness > 0:
				content_un = True
		additional_kwargs = {
			'src_form': pf.src_form,
			'tgt_form': pf.tgt_form,
			'content': pc,
			'src_form_un': s.forms['src_form'][pf.src_form].unnaturalness,
			'tgt_form_un': s.forms['tgt_form'][pf.tgt_form].unnaturalness,
			'content_un': content_un, # s.contents[pc].unnaturalness,
			'seed': 0,
			'schema_type': schema_type.__name__
		}
		evaluate(gpt3, sm[:n_train], sm[n_train:], additional_kwargs=additional_kwargs, **kwargs)

def save_df(rows):
	# print(rows)
	df = pd.DataFrame(rows) # , columns=column_names)
	# print(len(df))
	if os.path.isfile(CSV_PATH):
		df_prev = pd.read_csv(CSV_PATH)
		df = pd.concat([df, df_prev], sort=False)
		df['is_duplicate'] = df[keys_for_comparison].duplicated()
		df = df[~df['is_duplicate']]
		# print(df['rel'].value_counts())
	df = df[keys_to_keep]
	# print(df)
	# print(len(df))
	df.to_csv(CSV_PATH)

"""
from sample import load_df, print_stats
df = load_df(); len(df)
df[(df.rel != 'EQUALS') & (df.num_examples == 3)]
df[(df.src_form_un > 0) & !(df.tgt_form_un > 0)]
"""
def load_df(csv_path=CSV_PATH):
	df = pd.read_csv(csv_path)
	df = df[keys_to_keep]
	df = df.dropna(subset=['pred', 'schema_type', 'content_un'])
	return df

"""
from sample import load_df, print_stats
df = load_df(); len(df)
condition = (df.num_examples == 3) & ~((df.src_form_un == 0) & (df.tgt_form_un == 0)) & (df.content_un == 0)
condition = (df.num_examples == 3) & ((df.src_form_un == 0) & (df.tgt_form_un == 0)) & ~(df.content_un == 0)

for schema_type in ['DateSchema', 'NameSchema', 'UrlSchema']:
	condition = (df.num_examples == 3) & ~((df.src_form_un == 0) & (df.tgt_form_un == 0)) & (df.content_un == 0) & (df.schema_type == schema_type)
	print(print_stats(df, condition))
"""
def print_stats(df, condition):
	"""
	condition = (df.num_examples == 3) & ~((df.src_form_un == 0) & (df.tgt_form_un == 0))
	condition = (~(df.src_form_un == 0) & (df.tgt_form_un == 0)) | ((df.src_form_un == 0) & ~(df.tgt_form_un == 0))
	"""
	# model_info = util.load_file('model_info.json')
	scores = df[(df.rel == 'EQUALS') & condition]['engine'].value_counts() / df[condition]['engine'].value_counts()
	# , df[condition]['engine'].value_counts()
	data = []
	for model_name in [
			'gpt2', 
			'gpt2-medium', 
			'gpt2-large', 
			'gpt2-xl',
		] + [
			'ada', 
			'babbage', 
			'curie', 
			'davinci'
		]:
		num_params = model_info[model_name]['num_parameters']
		data += [(num_params, 100. * scores.get(model_name, 0.))]
	return str(data).replace('), (', ')(')[1:-1]
	# return data

def main(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	gpt3.clear_staged_queries()

	date_schema = create_date_schema()
	name_schema = create_name_schema()
	url_schema = create_url_schema()

	# # for engine in ['ada', 'babbage', 'curie']:
	# for engine in ['ada', 'davinci']:
	for engine in ['ada', 'babbage', 'curie', 'davinci']:
		print('Processing model %s' % engine)
		run_schema_task(gpt3, engine, date_schema, max_tokens=20)
		run_schema_task(gpt3, engine, name_schema, max_tokens=20)
		run_schema_task(gpt3, engine, url_schema, max_tokens=150)
		print()
	gpt3.run_staged_queries()
	save_df(rows)

	default_generation_kwargs = {
	    'do_sample': True, 
	    # 'max_length': 15, 
	    'top_k': 0, 
	    # 'top_p': 0.95, 
	    'temperature': 0.0001, 
	    # 'num_return_sequences': 5, 
	    'num_return_sequences': 1, 
	    'stop': '\n',
	}
	for model_name in [
		'gpt2-xl',
		'gpt2-large', 
		'gpt2-medium', 
		'gpt2', 
	]:
	# for model_name in model_names:
		# if 'gpt2' in model_name:
		#     continue
		print('Evaluating model %s' % model_name)
		set_seed()
		cache_fname = f'cache_{model_name}.jsonl'
		cache = read_cache(cache_fname)
		# settings = {'generation_kwargs': default_generation_kwargs}
		# cache['generation_kwargs'] = settings['generation_kwargs']
		# runner = Runner(model_name=model_name, settings=settings, cache=cache)
		# # test_copycat_remove(runner)
		# run_task_suite(runner, cache, cache_fname)
		# import pdb; pdb.set_trace()
		# write_cache(cache, cache_fname)
		# eval_copycat(model_name)
		# eval_arithmetic(model_name)
		gpt = Runner(model_name=model_name, settings={'generation_kwargs': default_generation_kwargs}, cache=cache)
		run_schema_task(gpt, model_name, date_schema, max_tokens=20, temperature=default_generation_kwargs['temperature'])
		run_schema_task(gpt, model_name, name_schema, max_tokens=20, temperature=default_generation_kwargs['temperature'])
		run_schema_task(gpt, model_name, url_schema, max_tokens=150, temperature=default_generation_kwargs['temperature'])
		# gpt.model.num_parameters()
	save_df(rows)

if __name__ == '__main__':
	main(sys.argv)



