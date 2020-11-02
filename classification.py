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

from synthetic_syntax import escape_ansi

def _run_task(gpt3, prefix, train_examples, test_examples):
	engine = 'curie'
	prefix = None
	score = 0
	total = 0
	pending = 0
	formatter = lambda x: f'{x[0]}: {x[1]}'
	for x, y in test_examples:
		response, rel, kwargs = gpt3.few_shot(
			train_examples, 
			x=x, y=y, 
			temperature=0, prefix=prefix, engine=engine, 
			max_tokens=5, staged=True, return_kwargs=True,
			formatter=formatter)
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
	print(colored('Engine: %s' % engine, 'magenta'))
	print(colored('Score: %d/%d; %d pending' % (score, total, pending), 'magenta'))
	print('')

def run_task_A(gpt3):
	prefix = """Classify the word or phrase using the labels "^*" for animals and "#@#" for plants & vegetables."""
	train_examples = [ # 5 of each
		('camel', '^*'),
		('goldfish', '^*'),
		('lettuce', '#@#'),
		('leopard', '^*'),
		('porcupine', '^*'),
		('beans', '#@#'),
		('celery', '#@#'),
		('horse', '^*'),
		('radish', '#@#'),
		('broccoli', '#@#'),
	]
	test_examples = [
		('llama', '^*'),
		('cat', '^*'),
		('elephant', '^*'),
		('monkey', '^*'),
		('panda', '^*'),
		('cucumber', '#@#'),
		('peas', '#@#'),
		('tomato', '#@#'),
		('spinach', '#@#'),
		('carrots', '#@#'),
	]
	_run_task(gpt3, prefix, train_examples, test_examples)

def run_task_B(gpt3):
	prefix = """Classify the word or phrase using the labels "^*" for animals, "#@#" for plants & vegetables, and "!!~" for cities."""
	train_examples = [ # 5 of each
		('camel', '^*'),
		('goldfish', '^*'),
		('Toronto', '!!~'),
		('lettuce', '#@#'),
		('Detroit', '!!~'),
		('Tokyo', '!!~'),
		('leopard', '^*'),
		('porcupine', '^*'),
		('beans', '#@#'),
		('celery', '#@#'),
		('Busan', '!!~'),
		('horse', '^*'),
		('radish', '#@#'),
		('broccoli', '#@#'),
		('Abu Dhabi', '!!~'),
	]
	test_examples = [
		('llama', '^*'),
		('cat', '^*'),
		('elephant', '^*'),
		('monkey', '^*'),
		('panda', '^*'),
		('cucumber', '#@#'),
		('peas', '#@#'),
		('tomato', '#@#'),
		('spinach', '#@#'),
		('carrots', '#@#'),
		('Vancouver', '!!~'),
		('Honolulu', '!!~'),
		('Miami', '!!~'),
		('Beijing', '!!~'),
		('San Marcos', '!!~'),
	]
	_run_task(gpt3, prefix, train_examples, test_examples)

def run_task_C(gpt3):
	prefix = """Classify the word or phrase using the labels "^*" for animals, "#@#" for plants & vegetables, and "!!~" for sports."""
	train_examples = [ # 5 of each
		('camel', '^*'),
		('goldfish', '^*'),
		('volleyball', '!!~'),
		('lettuce', '#@#'),
		('lacrosse', '!!~'),
		# ('luge', '!!~'),
		# ('leopard', '^*'),
		# ('porcupine', '^*'),
		('beans', '#@#'),
		# ('celery', '#@#'),
		# ('hockey', '!!~'),
		# ('horse', '^*'),
		# ('radish', '#@#'),
		# ('broccoli', '#@#'),
		# ('archery', '!!~'),
	]
	test_examples = [
		('llama', '^*'),
		('cat', '^*'),
		('elephant', '^*'),
		('monkey', '^*'),
		('panda', '^*'),
		('cucumber', '#@#'),
		('peas', '#@#'),
		('tomato', '#@#'),
		('spinach', '#@#'),
		('carrots', '#@#'),
		('rugby', '!!~'),
		('cycling', '!!~'),
		('baseball', '!!~'),
		('tennis', '!!~'),
		('judo', '!!~'),
	]
	_run_task(gpt3, prefix, train_examples, test_examples)

def run_task_D(gpt3):
	prefix = """Classify the word or phrase using the labels "animal" for animals, "plant/vegetable" for plants & vegetables, and "sport" for sports."""
	train_examples = [ # 5 of each
		('camel', 'animal'),
		('goldfish', 'animal'),
		('volleyball', 'sport'),
		('lettuce', 'plant/vegetable'),
		('lacrosse', 'sport'),
		('luge', 'sport'),
		('leopard', 'animal'),
		('porcupine', 'animal'),
		('beans', 'plant/vegetable'),
		('celery', 'plant/vegetable'),
		('hockey', 'sport'),
		('horse', 'animal'),
		('radish', 'plant/vegetable'),
		('broccoli', 'plant/vegetable'),
		('archery', 'sport'),
	]
	test_examples = [
		('llama', 'animal'),
		('cat', 'animal'),
		('elephant', 'animal'),
		('monkey', 'animal'),
		('panda', 'animal'),
		('cucumber', 'plant/vegetable'),
		('peas', 'plant/vegetable'),
		('tomato', 'plant/vegetable'),
		('spinach', 'plant/vegetable'),
		('carrots', 'plant/vegetable'),
		('rugby', 'sport'),
		('cycling', 'sport'),
		('baseball', 'sport'),
		('tennis', 'sport'),
		('judo', 'sport'),
	]
	_run_task(gpt3, prefix, train_examples, test_examples)

def main(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	cache_fname = f'cache_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	run_task_C(gpt3)
	gpt3.run_staged_queries()


if __name__ == '__main__':
	main(sys.argv)