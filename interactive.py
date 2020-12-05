import sys, os 
import exrex
# import matplotlib
# matplotlib.use('tkAgg')
from Levenshtein import jaro_winkler as levenshtein
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

keys = [
	'max_tokens',
	'temperature', 
	'top_p',
	'stop',
	'n',
	'stream',
	'logprobs',
	'echo',
	'presence_penalty',
	'frequency_penalty',
	'best_of',
	'engine',
]

def match_key(_key):
	return sorted(keys, key=lambda x: -levenshtein(x, _key))[0]

def run(gpt3):
	kwargs = {
		'engine': 'davinci',
		'max_tokens': 10,
		'stop': ["\n", "\r", "\n\n"],
		'logprobs': 100,
		'temperature': 0.,
	}
	while True:
		k = None
		while k not in list('qrst'):
			k = input('Command ([q]uit/[r]un/update [s]ettings/enter [t]ext): ')
		if k == 's':
			try:
				kv = input('Settings to update: ').split(' ')
				_key = kv[0]
				value = ' '.join(kv[1:])
			except Exception as e:
				print('ERROR: %s' % e)
				continue
			if _key == 'del':
				value = match_key(value)
				if value not in kwargs:
					print(f'{value} not set')
				else:
					del kwargs[value]
					print(f'Unset {value}')
			else:
				# vals = [levenshtein(x, _key) for x in keys]
				# print(list(sorted(zip(keys, vals), key=lambda x: -x[1])))
				key = match_key(_key)
				if key == 'stop':
					try:
						value = eval(value)
					except Exception as e:
						print('ERROR: %s' % e)
						continue
				try:
					value = float(value)
					if (value).is_integer():
						value = int(value)
				except (TypeError, ValueError):
					pass 
				try:
					kwargs[key] = value
					print(f'Set {key} to {value}')
				except Exception as e:
					print('ERROR: %s' % e)
			print('== Settings ==')
			for key, value in kwargs.items():
				print(f'{key}: {value}')
		elif k == 't':
			# s = input('Enter text:\n')
			print('Enter text:')
			s = sys.stdin.read()
			# s = """“Thank you for doing business at our house, and I hope to see you again!” I left not a little unnerved, and still woozy from the **leeches**. “Come on,” said Nepthys, “you could use a drink.”\nQ: What are appropriate substitutes for **leeches** in the above text?\nA: bloodsucker, parasite, bloodletting, bleeding, worm, blood sucker, blood let, insect\n\nElectronic theft by foreign and industrial spies and disgruntled employees is costing U.S. companies billions and eroding their international competitive advantage. That was the message delivered by government and private security experts at an all-day conference on corporate **electronic** espionage. "Hostile and even friendly nations routinely steal information from U.S. companies and share it with their own companies," said Noel D. Matchett, a former staffer at the federal National Security Agency and now president of Information Security Inc., Silver Spring, Md.\nQ: What are appropriate substitutes for **electronic** in the above text?\nA:"""
			gpt3.complete(
				prompt=s,
				**kwargs
			)
		elif k == 'r':
			gpt3.run_staged_queries()
		elif k == 'q':
			break

def main(argv):
	GPT = GPT3 if 'submit' in argv else MockGPT3
	print('Using ' + GPT.__name__)

	cache_fname = f'cache_chris_{GPT.__name__}.jsonl'
	cache = read_cache(cache_fname)
	gpt3 = GPT(cache)
	run(gpt3)

if __name__ == '__main__':
	main(sys.argv)