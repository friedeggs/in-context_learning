
import sys, os
sys.path.append('.')

import logging; log = logging.getLogger(__name__)
from termcolor import colored

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import numpy as np
# import pandas as pd

import atlas.test.test_dataset as t
from atlas.dates import create_date_dataset, run
from atlas.sequence_manipulation import permutations, reverse, dates, random_char, random_num
from atlas.sequence_manipulation import *
from atlas.smarter_search import run_smarter_search, run_random_search, plot_searches, run_chained_search, run_importance_sampling
from atlas.util import set_seed

if __name__ == '__main__':
	# t.test_product_dataset()
	# t.test_fewshot_dataset()
	# date_dataset = create_date_dataset()
	# d = date_dataset
	# for idx, el in enumerate(d):
	# 	if idx >= 5:
	# 		break
	# 	print(idx, el)
	# run()
	# permutations(sys.argv, n=5)
	# permutations(sys.argv, n=4)
	# permutations(sys.argv, n_train=150, n=3)
	# reverse(sys.argv, n=5)
	# # reverse(sys.argv, n=4)
	# reverse(sys.argv, n=3)
	# dates(sys.argv)
	# # random_char(sys.argv, n_train=500)
	# # random_num(sys.argv, n_train=500)
	# setup_calendar_2x2_exception(sys.argv, n_train=10)
	# setup_calendar_2x2_exception(sys.argv, n_train=60)
	# setup_calendar_2x2_exception(sys.argv, n_train=35)
	# setup_calendar_2x2_exception(sys.argv, n_train=70, n_test=5, exclude_train_from_test=False)
	# setup_calendar_2x2_exception_dummy(sys.argv, n_train=70, n_test=5, exclude_train_from_test=False)
	
	# reverse(sys.argv, n=5, n_train=80, n_test=500)
	# dates_unnatural_content(sys.argv, n_train=15, n_test=500)
	# dates_natural_format(sys.argv, n_train=15, n_test=500)
	# dates(sys.argv, n_train=15, n_test=500)

	# # reverse(sys.argv, n=5, n_train=50, n_test=500)
	# reverse_natural_content(sys.argv, n=5, n_train=80, n_test=10)
	# reverse_to_natural_content(sys.argv, n=5, n_train=80, n_test=10)
	# dates_unnatural_content(sys.argv, n_train=10, n_test=500)
	# dates_natural_format(sys.argv, n_train=10, n_test=500)
	# dates(sys.argv, n_train=10, n_test=500)
	# addition_3_digit(sys.argv, n_train=100, n_test=500)
	# reverse_to_natural_content(sys.argv, n=5, n_train=80, n_test=100)
	# xs = []
	# ys = []
	# for engine in ['gpt2', 'gpt2-medium', 'gpt2-large']:
	# 	set_seed()
	# 	x, y = run_smarter_search(sys.argv, n_train=2, T=100, engine=engine)
	# 	xs.append(x)
	# 	ys.append(y)
	# xs = []
	# ys = []
	# for n_train in range(2,6):
	# 	set_seed()
	# 	x, y = run_smarter_search(sys.argv, n_train=n_train, T=100, engine='gpt2-large')
	# 	xs.append(x)
	# 	ys.append(y)
	# sz = 15
	# import matplotlib
	# matplotlib.use('tkAgg')
	# from matplotlib import pyplot as plt
	# output_file = f'../outputs/compare_search_n_train.png'
	# _ = plt.figure(figsize=(sz, sz))
	# for x, y, c in zip(xs, ys, list('rgbkc')):
	# 	plt.scatter(x=x, y=y, c=c)
	# 	plt.axhline(y=y[-1], color=c, linestyle='-')
	# plt.savefig(output_file)
	# plot_searches(sys.argv, n_train=2, T=80000, engine='gpt2')

	# engine = 'gpt2-large'
	for n_train in range(2,11):
		for source_engine, target_engine in [('gpt2', 'gpt2-medium'), ('gpt2-medium', 'gpt2-large'), ('gpt2', 'gpt2-large')]:
			log.info(colored(f'===== n_train={n_train} source_engine={source_engine} target_engine={target_engine}', 'magenta'))

			set_seed()
			run_random_search(sys.argv, n_train=n_train, T=500, engine=target_engine)
			# set_seed()
			# run_smarter_search(sys.argv, n_train=2, T=500, engine=engine)
			set_seed()
			run_importance_sampling(sys.argv, n_train=n_train, T=500, source_engine=source_engine, target_engine=target_engine)
			print('')

	# set_seed()
	# run_random_search(sys.argv, n_train=5, T=200, engine='gpt2-large')
	# set_seed()
	# run_smarter_search(sys.argv, n_train=5, T=200, engine='gpt2-large')

	# n_train = 2
	# engine = 'gpt2-large'
	# xs = []
	# ys = []
	# lbls = []
	# set_seed()
	# x1, y1 = run_chained_search(sys.argv, n_train=n_train, T=2000)
	# lbl1 = f'n_train={n_train},engine={engine},chained'
	# set_seed()
	# x2, y2 = run_smarter_search(sys.argv, n_train=n_train, T=2000, engine=engine)
	# lbl2 = f'n_train={n_train},engine={engine},MH'
	# xs.extend([x1, x2])
	# ys.extend([y1, y2])
	# lbls.extend([lbl1, lbl2])
	# output_file = f'../outputs/compare_chained_search_n_train-{n_train}.png'
	# sz = 15
	# _, ax = plt.subplots(figsize=(sz, sz))
	# for x, y, c, lbl in zip(xs, ys, list('rbgkcm')*50, lbls):
	# 	ls = 'dotted' if 'chained' in lbl else 'dashed' if 'random' in lbl else 'solid'
	# 	result = ax.scatter(x=x, y=y, linestyle=ls, label=lbl)
	# 	c = result.get_facecolor()[0]
	# 	plt.axhline(y=y[-1], color=c, linestyle='-')
	# plt.ylim(0, .25)
	# ax.legend()
	# plt.savefig(output_file)
	# plt.clf()
	# plt.close('all')

	# plot_searches(sys.argv, n_train=1, T=3000, engine='gpt2')
	# plot_searches(sys.argv, n_train=1, T=1000, engine='gpt2-medium')
	# plot_searches(sys.argv, n_train=3, T=1000, engine='gpt2-medium')
	# plot_searches(sys.argv, n_train=3, T=1500, engine='gpt2-medium')
	# plot_searches(sys.argv, n_train=5, T=1000, engine='gpt2-medium')
	# set_seed()
	# run_smarter_search(sys.argv, n_train=1, T=1000, engine='gpt2')
	# set_seed()
	# run_random_search(sys.argv, n_train=1, T=2000)
	# for T in [2000, 10000]: # range(2000, 11000, 1000):
	# 	xs = []
	# 	ys = []
	# 	lbls = []
	# 	rows = []
	# 	for n_train in list(range(2,11)) + [41]:
	# 		for engine in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
	# 			log.info(colored(f'===== n_train={n_train} engine={engine}', 'magenta'))
	# 			set_seed()
	# 			x1, y1 = run_smarter_search(sys.argv, n_train=n_train, T=T, engine=engine)
	# 			lbl1 = f'n_train={n_train},engine={engine},MH'
	# 			set_seed()
	# 			x2, y2 = run_random_search(sys.argv, n_train=n_train, T=T, engine=engine)
	# 			lbl2 = f'n_train={n_train},engine={engine},random'
	# 			xs.extend([x1, x2])
	# 			ys.extend([y1, y2])
	# 			lbls.extend([lbl1, lbl2])

	# 			output_file = f'../outputs/compare_search_T-{T}.png'
	# 			sz = 15
	# 			_, ax = plt.subplots(figsize=(sz, sz))
	# 			for x, y, c, lbl in zip(xs, ys, list('rbgkcm')*50, lbls):
	# 				result = ax.scatter(x=x, y=y, c=c, label=lbl)
	# 				# c = result.get_facecolor()[0]
	# 				plt.axhline(y=y[-1], color=c, linestyle='-')
	# 			plt.ylim(0, .25)
	# 			ax.legend()
	# 			plt.savefig(output_file)
	# 			plt.clf()
	# 			plt.close('all')
	# 			rows.extend([
	# 				{'f': y1[-1],'n_train': n_train,'engine': engine,'mode': 'MH'},
	# 				{'f': y2[-1],'n_train': n_train,'engine': engine,'mode': 'random'},
	# 			])
	# 			df = pd.DataFrame(rows) # , columns=column_names)
	# 			df.to_csv('../outputs/compare_search.csv')
	# 			log.info(df)
	# 			log.info(f'Wrote {len(df)} rows to ../outputs/compare_search.csv')


