
import sys, os
sys.path.append('.')

import logging; log = logging.getLogger(__name__)
from termcolor import colored

import atlas.test.test_dataset as t
from atlas.dates import create_date_dataset, run
from atlas.sequence_manipulation import permutations, reverse, dates, random_char, random_num
from atlas.sequence_manipulation import *
from atlas.smarter_search import run_smarter_search, run_random_search, plot_searches
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
	# plot_searches(sys.argv, n_train=1, T=3000, engine='gpt2')
	# plot_searches(sys.argv, n_train=1, T=1000, engine='gpt2-medium')
	# plot_searches(sys.argv, n_train=3, T=1000, engine='gpt2-medium')
	# plot_searches(sys.argv, n_train=3, T=1500, engine='gpt2-medium')
	# plot_searches(sys.argv, n_train=5, T=1000, engine='gpt2-medium')
	# set_seed()
	# run_smarter_search(sys.argv, n_train=1, T=1000, engine='gpt2')
	# set_seed()
	# run_random_search(sys.argv, n_train=1, T=2000)
	for n_train in range(2,6):
		for engine in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
			log.info(colored(f'===== n_train={n_train} engine={engine}', 'magenta'))
			set_seed()
			x, y = run_smarter_search(sys.argv, n_train=n_train, T=2000, engine=engine)
			set_seed()
			x, y = run_random_search(sys.argv, n_train=n_train, T=2000, engine=engine)

