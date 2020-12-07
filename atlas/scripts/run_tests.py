
import sys, os
sys.path.append('.')
import atlas.test.test_dataset as t
from atlas.dates import create_date_dataset, run
from atlas.sequence_manipulation import permutations, reverse, dates, random_char, random_num
from atlas.sequence_manipulation import *

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
	setup_calendar_2x2_exception(sys.argv, n_train=70, n_test=5, exclude_train_from_test=False)
	setup_calendar_2x2_exception_dummy(sys.argv, n_train=70, n_test=5, exclude_train_from_test=False)