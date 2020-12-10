
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
	# setup_calendar_2x2_exception(sys.argv, n_train=70, n_test=5, exclude_train_from_test=False)
	# setup_calendar_2x2_exception_dummy(sys.argv, n_train=70, n_test=5, exclude_train_from_test=False)
	
	# reverse(sys.argv, n=5, n_train=80, n_test=500)
	# dates_unnatural_content(sys.argv, n_train=15, n_test=500)
	# dates_natural_format(sys.argv, n_train=15, n_test=500)
	dates(sys.argv, n_train=15, n_test=500)

	# reverse(sys.argv, n=5, n_train=50, n_test=500)
	reverse_natural_content(sys.argv, n=5, n_train=80, n_test=10)
	reverse_to_natural_content(sys.argv, n=5, n_train=80, n_test=10)
	dates_unnatural_content(sys.argv, n_train=10, n_test=500)
	dates_natural_format(sys.argv, n_train=10, n_test=500)
	dates(sys.argv, n_train=10, n_test=500)
	addition_3_digit(sys.argv, n_train=100, n_test=500)
	reverse_to_natural_content(sys.argv, n=5, n_train=80, n_test=100)

