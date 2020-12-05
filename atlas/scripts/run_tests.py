
import sys, os
sys.path.append('.')
import atlas.test.test_dataset as t
from atlas.dates import create_date_dataset, run

if __name__ == '__main__':
	# t.test_product_dataset()
	# t.test_fewshot_dataset()
	# date_dataset = create_date_dataset()
	# d = date_dataset
	# for idx, el in enumerate(d):
	# 	if idx >= 5:
	# 		break
	# 	print(idx, el)
	run()