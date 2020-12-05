import numpy as np


# first_names = util.load_file('data/common_first_names.txt')
# last_names = util.load_file('data/common_last_names.txt')

def random_distinct_alpha_chars(n):
	return np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n, replace=False)

def random_permutation(n):
	return np.random.permutation(n)