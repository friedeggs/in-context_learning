from collections import defaultdict, OrderedDict
import numpy as np

from .util import (
	load_file
)

# first_names = util.load_file('data/common_first_names.txt')
# last_names = util.load_file('data/common_last_names.txt')

def random_distinct_alpha_chars(n):
	return np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n, replace=False)

def random_permutation(n):
	return np.random.permutation(n)

# words_1w100k = None
def load_words(minlen=4, maxlen=15):
	# global words_1w100k
	# if words_1w100k is None:
	lines = load_file('../data/count_1w100k.txt')
	words_1w100k = {line.split('\t')[0].lower(): int(line.split('\t')[1]) for line in lines}
	words_1w100k = {k: v for k, v in words_1w100k.items() if minlen < len(k) < maxlen}
	words_1w100k = OrderedDict(sorted(words_1w100k.items(), key=lambda x: -x[1]))
	words_1w100k = list(words_1w100k.keys())[:10_000]
	# words_1w100k = list(set(map(lambda line: line.split('\t')[0].lower(), lines)))
	# words_1w100k = list(filter(lambda w: 4 < len(w) < 15, words_1w100k))
	return words_1w100k
# words_1w100k = list(filter(lambda w: len(w) == 5, words_1w100k))

words_5 = None
def load_words_5():
	global words_5
	if words_5 is None:
		words_5 = load_words(4,6)
		words_5 = words_5[:2500]
	return words_5

def random_word_length_5():
	words = load_words_5()
	return np.random.choice(words)
