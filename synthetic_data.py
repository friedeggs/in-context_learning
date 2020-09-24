import numpy as np
import quantecon as qe

import os
import random
import sys
from typing import Any, Callable, Dict, List, Tuple, Optional

def set_seed(seed: int = 0):
	random.seed(seed)
	np.random.seed(seed)
	# tf.random.set_seed(seed)
	# torch.manual_seed(seed)

def sample_categorical_distribution(alphas):
	return np.random.dirichlet(alphas)

def sample_transition_matrix(N: int, alpha: float = .2):
	"""
		alpha: lower is peakier 
	"""
	if N == 1:
		return np.eye(1)
	while True:
		M = np.stack([sample_categorical_distribution(alpha * np.ones(N)) for i in range(N)])  # rows sum to 1
		for i in range(N):
			if np.argmax(M[i]) == i:
				idx = np.random.randint(1, N)
				temp = M[i,i]
				M[i,i] = M[i,(i+idx)%N]
				M[i,(i+idx)%N] = temp
				
		mc = qe.MarkovChain(M) 
		if mc.is_aperiodic and mc.is_irreducible:
			return M

def compute_start_idxs(mc):
	if 'start_idxs' in mc:
		return 
	if mc['n_levels'] == 1:
		mc['start_idxs'] = np.array([0, mc['n_symbols']])
	else:
		for mc1 in mc['mcs']:
			compute_start_idxs(mc1)
		mc['start_idxs'] = np.cumsum([0] + [mc1['start_idxs'][-1] for mc1 in mc['mcs']])

def sample_multilevel_markov_chain(levels: int = 2, lam: float = 1., alpha: float = .2):
	n_symbols = np.random.poisson(lam=lam) + 2  # 2 for EOS + non-EOS
	mc = {
		'n_levels': levels,
		'n_symbols': n_symbols,
		'probs': sample_transition_matrix(n_symbols, alpha),
		'mcs': [sample_multilevel_markov_chain(levels-1, lam) for i in range(n_symbols-1)] if levels > 1 else None,
	}
	compute_start_idxs(mc)
	return mc

def sample_from_multilevel_markov_chain(mc, prefix: str = '', MAX_LENGTH: int = 1024):
	n_symbols = mc['n_symbols']
	n_levels = mc['n_levels']
	sequence = []
	symbol = np.random.choice(range(1, n_symbols), p=mc['probs'][0][1:]/mc['probs'][0][1:].sum())  # take one step from STOP state 
	while symbol != 0 and len(sequence) < MAX_LENGTH:  # STOP state
		if n_levels == 1:
			sequence.append(f'{prefix}{symbol}')
		else:
			sequence.extend(sample_from_multilevel_markov_chain(mc['mcs'][symbol-1], prefix + f'{symbol}_'))
		symbol = np.random.choice(range(n_symbols), p=mc['probs'][symbol])
	return sequence

def sample_hmm(lam_hidden: float = 5., lam_visible: float = 2., alpha_hidden: float = .2, alpha_visible: float = .5):
	n_hidden_states = np.random.poisson(lam=lam_hidden) + 2  # 2 for EOS + non-EOS
	hmm = {
		'n_hidden_states': n_hidden_states,
		'probs': sample_transition_matrix(n_hidden_states, alpha=alpha_hidden),
		'emission_probs': [
			sample_categorical_distribution(alpha_visible * np.ones(np.random.poisson(lam=lam_visible) + 1)) 
			for i in range(n_hidden_states)
		], 
	}
	hmm['start_idxs'] = np.cumsum([0] + [len(p) for p in hmm['emission_probs']])
	return hmm

def sample_from_hmm(hmm, N):
	n_hidden_states = hmm['n_hidden_states']
	sequence = []
	hidden_state = np.random.randint(0, n_hidden_states)
	while len(sequence) < N:
		observed_state = np.random.choice(range(len(hmm['emission_probs'][hidden_state])), p=hmm['emission_probs'][hidden_state])
		sequence.append(f'{hidden_state}_{observed_state}')
		hidden_state = np.random.choice(range(n_hidden_states), p=hmm['probs'][hidden_state])
	return sequence

def tup_str_to_tup(tup_str, _delimiter='_'):
	return list(map(int, tup_str.split(_delimiter)))

def hmm_sequence_to_str(hmm: Dict, sequence: List[Tuple[int, int]], vocab: List[str] = list('abcdefghijklmnopqrstuvwxyz'), delimiter: str = ' ') -> str:
	return delimiter.join(vocab[hmm['start_idxs'][hid_idx] + out_idx] for tup in sequence for hid_idx, out_idx in [tup_str_to_tup(tup)])

def get_vocab(N: int, base_vocab: List[str] = list('abcdefghijklmnopqrstuvwxyz')) -> List[str]:
	"""
	>>> print(get_vocab((((3+1)*3+1)*3), list('abc')))
	['a', 'b', 'c', 'aa', 'ab', 'ac', 'ba', 'bb', 'bc', 'ca', 'cb', 'cc', 'aaa', 'aab', 'aac', 'aba', 'abb', 'abc', 'aca', 'acb', 'acc', 'baa', 'bab', 'bac', 'bba', 'bbb', 'bbc', 'bca', 'bcb', 'bcc', 'caa', 'cab', 'cac', 'cba', 'cbb', 'cbc', 'cca', 'ccb', 'ccc']

	"""
	N += 1
	cur_vocab = [''] + base_vocab
	while len(cur_vocab) < N:
		cur_vocab = [''] + [f'{w}{c}' for w in cur_vocab for c in base_vocab]
	return cur_vocab[1:N]

def multilevel_markov_chain_sequence_to_str(mc: Dict, sequence: List[Tuple[int, int]], vocab: List[str] = list('abcdefghijklmnopqrstuvwxyz'), delimiter: str = ' ') -> str:
	def get_idx(tup, mc):
		if mc['n_levels'] == 1:
			return tup[0]-1
		return mc['start_idxs'][tup[0]-1] + get_idx(tup[1:], mc['mcs'][tup[0]-1])

	return delimiter.join(vocab[get_idx(tup_str_to_tup(tup_str), mc)] for tup_str in sequence)

if __name__ == '__main__':
	np.set_printoptions(suppress=True, precision=2)
	for i in range(3):
		set_seed(i)
		mc = sample_multilevel_markov_chain(lam=2., alpha=.1)
		# print(mc)
		vocab = get_vocab(mc['start_idxs'][-1])
		print(len(vocab))
		print(vocab)
		seq_full = []
		while sum(len(seq) for seq in seq_full) < 512: 
			seq = sample_from_multilevel_markov_chain(mc)
			seq_full.append(seq)
		seq_full_str = ' '.join([multilevel_markov_chain_sequence_to_str(mc, seq, vocab, delimiter='') for seq in seq_full])
		print(seq_full_str)

	print('---')

	for i in range(3):
		set_seed(i)
		hmm = sample_hmm(alpha_hidden=.1, alpha_visible=.2)
		# print(hmm)
		vocab = get_vocab(hmm['start_idxs'][-1])
		print(len(vocab))
		print(vocab)
		for i in range(5):
			seq = sample_from_hmm(hmm, 100)
			print(len(seq))
			print(seq[:5])
			seq_str = hmm_sequence_to_str(hmm, seq, vocab, delimiter=' ')
			print(seq_str)

	# print(get_vocab((((3+1)*3+1)*3), list('abc')))

