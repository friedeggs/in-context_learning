import multiprocessing
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple, Optional

from .dataset import Dataset, FewShotDataset, IndexDataset, IntDataset, ListDataset, SumDataset
# from .gpt import GPT, completion_kwargs  # TODO fake import

class FuncDataset(Dataset):
	def __init__(self, dataset, funcs: List[Callable], **kwargs):
		super(FuncDataset, self).__init__(**kwargs)
		self.dataset = dataset
		self.funcs = funcs

	def getitem(self, idx: Optional[int] = None, item = None):
		item = item or self.dataset[idx]
		for func in self.funcs:
			item = func(item)
		return item

	def __len__(self):
		return len(self.dataset)

class FormattingDataset(Dataset):
	def __init__(self, dataset, src_form: Callable, tgt_form: Callable, **kwargs):
		super(FormattingDataset, self).__init__(**kwargs)
		self.dataset = dataset
		self.src_form = src_form
		self.tgt_form = tgt_form

	def getitem(self, idx: Optional[int] = None, item = None):
		item = item or self.dataset[idx]
		return (self.src_form(item), self.tgt_form(item))

	def __len__(self):
		return len(self.dataset)

def create_date_dataset():
	year = IntDataset(1970, 2020)
	month = IntDataset(1, 12+1)
	day = IntDataset(1, 28+1)

	dataset = SumDataset([year, month, day])
	dataset = FormattingDataset(dataset, 
		# lambda x: '/'.join(map(lambda n: f'{n:02d}', x)), 
		lambda x: '-'.join(map(lambda n: f'{n:02d}', x)),
		lambda x: '!'.join([''] + list(map(lambda n: f'{n:02d}', [x[1],x[2],x[0]])) + ['']),
	)
	indexer = IndexDataset(len(dataset))
	dataset = FewShotDataset(dataset, n_train=50, n_test=2, same_train=False)
	indexer = FewShotDataset(indexer, n_train=50, n_test=2, same_train=False)
	def fmt(tups):
		return '\n'.join(map(lambda t: f'Input: {t[0]}\nOutput: {t[1]}', tups))
	dataset = FuncDataset(dataset, 
		funcs=[
			# lambda x: '\n'.join(x[:-1]),
			fmt,
			# lambda x: (' = '.join('\n'.join(x).split(' = ')[:-1] + ['']).rstrip(), x[-1].split(' = ')[-1]),
			# lambda x: (' = '.join('\n'.join(x).split(' = ')[:-1] + ['']).rstrip(), x[-1].split(' = ')[-1]),
			# lambda x: gpt.complete(x)
		]
	)

	# year = intd(1970, 2020)
	# dataset = factory				\
	# 	.sum([year, month, day])	\
	# 	.few_shot()					\
	# 	.to_dataset()
	# dataset[0]
	# dataset.get('sum', None, )
	return indexer, dataset

# replace "<x>Dataset" with "Gen<x>"?

def create_arithmetic_dataset():
	a = IntDataset(10, 100)
	b = IntDataset(10, 100)
	dataset = SumDataset([a, b])
	dataset = FuncDataset(dataset, 
		funcs=[
			lambda x: x + [sum(x)],
			lambda x: f'{x[0]} + {x[1]} = {x[2]}',
		]
	)
	indexer = IndexDataset(len(dataset))
	dataset = FewShotDataset(dataset, n_train=3, n_test=2, same_train=False)
	indexer = FewShotDataset(indexer, n_train=3, n_test=2, same_train=False)
	dataset = FuncDataset(dataset, 
		funcs=[
			# lambda x: '\n'.join(x[:-1]),
			lambda x: (' = '.join('\n'.join(x).split(' = ')[:-1] + ['']).rstrip(), x[-1].split(' = ')[-1]),
			# lambda x: gpt.complete(x)
		]
	)
	return indexer, dataset

def run_parallel(func, dataset, N_PARALLEL=8):
	with multiprocessing.Pool(N_PARALLEL) as p:
		list(tqdm(p.imap(func, dataset), total=len(dataset)))

def run():
	# gpt = GPT()
	indexer, dataset = create_date_dataset()
	# dates = dataset.dataset.dataset
	# form = dataset.dataset
	for idx, (index, el) in enumerate(zip(indexer, dataset)):
		if idx >= 1:
			break
		# print(idx, index, el)
		print(el)
		print()
		# year, month, day = dates[index[0]]
		# # print(year, month, day)
		# print(form.getitem(item=[year, month, day]))
	indexer, dataset = create_arithmetic_dataset()

	# results = Result()
	# for idx, (index, el) in enumerate(zip(indexer, dataset)):
	# 	if idx >= 5:
	# 		break
	# 	print(idx, index, el)
		# year, month, day = dates[index[0]]
		# # print(year, month, day)
		# print(form.getitem(item=[year, month, day]))
		# x, y = el
		# pred = gpt.complete(x, **completion_kwargs)
		# results.add(x, pred, y)

	# preds_inc, inds = results.get_incorrect(return_indices=True)
	# dataset = ListDataset(preds_inc)

	# for idx, (index, el) in enumerate(zip(indexer, dataset)):
	# 	if idx >= 5:
	# 		break
	# 	print(idx, index, *el)



