import sys, os
import abc
import functools
import numpy as np
import traceback
from typing import Any, Callable, Dict, List, Tuple, Optional

from .util import MAX_INT, set_seed

def handle_error(e):
	# traceback.print_exc(sys.stderr)
	raise Exception(e)

class Dataset(abc.ABC):
	def __init__(self, splits: list = [], MAX_SPLITS: int = 100, handle_error=handle_error, is_finite: bool = True): 
		self.max_splits = max(MAX_SPLITS, 2 + len(splits))
		self.split_ids = dict([('train', 0), ('test', 1)] + [(name, idx+2) for idx, name in enumerate(splits)])
		self.handle_error = handle_error
		self.is_finite = is_finite

	def get_train(self, idx):
		return self.get_split('train', idx)

	def get_test(self, idx):
		return self.get_split('test', idx)

	def get_split(self, name, idx):
		idx2 = self.max_splits * idx + self.split_ids[name]
		return self[idx2]

	def __getitem__(self, idx):
		if self.is_finite:
			if idx >= len(self):
				raise IndexError
		else:
			idx = idx % len(self)
		try:
			return self.getitem(idx)
		except Exception as e:
			self.handle_error(e)

	@functools.lru_cache(maxsize=1024)  # TODO check if is inherited 
	@abc.abstractmethod
	def getitem(self, idx):
		pass

	@abc.abstractmethod
	def __len__(self):
		pass

class FewShotDataset(Dataset):
	def __init__(self, dataset: Dataset, n_train: int, n_test: int, same_train: bool = False, **kwargs):
		super(FewShotDataset, self).__init__(**kwargs)
		self.dataset = dataset
		self.n_train = n_train
		self.n_test = n_test
		self.same_train = same_train

	def getitem(self, idx):
		if self.same_train:
			r = range(self.n_train)
		else:
			r = range(idx * self.n_train, (idx + 1) * self.n_train)
		return [self.dataset.get_train(i) for i in r] + [self.dataset.get_test(idx)]

	def __len__(self):
		return MAX_INT

class ProductDataset(Dataset):
	def __init__(self, datasets: List[Dataset], **kwargs):
		super(ProductDataset, self).__init__(**kwargs)
		self.datasets = datasets

	def getitem(self, idx):
		lens = list(map(len, self.datasets))
		sample = []
		for mod, ds in zip(reversed(lens), reversed(self.datasets)):
			idx2 = idx % mod
			idx = idx // mod
			sample.append(ds[idx2])
		return list(reversed(sample))

	def __len__(self):
		return min(MAX_INT, np.prod(list(map(len, self.datasets))))

class SumDataset(Dataset):
	def __init__(self, datasets: List[Dataset], **kwargs):
		super(SumDataset, self).__init__(**kwargs)
		self.datasets = datasets

	def getitem(self, idx):
		return [ds[idx] for ds in self.datasets]

	def __len__(self):
		return min(list(map(len, self.datasets)))

class IndexDataset(Dataset):
	def __init__(self, n: int, **kwargs):
		super(IndexDataset, self).__init__(**kwargs)
		self.n = n

	def getitem(self, idx):
		return idx

	def __len__(self):
		return self.n

class ListDataset(Dataset):
	def __init__(self, data: List, **kwargs):
		super(ListDataset, self).__init__(**kwargs)
		self.data = data

	def getitem(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)

class NondeterministicDataset(Dataset):
	N_INT_DATASETS = 0

	def __init__(self, func: Callable, max_datasets: int = 100, offset: Optional[int] = None, **kwargs):
		super(NondeterministicDataset, self).__init__(**kwargs)
		if offset is None:
			offset = NondeterministicDataset.N_INT_DATASETS
		NondeterministicDataset.N_INT_DATASETS += 1
		self.max_datasets = max(NondeterministicDataset.N_INT_DATASETS, max_datasets) 
		self.offset = offset
		self.func = func

	def getitem(self, idx):
		set_seed(self.max_datasets * idx + self.offset)
		return self.func(idx)

	def __len__(self):
		return MAX_INT

class IntDataset(NondeterministicDataset):
	def __init__(self, low: int, high: int, **kwargs):
		self.min = low
		self.max = high
		func = lambda idx: np.random.randint(self.min, self.max)
		super(IntDataset, self).__init__(func=func, **kwargs)


