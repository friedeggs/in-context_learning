import sys, os
import abc
import functools
import logging; log = logging.getLogger(__name__)
import numpy as np
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .form_lib import io_format
from .util import make_immutable, is_iterable, MAX_INT, set_seed

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

	@functools.lru_cache(maxsize=10240)
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

	@abc.abstractmethod
	def getitem(self, idx):
		pass

	@abc.abstractmethod
	def __len__(self):
		pass

class FewShotDataset(Dataset):
	def __init__(self, dataset: Dataset, n_train: int, n_test: int, same_train: bool = False,
			exclude_train_from_test: bool = False, **kwargs):
		super(FewShotDataset, self).__init__(**kwargs)
		self.dataset = dataset
		self.n_train = n_train
		self.n_test = n_test
		self.same_train = same_train
		self.exclude_train_from_test = exclude_train_from_test

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx):
		if self.same_train:
			r = range(self.n_train)
		else:
			r = range(idx * self.n_train, (idx + 1) * self.n_train)
		if self.exclude_train_from_test:
			train_examples = [self.dataset.get_train(i) for i in r]
			test_example = []
			cntr = 0
			while len(test_example) == 0 and cntr < min(len(self.dataset), 5 * self.n_train):  # TODO n_test not len(self.dataset)?
				example = self.dataset.get_test(idx + cntr * 1_000_009)
				if example not in train_examples:
					test_example.append(example)
				else:
					log.debug('Skipping ' + str(example))
				cntr += 1
			if len(test_example) == 0:
				raise Exception(f'Could not find a test example for index {idx}.')
			return train_examples + test_example
		return [self.dataset.get_train(i) for i in r] + [self.dataset.get_test(idx)]

	def __len__(self):
		return self.n_test

class ProductDataset(Dataset):
	def __init__(self, datasets: List[Dataset], **kwargs):
		super(ProductDataset, self).__init__(**kwargs)
		self.datasets = datasets

	@functools.lru_cache(maxsize=10240)
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
	def __init__(self, datasets: List[Dataset], keys: Optional[List] = None, **kwargs):
		super(SumDataset, self).__init__(**kwargs)
		self.datasets = datasets
		self.keys = keys

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx):
		values = [ds[idx] for ds in self.datasets]
		if self.keys is not None:
			return {k: v for k, v in zip(self.keys, values)}
		return values

	def __len__(self):
		return min(list(map(len, self.datasets)))

class IndexDataset(Dataset):
	def __init__(self, n: int, **kwargs):
		super(IndexDataset, self).__init__(**kwargs)
		self.n = n

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx):
		return idx

	def __len__(self):
		return self.n

class ListDataset(Dataset):
	"""A standard dataset.
	"""
	def __init__(self, data: List, **kwargs):
		super(ListDataset, self).__init__(**kwargs)
		self.data = data

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)

class NondeterministicDataset(Dataset):
	N_INT_DATASETS = 0
	max_datasets = 1_000_000

	def __init__(self, func: Callable, offset: Optional[int] = None, **kwargs):
		super(NondeterministicDataset, self).__init__(**kwargs)
		if offset is None:
			offset = NondeterministicDataset.N_INT_DATASETS
		NondeterministicDataset.N_INT_DATASETS += 1
		self.offset = offset
		self.func = func

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx):
		max_datasets = max(NondeterministicDataset.N_INT_DATASETS, NondeterministicDataset.max_datasets)
		set_seed(max_datasets * idx + self.offset)
		return self.func(idx)

	def __len__(self):
		return MAX_INT

class IntDataset(NondeterministicDataset):
	def __init__(self, low: int, high: int, **kwargs):
		self.min = low
		self.max = high
		func = lambda idx: np.random.randint(self.min, self.max)
		super(IntDataset, self).__init__(func=func, **kwargs)

class FuncDataset(Dataset):
	def __init__(self, dataset, funcs: Union[Callable, List[Callable]], **kwargs):
		super(FuncDataset, self).__init__(**kwargs)
		self.dataset = dataset
		if not is_iterable(funcs):
			funcs = [funcs]
		self.funcs = funcs

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx: Optional[int] = None, item = None):
		item = item or self.dataset[idx]
		for func in self.funcs:
			item = func(item)
		return item

	def __len__(self):
		return len(self.dataset)

class FormattingDataset(Dataset):
	def __init__(self, dataset, src_form: Callable, tgt_form: Callable, map: bool = False, **kwargs):
		super(FormattingDataset, self).__init__(**kwargs)
		self.dataset = dataset
		self.src_form = src_form
		self.tgt_form = tgt_form
		self.map = map

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx: Optional[int] = None, item = None):
		item = item or self.dataset[idx]
		fmt = lambda x: (self.src_form(x), self.tgt_form(x))
		if self.map:
			result = list(map(fmt, item))
		else:
			result = fmt(item)
		return result

	def __len__(self):
		return len(self.dataset)

class InputOutputDataset(Dataset):
	def __init__(self, dataset, x_label: Optional[str] = 'Input', y_label: Optional[str] = 'Output',
			formatter: Optional[Callable] = None, 
			include_y: bool = False,
			intra_separator: str = ': ',
			x_y_separator: str = '\n',
			prefix: Optional[str] = None,
			transform: Callable = lambda x: x,
			**kwargs):
		super(InputOutputDataset, self).__init__(**kwargs)
		self.dataset = dataset
		self.x_label = x_label
		self.y_label = y_label
		self.formatter = formatter
		self.include_y = include_y
		self.intra_separator = intra_separator
		self.x_y_separator = x_y_separator
		self.prefix = prefix
		self.transform = transform

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx: Optional[int] = None, item = None):
		item = item or self.dataset[idx]  # List[Tuple[Any, Any]]
		return io_format(item, 
			x_label=self.x_label, y_label=self.y_label, 
			formatter=self.formatter, include_y=self.include_y, 
			intra_separator=self.intra_separator, 
			x_y_separator=self.x_y_separator, 
			prefix=self.prefix, transform=self.transform)

	def __len__(self):
		return len(self.dataset)

class GPTDataset(Dataset):
	def __init__(self, dataset, gpt, completion_kwargs: dict, **kwargs):
		super(GPTDataset, self).__init__(**kwargs)
		self.dataset = dataset
		self.gpt = gpt
		self.completion_kwargs = completion_kwargs

	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx: Optional[int] = None, item = None):
		item = item or self.dataset[idx]  # str
		return self.gpt.complete(item, self.completion_kwargs)

	def __len__(self):
		return len(self.dataset)

class IdentityDataset(Dataset):
	@functools.lru_cache(maxsize=10240)
	def getitem(self, idx):
		return idx

	def __len__(self):
		return MAX_INT

