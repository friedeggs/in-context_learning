
# from .. import dataset
from ..dataset import Dataset, ProductDataset, FewShotDataset

class A(Dataset):
	def __init__(self, n):
		super(A, self).__init__()
		self.n = n

	def getitem(self, idx):
		return idx 

	def __len__(self):
		return self.n

class B(Dataset):
	def getitem(self, idx):
		return chr(ord('a')+idx) 

	def __len__(self):
		return 7

def test_product_dataset():
	a1 = A(2)
	a2 = A(3)
	b = B()
	c = ProductDataset(
		datasets=[a1, a2, b],
	)
	print(len(c))
	for idx, el in enumerate(c):
		print(idx, el)

def test_fewshot_dataset():
	a = A(1_000_000_000)
	d = FewShotDataset(dataset=a, n_train=5, n_test=2)
	for idx, el in enumerate(d):
		if idx >= 5:
			break
		print(idx, el)

