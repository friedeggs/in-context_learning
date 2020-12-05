
# from .content import (
# 	random_distinct_alpha_chars
# )

import pandas as pd

from .api import (
	get_completion_s,
	get_ppl_s,
)
from .content_lib import (
	random_distinct_alpha_chars,
	random_permutation,
)
from .dataset import (
	Dataset, FewShotDataset, FormattingDataset, FuncDataset, GPTDataset, IndexDataset, InputOutputDataset, IntDataset, ListDataset, NondeterministicDataset, SumDataset
)
# from .error_analysis import (
# 	analyze_errors,
# 	get_value_dict,
# 	match_templates,
# 	filter_templates,
# 	print_templates,
# )
from .form_lib import (
	space_separate,
)
from .gpt import GPT, read_cache
from .util import (
	permute,
)

class RandomDistinctAlphaCharsDataset(NondeterministicDataset):
	def __init__(self, n: int, **kwargs):
		self.n = n
		func = lambda idx: list(random_distinct_alpha_chars(self.n))
		super(RandomDistinctAlphaCharsDataset, self).__init__(func=func, **kwargs)

class RandomPermutationDataset(NondeterministicDataset):
	def __init__(self, n: int, **kwargs):
		self.n = n
		func = lambda idx: list(random_permutation(self.n))
		super(RandomPermutationDataset, self).__init__(func=func, **kwargs)

def main(argv):
# argv = []
# n_permutations = 5
	n_train = 3
	n_test = 5
	completion_kwargs = {}
	mock = 'submit' not in argv
	cache = read_cache()
	gpt = GPT(cache, mock)

	content = RandomDistinctAlphaCharsDataset(5)
	order_dataset = RandomPermutationDataset(5)
	# order = order_dataset[0]
	# dataset = SumDataset([content, order_dataset])
	content = FewShotDataset(content, n_train=n_train, n_test=n_test)
	order_dataset = FuncDataset(order_dataset, funcs=[
		lambda x: [x for _ in range(n_train + 1)],
	])
	sample = SumDataset([content, order_dataset])
	sample = FuncDataset(sample, funcs=[
		lambda x: list(zip(*x)),
	])

	formatted = FormattingDataset(sample, 
		lambda pair: space_separate(pair[0]), 
		lambda pair: space_separate(permute(pair[0], pair[1])),
		map=True,
	)
	x = FuncDataset(formatted, lambda _: _[-1][0])
	y = FuncDataset(formatted, lambda _: _[-1][1])
	prompt = InputOutputDataset(formatted)  # str
	response = GPTDataset(prompt, gpt, completion_kwargs)
	pred = FuncDataset(response, lambda _: get_completion_s(_, completion_kwargs))  # str
	correct = FuncDataset(SumDataset([pred, y]), lambda _: _[0] == _[1] if _[0] is not None else None)  # Optional[bool]
	ppl = FuncDataset(response, lambda _: get_ppl_s(_, completion_kwargs))  # str
	# incorrect_indices = [i for i, val in enumerate(correct) if val == False]  # float
	# def analyze_templates(idx):
	# 	_sample = sample[idx]
	# 	_pred = pred[idx]
	# 	_x = x[idx]
	# 	tgt_form_func = lambda _: form_lib.permute(_, _sample[-1][1])
	# 	value_dict = get_value_dict(TODO, [tgt_form_func])
	# 	templates = match_templates(_pred, value_dict)
	# 	print_templates(templates, None, _pred, _x)
	# 	templates_by_name = list(map(lambda x1: list(map(lambda x2: list(map(lambda x3: x3[0], x2)), x1)), templates))
	# 	return templates_by_name
	# templates = FuncDataset(incorrect_indices, analyze_templates)
	rows = SumDataset([x, pred, y, correct, ppl, prompt,], keys=['x', 'pred', 'y', 'correct', 'ppl', 'prompt',])
	_rows = [r for r in rows] # rows[:]?
	# df = pd.DataFrame(_rows) # , columns=column_names)
	# _templates = [None for _ in range(len(_rows))]
	# for i, template in zip(incorrect_indices, templates):
	# 	_templates[i] = template
	# df = df.assign(templates=_templates)
	# dataset = rows
	# for i, batch in enumerate(dataset):
	# 	if i >= 5:
	# 		break
	# 	print(batch)
	print(len(_rows))
	print(_rows)




