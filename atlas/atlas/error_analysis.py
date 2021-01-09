import calendar
from collections import defaultdict
import copy
import logging; log = logging.getLogger(__name__)
import numpy as np
import regex
from termcolor import colored
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# from sample import (
# 	load_df,
# 	save_df,
# )
from .schema import (
	create_date_schema,
	sample,
	FormPair,
)
from .util import (
	flatten,
	set_seed,
)

DateSchema = create_date_schema()

def get_form_primitives(content, tgt_forms: List[Callable]):
	forbidden = {
		'': 'XXXXXXXXX',
		'str': 'XXXXXXXXX',
		'str_': 'XXXXXXXXX', # numpy
		'int': 9675278978,
		'int8': 9675278978,
		'int16': 9675278978,
		'int32': 9675278978,
		'int64': 9675278978,
	} # TODO make sure unused
	types = [type(c).__name__ for c in content]
	# log.info(types)
	dummy_content = [forbidden[t] for t in types]
	form_primitives = []
	for tgt_form in tgt_forms:
		dummy_formatted = tgt_form(dummy_content)
		for v in set(forbidden.values()):
			dummy_formatted = forbidden[''].join(dummy_formatted.split(str(v)))
		cur_form_primitives = list(set(dummy_formatted.split(str(forbidden['']))))
		cur_form_primitives = list(filter(lambda x: x, cur_form_primitives))
		form_primitives.extend(cur_form_primitives)
	form_primitives_dict = {f'fp{i}': fp for i, fp in enumerate(form_primitives)}
	return form_primitives_dict

def get_value_dict(content, tgt_forms: List[Callable]):
	t = type(content)
	forbidden = 96752 # TODO deal with types, argh # TODO make sure unused
	dummy_content = t(*([forbidden] * len(content._fields)))
	form_primitives = []
	for tgt_form in tgt_forms:
		cur_form_primitives = list(set(tgt_form(dummy_content).split(str(forbidden))))
		cur_form_primitives = list(filter(lambda x: x, cur_form_primitives))
		form_primitives.extend(cur_form_primitives)
	value_dict = {
		**{f: getattr(content, f) for f in content._fields},
		**{f'fp{i}': fp for i, fp in enumerate(form_primitives)},
		**{'<2-digit>': r'\d\d',},
		**{'<5-digit>': r'\d\d\d\d\d',},
		**{'Input': 'Input'},  # TODO use io_format_args instead of hardcoding
		**{'Output': 'Output'},
	}
	value_dict = add_neighbors(value_dict)
	return value_dict

def get_templates(content, tgt_form):
	output_text = tgt_form.function(content)
	value_dict = get_value_dict(content, tgt_form)
	templates = match_templates(output_text, value_dict)
	return templates

def add_neighbors(base_value_dict):
	new_value_dict = {}
	for k, v in base_value_dict.items():
		new_value_dict[k] = v

		if isinstance(v, int):
			new_value_dict[k] = f'{v:02d}'

		if isinstance(v, int) and f'{v:02d}' != str(v):
			new_value_dict[f'{k}_not-fmt02d'] = str(v)

		if isinstance(v, int) and v > 0: # TODO make 1?
			new_value_dict[f'{k}_minus-one'] = f'{v-1:02d}'
		if isinstance(v, int) and v > 0 and f'{v-1:02d}' != str(v-1):
			new_value_dict[f'{k}_minus-one_not-fmt02d'] = str(v-1)

		if isinstance(v, int): # TODO upper bound?
			new_value_dict[f'{k}_plus-one'] = f'{v+1:02d}'
		if isinstance(v, int) and f'{v+1:02d}' != str(v+1):
			new_value_dict[f'{k}_plus-one_not-fmt02d'] = str(v+1)

		if k == 'month_name':
			try:
				new_value_dict['month'] = list(calendar.month_abbr).index(v)
			except Exception:
				pass
		if k == 'month':
			try:
				new_value_dict['month_name'] = calendar.month_abbr[v]
			except Exception:
				pass

		if k == 'year':
			new_value_dict['year_2:4'] = str(v)[2:4]

	base_value_dict = new_value_dict
	new_value_dict = {}
	for k, v in base_value_dict.items():
		if v != '':
			new_value_dict[k] = v
	return new_value_dict

def get_solutions(backtrack, idx, arrs):
	if idx == 0:
		# return list(map(lambda _: list(reversed(_)), solutions))
		return [list(reversed(arrs))]
	res = []
	els = defaultdict(list)
	for name, matched in backtrack[idx]:
		# els[len(matched)].append(name)
		els[len(matched)].append((name, matched))
	for length, lst in els.items():
		solutions = get_solutions(backtrack, idx-length, arrs + [lst])
		# res.extend(get_solutions(backtrack, idx-len(matched), arrs.append([name])))
		res.extend(solutions)
		# import pdb; pdb.set_trace()
	return res

def match_templates(output_text, value_dict):
	
	matches = [{} for _ in range(len(output_text)+1)]
	dp = [0 for _ in range(len(output_text)+1)]
	backtrack = [[] for _ in range(len(output_text)+1)]

	for name, pattern in value_dict.items():
		for match in regex.finditer(str(pattern), output_text, overlapped=True):
			start_idx = match.span()[0]
			matched = match.group()
			# if pattern == r'\d\d' and matched in value_dict.values():
			if [name[0], name[-1]] == list('<>') and matched in value_dict.values():  # regex
				continue
			matches[start_idx+len(matched)][name] = matched
	# print(matches)
	
	for i in range(len(output_text)+1):
		dp[i] = dp[max(0, i-1)]
		if i > 0:
			backtrack[i] = [(output_text[i-1], output_text[i-1])]
		for name, matched in matches[i].items():
			# begin argmax solution
			newval = dp[i-len(matched)] + len(matched)
			if newval > dp[i]:
				backtrack[i] = []
				dp[i] = newval
			if newval == dp[i]:
				# end argmax solution
				backtrack[i].append((name, matched))
	# print(backtrack)

	# solutions = []
	# idx = len(output_text) - 1
	# while idx != 0:
	# 	for name, matched in backtrack[i]:

	return get_solutions(backtrack, len(output_text), [])

def getitem(lst, index, default_value=None):
	try:
		return lst[index]
	except IndexError:
		return default_value

def print_templates(templates, gt_template=None, output_text=None, input_text=None):
	if input_text is not None and output_text is not None:
		print(f'{input_text} -> {output_text}')
	elif output_text is not None:
		print(output_text)
	for solution in templates:
		for i in range(max(map(len, solution))):
			line = ''
			# width = max(map(len, map(lambda _: getitem(_, i, ''), solution))) + 1
			for index in range(len(solution)):
				# width = max(map(len, solution[index])) + 1
				width = max(map(lambda _: len(_[0]), solution[index])) + 1
				val = getitem(solution[index], i, ('', None))[0]
				if isinstance(val, tuple):
					val = '_'.join(val)
				# val = '{0: <{width}}'.format(val, width=width)
				# line += val + colored('|', 'magenta')
				line += f' {val:{width}s}' + colored('|', 'magenta')
			print(line)
		exemplar = list(map(lambda x: x[0][1], solution))
		if output_text is not None:
			assert ''.join(exemplar) == output_text
		print('')
	if gt_template is not None:
		predicted_templates = list(map(lambda s: list(map(lambda x: x[0][0], s)), templates))
		print(gt_template in predicted_templates)
	print('')

def test_match_templates():
	test_examples = [
		('7-Mar-1', '!01!1!7-Mar-1', ['!', ['day', 'fmt_02d'], '!', ['day'], '!', ['input_text']]),
		('7-Mar-1', '!7!Mar!1!7!Mar', ['!', 'year', '!', 'month', '!', 'day', '!', 'year', '!', 'month']),
		('7-Mar-1', '!03!01!1970!', ['!', ['month', 'number', 'fmt_02d'], '!', ['day', 'fmt_02d'], '!', ['year', 'projected'], '!']),
		('7-Mar-1', '!Mar!1!7!', ['!', 'month', '!', 'day', '!', 'year', '!']),
		('7-Mar-1', '!05!1!7!', ['!', 'month', '!', 'day', '!', 'year', '!']),
	]
	gt_template = ['!', 'month', '!', 'day', '!', 'year', '!']
	# TODO add scores
	value_dict = {
		'year': '7',
		'month': 'Mar',
		'day': '1',
		'day_fmt02d': '01',
		'month_number': '3',
		'month_fmt02d': '03',
		# 'month_number_fmt02d': '03',
		'input_text': '7-Mar-1',
		# 'extra': '1',
		'year_proj': '1970',
		# ('year', 'proj'): '1970',
		# 'dummy': '!1!',
		# '*0': '0',
		'2-digit': r'\d\d',
	}
	for input_text, output_text, expected_template in test_examples:
		print(f'{input_text} -> {output_text}')
		solutions = match_templates(output_text, value_dict)
		# import pdb; pdb.set_trace()
		for solution in solutions:
			for i in range(max(map(len, solution))):
				line = ''
				# width = max(map(len, map(lambda _: getitem(_, i, ''), solution))) + 1
				for index in range(len(solution)):
					# width = max(map(len, solution[index])) + 1
					width = max(map(lambda _: len(_[0]), solution[index])) + 1
					val = getitem(solution[index], i, ('', None))[0]
					if isinstance(val, tuple):
						val = '_'.join(val)
					# val = '{0: <{width}}'.format(val, width=width)
					# line += val + colored('|', 'magenta')
					line += f' {val:{width}s}' + colored('|', 'magenta')
				print(line)
			exemplar = list(map(lambda x: x[0][1], solution))
			assert ''.join(exemplar) == output_text
			# print('')
		predicted_template = list(map(lambda x: x[0][0], solutions[0]))
		print(predicted_template)
		predicted_templates = list(map(lambda s: list(map(lambda x: x[0][0], s)), solutions))
		print(gt_template in predicted_templates)
		print('')

def test_end_to_end():
	date_schema = create_date_schema()
	schema_type = date_schema
	# poss_fc = [
	# 	(FormPair(0, 4), schema_type(*([0] * len(schema_type._fields))))
	# ]
	set_seed()
	sm = sample(schema_type, 0, 4, schema_type(*([0] * len(schema_type._fields))))
	print(sm.src_form, sm.tgt_form)
	content = sm.content
	# input_text = sm.src_form
	# output_text = sm.tgt_form
	tgt_form_func = schema_type.forms['tgt_form'][4]
	# templates = get_templates(content, tgt_form_func)
	# # gt_template = ['!', 'month', '!', 'day', '!', 'year', '!']
	# gt_template = ['fp0', 'month_fmt02d', 'fp0', 'day_fmt02d', 'fp0', 'year', 'fp0']
	# print_templates(templates, gt_template, output_text)
	value_dict = get_value_dict(content, tgt_form_func)
	for idx, form in schema_type.forms['tgt_form'].items():
		if idx != 4:
			continue
		# form = schema_type.forms['tgt_form'][2]
		value_dict = get_value_dict(content, form)
		text = form.function(content)
		templates = match_templates(text, value_dict)
		templates = filter_templates(templates, 'fp0')
		# template = sorted(templates, key=len)[0]
		# print(template)
		# print_templates([template], None, text)
		print_templates(templates, None, text)

def filter_templates(templates, fp0):
	def alternating_fp0(template):
		# lst1 = list(map(lambda x: x[0][0], template[::2]))
		# lst2 = [fp0] * ((len(template) + 1) // 2)
		# print(template)
		# print(template[::2])
		# print(lst1)
		# print(lst2)
		return list(map(lambda x: x[0][0], template[::2])) == [fp0] * ((len(template) + 1) // 2)
	templates = list(filter(alternating_fp0, templates))
	return templates

def analyze_errors(df, filename):
	from sample import save_df
	date_schema = create_date_schema()
	schema_type = date_schema
	errors = []
	for row in df.itertuples():
		content = eval(row.content)
		tgt_form_func = schema_type.forms['tgt_form'][4]
		value_dict = get_value_dict(content, tgt_form_func)
		# text = form.function(content)
		x = row.x
		text = row.pred
		# print(content)
		# print(value_dict)
		# print(text)
		templates = match_templates(text, value_dict)
		templates = filter_templates(templates, 'fp0')
		print_templates(templates, None, text, x)
		# templates_by_name = list(map(lambda s: list(map(lambda x: x[0][0], s)), templates))

		templates_by_name = list(map(lambda x1: list(map(lambda x2: list(map(tuple, x2)), x1)), templates))
		templates_by_name = list(map(lambda x1: list(map(lambda x2: list(map(lambda x3: x3[0], x2)), x1)), templates))
		# if text == '!12!13!2017!':
		# 	import pdb; pdb.set_trace()
		errors.append(templates_by_name)
	df = df.assign(templates=errors)
	df = save_df(filename, df)

def run_error_analysis():
	from sample import load_df
	df = load_df()
	analyze_errors(df[(df.engine == 'davinci') & (df.num_examples == 5) & (df.rel != 'EQUALS')], filename='results_error_analysis_templates.csv')
	analyze_errors(df[(df.engine == 'ada') & (df.num_examples == 15) & (df.rel != 'EQUALS')], filename='results_error_analysis_templates.csv')
	analyze_errors(df[(df.engine == 'davinci') & (df.num_examples == 3) & (df.rel != 'EQUALS')], filename='results_error_analysis_templates.csv')
	analyze_errors(df[(df.engine == 'curie') & (df.num_examples == 15) & (df.rel != 'EQUALS')], filename='results_error_analysis_templates.csv')
	analyze_errors(df[(df.engine == 'curie') & (df.num_examples == 10) & (df.rel != 'EQUALS')], filename='results_error_analysis_templates.csv')
	analyze_errors(df[(df.engine == 'curie') & (df.num_examples == 5) & (df.rel != 'EQUALS')], filename='results_error_analysis_templates.csv')
	analyze_errors(df[(df.engine == 'babbage') & (df.num_examples == 15) & (df.rel != 'EQUALS')], filename='results_error_analysis_templates.csv')


if __name__ == '__main__':
	# test_match_templates()
	# test_end_to_end()
	run_error_analysis()

# extend to log probs?
# 	- beam search the top predictions when variance is high, and apply template search to the predictions


# for remaining unmatched, apply Levenshtein distance
# 	prefer unmatched in input

# 			# dp[i] = max(dp[i], dp[i-len(matched)] + len(matched))
# 		# matches = list(regex.finditer(pattern, output_text, overlapped=True))
# 	# for i in range(len(output_text)):
# 	# 	for v, neighbors in value_dict.items():


# template:
# 	constant
# 	value
# 	transformation applied to value
# 		numerical: plus 1, close 
# 		copy 
# 		string: take substring (more generally, levenshtein distance edits)
