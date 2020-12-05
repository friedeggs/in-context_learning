import sys, os 
import copy
from collections import OrderedDict, namedtuple
from itertools import product as cartesian_product
import exrex
import json
import numpy as np
import pandas as pd
import random
import re
from termcolor import colored
import traceback
from typing import Any, Callable, Dict, List, Tuple, Optional

import util
from util import set_seed


def process(df_old, func, **kwargs):
	rows_new = []
	for idx, row_old in df_old.iterrows():
		row_new = func(idx=idx, **row_old, **kwargs)
		if util.is_iterable(row_new):
			rows_new.extend(row_new)
		else:
			rows_new.append(row_new)
	df_new = pd.DataFrame(rows_new)
	return df_new


class Example:
	def __init__(self):
		pass

class TaskSuite:
	def __init__(self, name: str, description: str, tasks: List[Union[Task, TaskSuite]]):
		self.name = name
		self.description = description
		self.tasks = tasks

	def add_task(self, task):
		self.tasks.append(task)

	def run_tasks(self):
		for task in self.tasks:
			task.run()

	def render(self):
		pass

class Task:
	def __init__(self, name: str, description: str, subtasks: List[Task]):
		self.name = name
		self.description = description
		self.subtasks = subtasks

	def initialize(self):
		pass

	def inflate(self, **kwargs):
		"""Set all primary keys
		"""
		pass

	def populate(self, df_old, populate_func, **kwargs):
		"""Fill all derivable keys
		"""
		rows_new = []
		for idx, row_old in df_old.iterrows():
			row_new = populate_func(idx=idx, **row_old, **kwargs)
			if util.is_iterable(row_new):
				rows_new.extend(row_new)
			else:
				rows_new.append(row_new)
		df_new = pd.DataFrame(rows_new)
		return df_new

	def generate(self):
		pass

	def process(self):
		table = self.initialize(); self.tables.append(table)
		for func in self.functions:
			table = func(table); self.tables.append(table)
		return table

	def run_tasks(self, model):
		pass

	def run(self):
		self.inflate()
		self.populate()
		self.run_tasks()
		self.analyze()

	def summarize(self):
		pass

	def add_task(self):
		pass

	def analyze(self):
		pass

	def get_view(self):
		pass

	def load(self, filename):
		pass

	def save(self, filename):
		pass

	def filter(self):
		pass

	def render(self):
		pass

	def perturb(self, example):
	"""Derived conditional subtask
	"""
		generating_row_idxs = self.lookup(example)
		table_idx = np.random.choice(range(generating_row_idxs))
		value_new = sample(attr, value_old, seed)
		example_new = copy.deepcopy(example)
		example_new[attr] = value_new
		for i in range(table_idx, len(self.tables)):
			self.tables[i].append(example_new)
		neighbors = sample()

		# idx=idx



'/foo/bar' # subtasks
# tags

class Experiment:
	pass

class View:
	pass

def main():
	date_task = Task(
		,
		functions=[

		],
	)

	adversarial_task = LinkedTask(
		date_task,
		# initialize = lambda: date_task.tables[-1],
		functions=[
		],
	)


