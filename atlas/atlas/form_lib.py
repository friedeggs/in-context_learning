
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def comma_separate(vals):
	return ', '.join(vals)

def space_separate(vals):
	return ' '.join(vals)

def str_separate(sep=' '):
	return lambda vals: sep.join(vals)

def io_format(item, x_label: Optional[str] = 'Input', y_label: Optional[str] = 'Output', formatter: Optional[Callable] = None, include_y: bool = False, intra_separator: str = ': ', x_y_separator: str = '\n', prefix: Optional[str] = None, suffix: Optional[str] = None, transform: Callable = lambda x: x):
	default_formatter = lambda tup: f'{x_label}{intra_separator}{transform(tup[0])}{x_y_separator}{y_label}{intra_separator}{transform(tup[1])}'
	formatter = formatter or default_formatter
	prompt = '\n'.join(list(map(formatter, item)))
	if not include_y:
		if intra_separator:
			prompt = intra_separator.join(prompt.split(intra_separator)[:-1]) + intra_separator.rstrip()
		else:
			prompt = x_y_separator.join(prompt.split(x_y_separator)[:-1]) + x_y_separator.rstrip()
	if prefix is not None:
		prompt = prefix + prompt
	if suffix is not None and include_y:
		prompt = prompt + suffix
	return prompt