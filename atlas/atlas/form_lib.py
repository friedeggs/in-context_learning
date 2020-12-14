
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def comma_separate(vals):
	return ', '.join(vals)

def space_separate(vals):
	return ' '.join(vals)

def io_format(item, x_label: Optional[str] = 'Input', y_label: Optional[str] = 'Output', formatter: Optional[Callable] = None, include_y: bool = False, intra_separator: str = ': ', prefix: Optional[str] = None, transform: Callable = lambda x: x):
	default_formatter = lambda tup: f'{x_label}{intra_separator}{transform(tup[0])}\n{y_label}{intra_separator}{transform(tup[1])}'
	formatter = formatter or default_formatter
	prompt = '\n'.join(list(map(formatter, item)))
	if not include_y:
		prompt = intra_separator.join(prompt.split(intra_separator)[:-1]) + intra_separator.rstrip()
	if prefix is not None:
		prompt = prefix + prompt
	return prompt