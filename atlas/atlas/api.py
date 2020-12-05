from collections import OrderedDict
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def get_completion(response, completion_kwargs, index: int = 0, prompt: Optional[str] = None) -> Optional[str]:
	if response is None:
		return None
	text = response['choices'][index]['text']
	if 'echo' in completion_kwargs and completion_kwargs['echo']:
		return text[len(prompt):]
	return text

def get_completions(response, completion_kwargs, prompt: Optional[str] = None) -> Optional[List[str]]:
	if response is None:
		return None
	completions = [
		get_completion(response, completion_kwargs, idx, prompt) 
		for idx in range(len(response['choices']))
	]
	return completions

def get_completion_s(response, completion_kwargs, prompt: Optional[str] = None) -> Optional[Union[str, List[str]]]:
	if response is None:
		return None
	completions = get_completions(response, completion_kwargs, prompt)
	if len(completions) == 1:
		completions = completions[0]
	return completions

def get_ppl(choice, completion_kwargs) -> Optional[float]:
	if choice is None:
		return None
	lps = choice['logprobs']['top_logprobs']
	lps = [OrderedDict(sorted(lp.items(), key=lambda x: -x[1])) for lp in lps if lp is not None]
	p = sum([list(lp.values())[0] for lp in lps])
	return np.exp(-p/max(len(lps), 1))

def get_ppl_s(response, completion_kwargs) -> Optional[Union[float, List[float]]]:
	if response is None:
		return None
	ps = [get_ppl(choice, completion_kwargs) for choice in response['choices']]
	if len(ps) == 1:
		ps = ps[0]
	return ps

def get_logprobs(choice, completion_kwargs):
	pass
