from collections import defaultdict, OrderedDict
import logging; log = logging.getLogger(__name__)
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .util import fix

def get_completion(response, completion_kwargs: Dict, index: int = 0, prompt: Optional[str] = None, strip_whitespace: list = ['left']) -> Optional[str]:
	if response is None:
		return None
	text = response['choices'][index]['text']
	if 'echo' in completion_kwargs and completion_kwargs['echo']:
		text = text[len(prompt):]
	if 'left' in strip_whitespace:
		text = text.lstrip()
	if 'right' in strip_whitespace:
		text = text.rstrip()
	return text

def get_completions(response, completion_kwargs: Dict, prompt: Optional[str] = None) -> Optional[List[str]]:
	if response is None:
		return None
	completions = [
		get_completion(response, completion_kwargs, idx, prompt) 
		for idx in range(len(response['choices']))
	]
	return completions

def get_completion_s(response, completion_kwargs: Dict, prompt: Optional[str] = None) -> Optional[Union[str, List[str]]]:
	if response is None:
		return None
	completions = get_completions(response, completion_kwargs, prompt)
	if len(completions) == 1:
		completions = completions[0]
	return completions

def get_ppl(choice, completion_kwargs, prompt: Optional[str] = None, completion_only=True, n=1, keys: Union[bool, Optional[List[str]]] = None) -> Optional[float]:
	top_logprobs = get_top_logprobs(choice, completion_kwargs, prompt, completion_only, n, keys)
	# try:
	p = top_logprobs.sum()
	return np.exp(-p/max(len(top_logprobs.reshape(-1)), 1))
	# except Exception as e:
	# 	log.warn(e)
	# 	return np.nan

def get_ppl_s(response, completion_kwargs: Dict, prompt: Optional[str] = None, completion_only=True, n=1, keys: Union[bool, Optional[List[str]]] = None) -> Optional[Union[float, List[float]]]:
	if response is None:
		return None
	ps = [get_ppl(choice, completion_kwargs, prompt, completion_only, n, keys) for choice in response['choices']]
	if len(ps) == 1:
		ps = ps[0]
	return ps

def get_top_logprobs(choice, completion_kwargs, prompt: Optional[str] = None, completion_only=True, n=1, keys: Union[bool, Optional[List[str]]] = None) -> Optional[Union[Dict[str, float], List[float]]]:
	if choice is None:
		return None
	lps = choice['logprobs']['top_logprobs']
	if len(lps) > 1 and not isinstance(lps[1], OrderedDict):
		lps = [OrderedDict(sorted(lp.items(), key=lambda x: -x[1])) if lp is not None else None for lp in lps]
		choice['logprobs']['top_logprobs'] = lps
	if len(lps) > 0 and lps[0] is None:
		lps = lps[1:]
	start_idx = 0
	if 'echo' in completion_kwargs and completion_kwargs['echo'] and completion_only:
		s = ''
		while s != prompt:
			s += choice['logprobs']['tokens'][start_idx]
			start_idx += 1
		if choice['logprobs']['top_logprobs'][0] is None:
			start_idx -= 1
	lps = lps[start_idx:]
	end_idx = len(lps)
	if 'stop' in completion_kwargs:
		tokens = completion_kwargs['stop']
		if isinstance(tokens, str):
			tokens = [tokens]
		if choice['logprobs']['top_logprobs'][0] is None:
			start_idx += 1
		for token in tokens:
			try:
				end_idx = min(end_idx, choice['logprobs']['tokens'][start_idx:].index(token))
			except (IndexError, ValueError):
				pass
	lps = lps[:end_idx]
	if keys == True:
		top_logprobs = [{list(lp.keys())[i]: list(lp.values())[i] for i in range(n)} for lp in lps]
	elif keys is not None:
		lps = [defaultdict(lambda: -np.inf, lp) for lp in lps]
		top_logprobs = [{k: lp[k] for k in keys} for lp in lps]
	else:
		top_logprobs = np.array([[list(lp.values())[i] for i in range(n)] for lp in lps]).squeeze()
	return top_logprobs

def get_top_logprobs_s(response, completion_kwargs: Dict, prompt: Optional[str] = None, completion_only=True, n=1, keys: Optional[List[str]] = None) -> Optional[Union[Dict[str, float], List[float], List[Dict[str, float]], List[List[float]]]]:
	if response is None:
		return None
	tlps = [get_top_logprobs(choice, completion_kwargs, prompt, completion_only, n, keys) for choice in response['choices']]
	if len(tlps) == 1:
		tlps = tlps[0]
	return tlps

def get_top_tokens(choice, completion_kwargs, prompt: Optional[str] = None, completion_only=True) -> Optional[List[str]]:
	if choice is None:
		return None
	top_tokens = choice['logprobs']['tokens']
	if 'echo' in completion_kwargs and completion_kwargs['echo'] and completion_only:
		s = ''
		start_idx = 0
		while s != prompt:
			s += choice['logprobs']['tokens'][start_idx]
			start_idx += 1
		top_tokens = top_tokens[start_idx:]
	end_idx = top_tokens[::-1].index('\n')  # TODO doesn't work for '\n' with other whitespace; assumes one '\n'
	top_tokens = top_tokens[:fix(-end_idx)]
	return top_tokens

def get_top_tokens_s(response, completion_kwargs: Dict, prompt: Optional[str] = None, completion_only=True) -> Optional[Union[List[str], List[List[str]]]]:
	if response is None:
		return None
	tlps = [get_top_tokens(choice, completion_kwargs, prompt, completion_only) for choice in response['choices']]
	if len(tlps) == 1:
		tlps = tlps[0]
	return tlps

def get_completion_logprobs(choice, completion_kwargs: Dict, prompt: Optional[str] = None, strip_whitespace: list = ['left', 'right']) -> str:
	kvs = get_top_logprobs(choice, completion_kwargs, prompt, completion_only=prompt is not None, n=1, keys=True)
	completion = ''.join([list(kv.keys())[0] for kv in kvs])
	if 'left' in strip_whitespace:
		completion = completion.lstrip()
	if 'right' in strip_whitespace:
		completion = completion.rstrip()
	return completion

def get_completion_logprobs_s(response, completion_kwargs: Dict, prompt: Optional[str] = None, completion_only=True, strip_whitespace: list = ['left', 'right']) -> Optional[Union[str, List[str]]]:
	if response is None:
		return None
	res = [get_completion_logprobs(choice, completion_kwargs, prompt, strip_whitespace) for choice in response['choices']]
	if len(res) == 1:
		res = res[0]
	return res

def evaluate(choice, completion_kwargs: Dict, y: str, prompt: Optional[str] = None, strip_whitespace: list = ['left', 'right']) -> bool:
	completion = get_completion_logprobs(choice, completion_kwargs, prompt, strip_whitespace)
	return completion == y

def evaluate_s(response, completion_kwargs: Dict, y: str, prompt: Optional[str] = None, strip_whitespace: list = ['left', 'right']) -> Optional[Union[bool, List[bool]]]:
	if response is None:
		return None
	res = [evaluate(choice, completion_kwargs, y, prompt, strip_whitespace) for choice in response['choices']]
	if len(res) == 1:
		res = res[0]
	return res


