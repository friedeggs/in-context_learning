import sys, os
sys.path.append('.')

import logging; log = logging.getLogger(__name__)

from atlas.gpt import GPT3, get_cache
from atlas.util import count_tokens, get_tokenization, show_tokenization

def main(argv):
	mock = 'submit' not in argv
	cache = get_cache()
	gpt = GPT3(cache, mock)

	log.info(get_tokenization('The cat slumbers.'))
	log.info(show_tokenization('The cat slumbers.'))
	log.info(count_tokens('The cat slumbers.'))

	examples = [
		('a b c', 'c b a'),
		('t h e', 'e h t'),
		('h e l l o', 'o l l e h'),
		('c a p i t a l', 'l a t i p a c'),
	]
	## Setting 'echo' to True will return the prompt along with the completion.
	## This enables getting the logprobs for arbitrary strings.
	# completion_kwargs = {
	# 	'staged': True,
	# 	'temperature': 0, 
	# 	'engine': 'ada', 
	# 	'max_tokens': 0, 
	# 	'stop': '\n',
	# 	'logprobs': 100,
	# 	'echo': True,
	# }
	completion_kwargs = {
		'staged': True,
		'temperature': 0, 
		'engine': 'ada', 
		'max_tokens': 20, 
		'stop': '\n',
		'logprobs': 100,
		'echo': False,
	}
	## Simple version - no staging
	# gpt.few_shot(examples, x='h o r s e', y='e s r o h', prefix='Reverse the input.', temperature=1)
	gpt.few_shot(examples, x='h o r s e', y='e s r o h', prefix='Reverse the input.', **completion_kwargs)

	cost = gpt.calculate_cost()
	if cost:
		log.info('This request will cost %d tokens (including completion)' % cost)
		# k = 'y'
		k = None
		if k == 'y':
			log.warn('Submitting queries without confirmation!')
		gpt.run_staged_queries(k)

if __name__ == '__main__':
	main(sys.argv)