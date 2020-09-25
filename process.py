from collections import OrderedDict
import json
import numpy as np
import os
import random
import sys
from typing import Any, Callable, Dict, List, Tuple, Optional
from termcolor import colored

from naclo_problems import run_naclo_test_suite
from tasks import (
    gen_borple_1, gen_borple_2, gen_borple_3, 
    test_copycat_remove, 
    gen_substitute_1, gen_substitute_2, 
    run_task_suite, 
)
from synthetic_data import (
    get_vocab, 
    sample_multilevel_markov_chain, 
    sample_from_multilevel_markov_chain, 
    multilevel_markov_chain_sequence_to_str, 
    sample_hmm, 
    sample_from_hmm, 
    hmm_sequence_to_str, 
)

try:
    import openai  # type: ignore

    # https://beta.openai.com/api-ref
    openai.api_key = open('api-key').read().strip()
except Exception as e:
    print(e)
    print('Falling back to MockGPT3')

DEFAULT_CACHE_PATH = 'cache.jsonl'

HEADER_COLOR = 'green'
RESPONSE_COLOR = 'red'

def make_header(s: Any):
    return colored(f'===== {s}', 'green')

def get_key(request):
    return tuple(sorted(request.items()))

def read_cache(filename: str = DEFAULT_CACHE_PATH):
    cache = OrderedDict()
    if os.path.exists(filename):
        for line in open(filename):
             item = json.loads(line)
             cache[get_key(item['request'])] = item['response']
    #print(f"Read {len(cache)} cache entries")
    cache['__filename__'] = filename
    return cache

def write_cache(cache: Dict, filename: Optional[str] = None):
    filename = cache.get(filename) or filename or DEFAULT_CACHE_PATH
    with open(filename, 'w') as f:
        for key, value in cache.items():
            item = {
                'request': dict(key),
                'response': value,
            }
            print(json.dumps(item), file=f)
    #print(f"Wrote {len(cache)} cache entries")

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    # torch.manual_seed(seed)

class GPT3:
    def __init__(self, cache: Dict):
        self.cache = cache

    def make_query(self, **kwargs) -> Dict:
        key = get_key(kwargs)
        if key in self.cache:
            response = self.cache[key]
        else:
            kwargs = dict(kwargs)
            if 'random' in kwargs:
                del kwargs['random']
            response = openai.Completion.create(engine="davinci", **kwargs)
            self.cache[key] = response
            write_cache(self.cache)
        return response

    def complete(self, **kwargs):
        response = self.make_query(**kwargs) 
        prompt = kwargs['prompt']
        del kwargs['prompt']
        print(make_header(kwargs))
        print(prompt, end='')
        for choice in response['choices']:
            print(colored(choice['text'], RESPONSE_COLOR))
        print('')

    def few_shot(self, examples: List[Tuple[str, str]], x: str, y: Optional[str] = None, prefix: Optional[str] = None, x_label: str = 'Input', y_label: str = 'Output', **kwargs):
        prompt = f'{x_label}: {x}\n{y_label}:'
        if len(examples) > 0:
            prompt = '\n'.join([f'{x_label}: {x}\n{y_label}: {y}\n' for x, y in examples]) + '\n' + prompt
        if prefix is not None:
            prompt = prefix + '\n' + prompt
        kwargs['prompt'] = prompt
        kwargs['stop'] = '\n'
        response = self.make_query(**kwargs)
        #prompt = kwargs['prompt']
        #del kwargs['prompt']
        #print(make_header(kwargs))
        for choice in response['choices']:
            predicted_y = choice['text'].lstrip().rstrip()
            if y is not None:  # Correct answer given
                if y == predicted_y:
                    rel = colored('EQUALS', 'green')
                elif y in predicted_y:
                    rel = colored('CONTAINS', 'yellow')
                else:
                    rel = 'NOT EQUALS'
                extra = f' {rel} {y}'
            else:
                extra = ''
            print(f'[{len(examples)} examples] {x} -> {colored(predicted_y, RESPONSE_COLOR)}{extra}')

class MockGPT3:
    def __init__(self, cache: Dict):
        self.cache = cache

    def make_query(self, **kwargs) -> Dict:
        key = get_key(kwargs)
        if key in self.cache:
            response = self.cache[key]
        else:
            kwargs = dict(kwargs)
            if 'random' in kwargs:
                del kwargs['random']
            response = {
                'choices': []  #
            }
            self.cache[key] = response
        return response

    def complete(self, **kwargs):
        response = self.make_query(**kwargs) 
        prompt = kwargs['prompt']
        del kwargs['prompt']
        print(make_header(kwargs))
        print(prompt, end='')
        for choice in response['choices']:
            print(colored(choice['text'], RESPONSE_COLOR))
        print('')

    def few_shot(self, examples: List[Tuple[str, str]], x: str, y: Optional[str] = None, prefix: Optional[str] = None, x_label: str = 'Input', y_label: str = 'Output', **kwargs):
        prompt = f'{x_label}: {x}\n{y_label}:'
        if len(examples) > 0:
            prompt = '\n'.join([f'{x_label}: {x}\n{y_label}: {y}\n' for x, y in examples]) + '\n' + prompt
        if prefix is not None:
            prompt = prefix + '\n' + prompt
        if y is not None:
            prompt += ' ' + y
        kwargs['prompt'] = prompt
        kwargs['stop'] = '\n'
        response = self.make_query(**kwargs)
        print(prompt)

def main():
    if 'openai' in sys.modules:
        cache = read_cache()
        gpt3 = GPT3(cache)
    else:
        cache_fname = 'cache_mockgpt3.jsonl'
        cache = read_cache(cache_fname)
        gpt3 = MockGPT3(cache)

    # Generate free-form stuff
    for i in range(10):
        gpt3.complete(prompt='Hello!', max_tokens=100, random=i)

    # Biographies just made up
    for i in range(10):
        gpt3.complete(prompt='John Duchi is an assistant professor in statistics and electrical engineering at Stanford University. He', temperature=0.5, random=i, max_tokens=100)
    for i in range(5):
        gpt3.complete(prompt='[PERSON] John Duchi is an assistant professor in statistics and electrical engineering at Stanford University.\n[PERSON] Tommi Jaakkola is a professor in computer science at MIT.\n[PERSON] Clyde Drexler is', temperature=1, random=i, max_tokens=100, stop='\n')

    # Story involving two novel characters
    for i in range(5):
        gpt3.complete(prompt='One day, Elon Musk invited George Washington over for dinner.  Elon showed', temperature=1, random=i, max_tokens=100)
    for i in range(5):
        gpt3.complete(prompt='One day, Ghandi and Richard Stallman were sitting next to each other on the bus.  Ghandi looked over to see what Richard was working on and was', temperature=1, random=i, max_tokens=100)

    # Question answering
    gpt3.few_shot([], 'What is the tallest mountain?')
    for i in range(5):
        gpt3.few_shot([], 'Who was the first president of the United States?', random=i)
    for i in range(5):
        gpt3.few_shot([('What is the tallest mountain?', 'Mount Everest')], 'Who was the first president of the United States?', random=i)
    gpt3.few_shot([('What is the tallest mountain?', 'Mount Everest')], 'Who was the first president of the United States?', x_label='Q', y_label='A')

    examples = [
        ('What is human life expectancy in the United States?', 'Human life expectancy in the United States is 78 years.'),
        ('Who was president of the United States in 1955?', 'Dwight D. Eisenhower was president of the United States in 1955.'),
        #('What party did he belong to?', 'He belonged to the Republican Party.'),
        #('Who was president of the United States before George W. Bush?', 'Bill Clinton was president of the United States before George W. Bush.'),
        #('Who won the World Series in 1995?', 'The Atlanta Braves won the World Series in 1995.')
    ]
    gpt3.few_shot(examples, 'How many children does Barack Obama have?', x_label='Q', y_label='A')
    gpt3.few_shot(examples, 'Who was president before Bill Clinton?', x_label='Q', y_label='A')
    gpt3.few_shot(examples, 'Who was the first person to climb Mt. Everest?', x_label='Q', y_label='A')
    gpt3.few_shot(examples, 'What is the capital of Turkey?', x_label='Q', y_label='A')
    gpt3.few_shot(examples, 'How many states border Texas?', x_label='Q', y_label='A')  # Wrong
    gpt3.few_shot(examples, 'How long did Mozart live?', x_label='Q', y_label='A')
    gpt3.few_shot(examples, 'How old is Mozart?', x_label='Q', y_label='A')
    gpt3.few_shot(examples, 'What languages did Mozart speak?', x_label='Q', y_label='A')
    gpt3.few_shot(examples, 'What are good restaurants in Palo Alto?', x_label='Q', y_label='A')  # Wrong

    people = [
        'George Washington',
        'Albert Einstein',
        'Geoff Hinton',
        'Andrej Karpathy',  # Wrong
    ]
    for person in people:
        gpt3.few_shot(examples, f'When was {person} born?', x_label='Input', y_label='Output')

    examples = [
        ('Who was president of the United States in 1955?', 'Dwight D. Eisenhower'),
        ('What year was Microsoft founded?', '1975'),
        #('What is human life expectancy in the United States?', '78 years'),
        #('What party did he belong to?', 'He belonged to the Republican Party.'),
        #('Who was president of the United States before George W. Bush?', 'Bill Clinton was president of the United States before George W. Bush.'),
        #('Who won the World Series in 1995?', 'The Atlanta Braves won the World Series in 1995.')
    ]
    countries = ['United States', 'Canada', 'Mexico', 'Russia', 'China', 'Spain', 'Greece', 'Belgium', 'Japan', 'North Korea', 'Mongolia', 'Kenya', 'Ghana']
    for country in countries:
        gpt3.few_shot(examples, f'What is the capital of {country}?', x_label='Input', y_label='Output')

    gpt3.few_shot(examples, f'When was OpenAI founded?', x_label='Input', y_label='Output')
    gpt3.few_shot(examples, f'What does Microsoft do?', x_label='Input', y_label='Output')
    gpt3.few_shot(examples, f'What is Picasso known for?', x_label='Input', y_label='Output')
    gpt3.few_shot(examples, f'What did Stravinsky compose?', x_label='Input', y_label='Output')
    gpt3.few_shot(examples, f'Where is the tallest building in the world?', x_label='Input', y_label='Output')
    gpt3.few_shot(examples, f'Where is the tallest building in the world how how high is it?', x_label='Input', y_label='Output')  # Makes some stuff up

    gpt3.few_shot([], f'Where is the tallest building in the world?', x_label='Input', y_label='Output')  # Doesn't know how to continue

    # Math: messes up a bit
    examples = []
    for a in range(4):
        for b in range(4):
            examples.append((f'What is {a} + {b}?', f'{a + b}'))
    num_train = 5
    for x, y in examples[num_train:]:
        gpt3.few_shot(examples[:num_train], x=x, y=y)

    # Summarize

    # Natural language to bash
    examples = [
        ('list all files', 'ls'),
        ('make a directory called foo', 'mkdir foo'),
        ('print contents of report.txt', 'cat report.txt'),
    ]
    gpt3.few_shot(examples, 'delete file called yummy.pdf', temperature=1)
    gpt3.few_shot(examples, 'print the first 15 lines of a.txt', temperature=1)
    gpt3.few_shot(examples, 'print the last 15 lines of a.txt', temperature=1)
    gpt3.few_shot(examples, 'check if a.txt exists', temperature=1)  # fail
    gpt3.few_shot(examples, 'get the number of lines in a.txt', temperature=1) # not perfect
    gpt3.few_shot(examples, 'print the lines in a.txt but in reverse', temperature=1) # not quite what I had intended
    gpt3.few_shot(examples, 'search for "foo" in a.txt', temperature=1)
    gpt3.few_shot(examples, 'get the current directory', temperature=1)
    gpt3.few_shot(examples, 'print the largest file in the current directory', temperature=1)
    gpt3.few_shot(examples, 'print the size of file "massive.txt"', temperature=1)
    gpt3.few_shot(examples, 'remove the first 3 lines of a.txt', temperature=1, random=1)  # Wrong
    gpt3.few_shot(examples, 'print the 3rd line of a.txt', temperature=1, random=1)  # Wrong

    # Reverse
    examples = [
        ('a b c', 'c b a'),
        ('t h e', 'e h t'),
        ('h e l l o', 'o l l e h'),
        ('c a p i t a l', 'l a t i p a c'),
    ]
    gpt3.few_shot(examples, x='h o r s e', y='e s r o h', prefix='Reverse the input.', temperature=1, random=1)  # fail

    # Translation
    examples = [
        ('the house', 'la maison'),
        ('I am a cat.', 'je suis un chat.'),
    ]
    gpt3.few_shot(examples, 'I like to drink water.')

    # Systematicity (capitals)

    # Sensitivity to prompt?

    # Logical forms
    train_examples = [
        ('what\'s the capital of Maine?', 'capitalOf(Maine)'),
        ('how many states border Texas?', 'count(and(states, border(Texas)))'),
        ('what is the largest state?', 'argmax(states, population)'),
    ]
    test_examples = [
        ('how many states border Illinois?', 'count(and(states, border(Illinois)))'),
        ('how many states border the largest state?', 'count(and(states, border(argmax(states, population))))'),
        ('how many states are adjacent to Illinois?', 'count(and(states, border(Illinois)))'),
    ]
    for x, y in test_examples:
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0)

    # Break robot actions into steps
    train_examples = [
        ('get the apples', 'go to apples; pick up apples'),
        ('wash the apples', 'go to apples; pick up apples; go to sink; turn on faucet; turn off faucet'),
        ('put the cereal on the shelf', 'go to cereal; pick up cereal; go to shelf; drop cereal'),
    ]
    test_examples = [
        ('wash the oranges', None),
        ('wash the bowl', None),
        ('wash the windows', None),
        ('put the milk in the fridge', None),
        ('put the watermelon on the counter', None),
        ('cut the apples', None),
        ('cut the oranges', None),
        ('peel the apple', None),
        ('boil an egg', None),
        ('put away the groceries', None),
        ('clean the living room', None),
        ('set the table', None),
    ]
    for x, y in train_examples + test_examples:
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0)

    # Thesaurus

    # Things beyond few-shot learning (soft influence)

    # Begin section (frieda) =============================================================================

    run_task_suite(gpt3, cache, cache_fname)

main()
