from collections import OrderedDict
from enum import Enum, IntEnum
import json
import numpy as np
import os
import pdb
import random
from typing import Any, Callable, Dict, List, Tuple, Optional
from termcolor import colored

from naclo_problems import run_naclo_test_suite
from synthetic_data import (
    get_vocab, 
    sample_multilevel_markov_chain, 
    sample_from_multilevel_markov_chain, 
    multilevel_markov_chain_sequence_to_str, 
    sample_hmm, 
    sample_from_hmm, 
    hmm_sequence_to_str, 
)

ALPHABET = list('abcdefghijklmnopqrstuvwxyz')

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    # torch.manual_seed(seed)

def successor_of_char(c: str) -> str:
    assert isinstance(c, str) and len(c) == 1 and c.isalpha()
    a = 'a' if c.lower() else 'A'
    return chr((ord(c) - ord(a) + 1) % 26 + ord(a))

def random_sequence(vocab: Optional[List[str]] = ALPHABET, n_tokens: Optional[List[str]] = list(range(3, 6))):
    return list(np.random.choice(vocab, np.random.choice(n_tokens)))

def generate_examples(
        apply_rule: Callable[[str], str], 
        n_examples: int = 100, 
        vocab: Optional[List[str]] = ALPHABET, 
        vocab_a: Optional[List[str]] = None, 
        vocab_b: Optional[List[str]] = None, 
        n_tokens: Optional[List[str]] = list(range(3, 6)), 
        n_tokens_a: Optional[List[int]] = None, 
        n_tokens_b: Optional[List[int]] = None, 
        seed: int = 0,
    ) -> List[List[str]]:
    set_seed(seed)
    vocab_a = vocab_a or vocab 
    vocab_b = vocab_b or vocab 
    n_tokens_a = n_tokens_a or n_tokens 
    n_tokens_b = n_tokens_b or n_tokens 

    examples = []
    for i in range(n_examples):
        a1 = list(np.random.choice(vocab_a, np.random.choice(n_tokens_a)))
        a2 = apply_rule(a1)
        b1 = list(np.random.choice(vocab_b, np.random.choice(n_tokens_b)))
        b2 = apply_rule(b1)
        examples.append((a1, a2, b1, b2))
    return examples

# Borple

def borple(list_of_chars: List[str]) -> List[str]:
    return list_of_chars + [successor_of_char(list_of_chars[-1])]

def gen_borple_1() -> Tuple[str, str]:
    """('Borple the sequence of letters a b c', 'a b c d')"""
    x1 = random_sequence()
    y = borple(x1)
    x1_str = ' '.join(x1)
    y_str = ' '.join(y)
    cmd = f'Borple the sequence of letters {x1_str}.'
    return (cmd, y_str)

def num_to_quantifier(n: int) -> str:
    if n == 1:
        return 'once'
    if n == 2:
        return 'twice'
    if n == 3:
        return 'thrice'
    return f'{n} times'

def apply_n_times(func: Callable, n: int, arg):
    cnt = 0
    while cnt < n:
        arg = func(arg)
        cnt += 1
    return arg

def gen_borple_2(n_times: Optional[int] = None) -> Tuple[str, str]:
    """('Borple the sequence of letters a b c thrice', 'a b c d e f')"""
    n_times = n_times or np.random.randint(1, 5)

    x = random_sequence()
    y = apply_n_times(borple, n_times, x)

    x_str = ' '.join(x)
    y_str = ' '.join(y)
    n_str = num_to_quantifier(n_times)
    cmd = f'Borple the sequence of letters {x_str} {n_str}.'
    return (cmd, y_str)

def random_bool(p: float = .5) -> bool:
    return random.random() < p

def gen_borple_3(n_times: Optional[int] = None, possible: Optional[bool] = None) -> Tuple[str, str]:
    """('How many borples are needed to transform a b c into a b c d e f?', '3')"""
    possible = possible or random_bool()

    x1 = random_sequence()
    x1_str = ' '.join(x1)

    if possible:
        n_times = n_times or np.random.randint(1, 5)
        x2 = apply_n_times(borple, n_times, x1)
        x2_str = ' '.join(x2)
        y = f'{n_times}'
    else:
        x2 = random_sequence()
        while x2[:len(x1)] == x1:
            x2 = random_sequence()
        x2_str = ' '.join(x2)
        y = f'It is impossible to borple {x1_str} into {x2_str}.'
    
    cmd = f'How many borples are needed to transform {x1_str} into {x2_str}?'
    return (cmd, y)

# Substitution 

def gen_substitute_1() -> Tuple[str, str]:
    letter1, letter2 = np.random.choice(ALPHABET, 2, replace=False)
    x = np.random.choice(ALPHABET + [letter1] * 24 + [letter2] * 14, random.randint(6, 12))
    x_str = ' '.join(x)
    cmd = f'Substitute all occurrences of the letter {letter1} with the letter {letter2} in the following sequence of letters: {x_str}'
    y_str = x_str.replace(letter1, letter2)
    return (cmd, y_str)

def gen_substitute_2(sample_swap: bool = True, prob_swap: float = .5) -> Tuple[str, str]:
    n_mappings = random.randint(1, 4)
    cmd = 'Substitute '
    mappings = []
    sub_cmds = []
    weights = np.array([1./26] * 26)

    def add_mapping(letter1, letter2):
        weights[ord(letter1) - ord('a')] += 1./2
        weights[ord(letter2) - ord('a')] += 1./20
        mappings.append((letter1, letter2))
        sub_cmds.append(f'{letter1} with {letter2}')

    while len(mappings) < n_mappings:
        letter1, letter2 = np.random.choice(ALPHABET, 2, replace=False)
        add_mapping(letter1, letter2)
        if sample_swap and random_bool(prob_swap):
            add_mapping(letter2, letter1)

    # order = np.random.permutation(range(n_mappings))
    # mappings = [mappings[i] for i in order]
    # sub_cmds = [sub_cmds[i] for i in order]
    sub_cmds = np.random.permutation(sub_cmds)

    cmd += ', '.join(sub_cmds[:-1])
    if len(sub_cmds) > 2:
        cmd += ','
    if len(sub_cmds) > 1:
        cmd += ' and '
    cmd += sub_cmds[-1]

    x = np.random.choice(ALPHABET, random.randint(6, 12), p=weights/weights.sum())
    x_str = ' '.join(x)
    cmd += f' in the sequence of letters {x_str}'

    y_str = x_str
    for letter1, letter2 in mappings:
        y_str = y_str.replace(letter1, letter2.upper())
    y_str = y_str.lower()
    return (cmd, y_str)

# Copycat

def copycat_rule_0(s: str) -> List[str]:
    return s[:-1] + [successor_of_char(s[-1])]

def copycat_rule_remove(s: str, forbidden: str = 'x') -> List[str]:
    return list(filter(lambda x: x != forbidden, s))

def test_copycat_remove(gpt3) -> Tuple[str, str]:
    n_test = 5
    raw_examples = generate_examples(copycat_rule_remove, 15, 
        vocab=ALPHABET + ['x'] * 24, 
        n_tokens=list(range(6, 12)), 
    )
    new_examples = generate_examples(lambda x: copycat_rule_remove(x, forbidden='v'), n_test, 
        vocab=ALPHABET + ['v'] * 24, 
        n_tokens=list(range(6, 12)), 
    )

    def formatter(example):
        a1, a2, b1, b2 = list(map(lambda x: ' '.join(list(x)), example))
        return (f'If {a1} maps to {a2}, what does {b1} map to?', f'{b2}')
    train_examples = list(map(formatter, raw_examples))
    test_examples = list(map(formatter, new_examples))
    
    for num_train in range(5, 20, 5):
        for x, y in train_examples[num_train:num_train + n_test]:  # use as test examples
            gpt3.few_shot(train_examples[:num_train], x=x, y=y, temperature=0)
        for x, y in test_examples:
            gpt3.few_shot(train_examples[:num_train], x=x, y=y, temperature=0)

    prefix = 'Delete all occurrences of the letter x from the following sequences of letters.'

    def formatter(example):
        a1, a2, _, _ = list(map(lambda x: ' '.join(list(x)), example))
        return (a1, a2)

    train_examples = list(map(formatter, raw_examples))
    for num_train in range(5, 20, 5):
        for x, y in train_examples[num_train:num_train + 5]:
            gpt3.few_shot(train_examples[:num_train], x=x, y=y, temperature=0, prefix=prefix, x_label='Input', y_label='Output')
        # TODO add modified prefix for letter v
        for x, y in test_examples:
            gpt3.few_shot(train_examples[:num_train], x=x, y=y, temperature=0, prefix=prefix)

# TODO hints
    # prefix = 'Delete all occurrences of the letter x from the following sequences of letters.'

    # def formatter(example):
    #     a1, a2, _, _ = list(map(lambda x: ' '.join(list(x)), example))
    #     return (a1, a2)

    # train_examples = list(map(formatter, raw_examples))
    # for num_train in range(5, 15, 5):
    #     for x, y in train_examples[num_train:num_train + 5]:
    #         gpt3.few_shot(train_examples[:num_train], x=x, y=y, temperature=0, prefix=prefix, x_label='Original', y_label='')

# Model-in-the-loop Interactions 
# Can we boost success on a task or even teach one from scratch via interaction with a few-shot learning/interactive LM?

class Result(IntEnum):
    INVALID = 0
    FAILURE = 1 # NOT_EQUALS
    ALMOST = 2
    # CONTAINS = 2
    # CLOSE = 3
    SUCCESS = 3 # EQUALS

class Hook:
    def __init__(self, condition: Callable[['Interaction'], bool], apply_func: Callable):
        self.condition = condition
        self.apply_func = apply_func

    def triggered(self, interaction: 'Interaction') -> bool:
        return self.condition(interaction)

    def __call__(self, interaction: 'Interaction'):
        self.apply_func(interaction)

class Task:
    pass 

class Interaction:
    def __init__(self, tasks: Dict[str, Callable], task_flow: Dict[str, Dict], hooks: List[Hook], model):
        self.tasks = tasks
        self.task_flow = task_flow
        self.hooks = hooks
        self.model = model

        self.interaction_sequence = []
        self.task = task_flow['root']  # type: str

    def trigger_possible_hooks(self):
        for hook in self.hooks:
            if hook.triggered(self):
                hook(self)

    def step(self):
        self.tasks[self.task](self.model, self)

def interact(self, tasks: Dict[str, Callable], task_flow: Dict[str, Dict], hooks: List[Hook]) -> List[Dict]:
    interaction = Interaction(tasks, task_flow, hooks, self)
    interaction.trigger_possible_hooks()
    while interaction.task:
        interaction.step()
        interaction.trigger_possible_hooks()
    return interaction.interaction_sequence

# Natural language program synthesis 
# gpt3.interaction(, )


# Elements of factored reasoning
## Parsing context 

# Novel classification: Mapping to interfaces

# Task suite
def run_task_suite(gpt3, cache, cache_fname):
    tasks = [
        run_data_cleaning, 
        run_sequence_prediction, 
        run_naclo_problems, 
        run_novel_instructions, 
    ]
    for task in tasks:
        set_seed()
        task(gpt3, cache, cache_fname)
        # pdb.set_trace()

def run_data_cleaning(gpt3, cache, cache_fname):
    # Data cleaning 
    prefix = 'Reformat each phone number:'
    train_examples = [
        ('4164904115', '[416] 490-4115'),
        ('528 333 2103', '[528] 333-2103'),
        ('202-318-9100', '[202] 318-9100'),
        ('(763)-200-3835', '[763] 200-3835'),
    ]
    test_examples = [
        ('3024105332', '[302] 410-5332'),
        ('610-105-3145', '[610] 105-3145'),
        ('(191)-423-9875', '[191] 423-9875'),
    ]
    for x, y in train_examples + test_examples:
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0, prefix=prefix, x_label='Original', y_label='Formatted')

def run_sequence_prediction(gpt3, cache, cache_fname):
    # What does GPT-3 infer? 
    prompts = [
        ', '.join(list('aaa')), 
        ', '.join(list('abab')), 
        ', '.join(list('aabb')), 
        ', '.join(list('aabbcc')), 
        ', '.join(list('aabbaa')), 
        ', '.join(list('a1b2c3')), 
        ', '.join(list('a1b2c3d4e5')), 
        ', '.join(list('z1y2x3')), 
        ', '.join(list('z1y2x3w4v5')), 
        ', '.join(list('z1y2x3')), 
        ', '.join(list('j1k2l3')), 
        ', '.join(list('j1k2l3m4n5')), 
        # Do inferred interleaved patterns match interleaved inferred patterns?
        ', '.join(list('258')), 
        ## Does GPT-3 model multiple possible completions 1 2 4 7 11 ... [(n)(n+1)/2+1] and 1 2 4 8 ... [2^n]?
        ', '.join(list('124')), 
        ', '.join(list('1247')), 
        ', '.join(list('1248')), 
        ', '.join(list('a2b5c8')), 
        ', '.join(list('a1b2c4')), 
        ', '.join(list('a1b2c4d7')), 
        ', '.join(list('a1b2c4d8')), 
        ', '.join(list('que')), 
        ', '.join(list('ques')), 
        ', '.join(list('q2u5e8')), 
        ', '.join(list('q1u2e4')), 
        ', '.join(list('q1u2e4s7')), 
        ', '.join(list('q1u2e4s8')), 
    ]
    for prompt in prompts:
        for i in range(5):
            gpt3.complete(prompt=prompt, temperature=0, random=i, max_tokens=50)
            gpt3.complete(prompt='Exercise 1. Continue the pattern:\n' + prompt, temperature=0, random=i, max_tokens=50)
            # gpt3.complete(prompt=prompt.replace(',',''), temperature=0, random=i, max_tokens=50)
            # gpt3.complete(prompt='Exercise 1. Continue the pattern:\n' + prompt.replace(',',''), temperature=0, random=i, max_tokens=50)


def run_naclo_problems(gpt3, cache, cache_fname):
    # NACLO problems
    run_naclo_test_suite(gpt3)

def run_synthetic_data(gpt3, cache, cache_fname):
    # Synthetic Markov Chains and HMMs
    for j in range(5):
        set_seed(j)
        mc = sample_multilevel_markov_chain(lam=2., alpha=.1)
        # print(mc)
        vocab = get_vocab(mc['start_idxs'][-1])
        # print(len(vocab))
        # print(vocab)
        seq_full = []
        while sum(len(seq) for seq in seq_full) < 512: 
            seq = sample_from_multilevel_markov_chain(mc)
            seq_full.append(seq)
        prompt_full = ' '.join([multilevel_markov_chain_sequence_to_str(mc, seq, vocab, delimiter='') for seq in seq_full])
        # entropy = compute_entropy_of_markov_chain(mc).  # TODO relate temperature and entropy
        for n_tokens in [32, 64, 128, 512]:
            for i in range(5):
                gpt3.complete(prompt=prompt_full[:n_tokens], temperature=0.7, random=i, max_tokens=1024)

    for j in range(3):
        set_seed(j)
        hmm = sample_hmm(alpha_hidden=.1, alpha_visible=.2)
        # print(hmm)
        vocab = get_vocab(hmm['start_idxs'][-1])
        seq = sample_from_hmm(hmm, 512)
        prompt_full = hmm_sequence_to_str(hmm, seq, vocab, delimiter=' ')
        for n_tokens in [32, 64, 128, 512]:
            for i in range(5):
                gpt3.complete(prompt=prompt_full[:n_tokens], temperature=0.7, random=i, max_tokens=1024)


def run_novel_instructions(gpt3, cache, cache_fname):
    # Novel instructions 1: to borple a sequence of letters
    # Examples:
    # [
    #     ('Borple the sequence of letters a b c', 'a b c d')
    #     ('Borple the sequence of letters a b c thrice', 'a b c d e f')
    #     ('How many borples are needed to transform a b c into a b c d e f?', '3')
    # ]
    set_seed()
    n_train = 3
    n_test = 3
    train_examples = [gen_borple_1() for i in range(n_train)]           \
                + [gen_borple_2() for i in range(n_train)]              \
                + [gen_borple_3(possible=True) for i in range(n_train)] \
                + [gen_borple_3(possible=False) for i in range(n_train)]
    test_examples = [gen_borple_1() for i in range(n_test)]           \
                + [gen_borple_2() for i in range(n_test)]              \
                + [gen_borple_3(possible=True) for i in range(n_test)] \
                + [gen_borple_3(possible=False) for i in range(n_test)]
    prefix = 'The following exercises demonstrate what it means to borple a sequence of letters. In words, borpling a sequence of letters modifies the sequence by adding the letter that comes after the last letter of the sequence in the alphabet to the end of the sequence.'
    for x, y in test_examples:
        # np.random.permutation(train_examples)
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0, prefix=prefix, x_label='Exercise', y_label='Answer')

    # Novel instructions 2: letter substitution 
    ## part i
    # pdb.set_trace()
    set_seed()
    prefix = 'The substitute command replaces all occurrences of a given letter with the specified letter. Complete the following exercises by applying the described substitutions.'
    train_examples = [gen_substitute_1() for i in range(n_train)]
    test_examples = [gen_substitute_1() for i in range(n_test)]
    for x, y in test_examples:
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0, prefix=prefix, x_label='Exercise', y_label='Answer')
    
    ## part ii
    # pdb.set_trace()
    set_seed()
    prefix = 'The substitute command replaces all occurrences of a given letter with the specified letter. Complete the following exercises by applying the described substitutions.'
    train_examples = [gen_substitute_2() for i in range(n_train)]
    test_examples = [gen_substitute_2() for i in range(n_test)]
    for x, y in test_examples:
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0, prefix=prefix, x_label='Exercise', y_label='Answer')

    # Novel instructions 3: return of the copycat removal task - TODO make into interaction 
    # pdb.set_trace()
    set_seed()
    test_copycat_remove(gpt3)

if __name__ == '__main__':
    set_seed()
    n_train = 3
    train_examples = [gen_borple_1() for i in range(n_train)]           \
                + [gen_borple_2() for i in range(n_train)]              \
                + [gen_borple_3(possible=True) for i in range(n_train)] \
                + [gen_borple_3(possible=False) for i in range(n_train)]
    print(train_examples)
    set_seed()
    print(gen_substitute_2())
    set_seed()
    print('\n'.join([' --> '.join(gen_substitute_1()) for i in range(10)]))
    set_seed()
    print('\n'.join([' --> '.join(gen_substitute_2()) for i in range(10)]))
    # import pdb; pdb.set_trace()

