import sys 

from process import (
    GPT3, MockGPT3,
    read_cache, 
)

def run_chris_tasks(gpt3):
    tasks = [
        task_spelling,
        task_pronunciation, 
        task_rhyming
    ]
    for task in tasks:
        task(gpt3)

def task_spelling(gpt3):
    prefix = """Take a word and produce the correct spelling of that word.

Examples:
"""
    train_examples = [
        ('percy', 'P-E-R-C-Y'), 
        ('fairness', 'F-A-I-R-N-E-S-S'), 
        ('active', 'A-C-T-I-V-E'), 
        ('probe', 'P-R-O-B-E'), 
    ]
    ans = lambda s: '-'.join(list(s.upper()))
    xs = [
        'program', 
        'garpeldorth',
        'longstwhile', 
        'ruzzurtle', 

    ]
    ys = list(map(ans, xs))
    test_examples = list(zip(xs, ys))
    for x, y in test_examples:
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0, prefix=prefix, x_label='Word', y_label='Spelling', max_tokens=150)

def task_pronunciation(gpt3):
    prefix = """Take a word and produce its pronunciation in the International Phonetic Alphabet.

Examples:
"""
    train_examples = [
        ('percy', 'pˈɜː-si'), 
        ('fairness', 'ˈfɛər-nɪs'), 
        ('active', 'ˈæk-tɪv'), 
        ('probe', 'proʊb'), 
    ]
    test_examples = [
        ('program', 'ˈproʊ-græm'),  
        ('garpeldorth', 'ɡˈɑː-pʊl-dˌɔːθ'), 
        ('longstwhile', 'lˈɒŋst-waɪl'),  
        ('ruzzurtle', 'ɹˈʌ-zˈɜː-təl'),  
    ]
    for x, y in test_examples:
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0, prefix=prefix, x_label='Word', y_label='Pronunciation', max_tokens=150)

def task_rhyming(gpt3): 
    prefix = """Given a word, produce a list of words that rhyme with it.

Examples:
"""
    train_examples = [
        ('percy', 'mercy, pursy, circe, controversy'), 
        ('fairness', 'bareness, rareness, squareness, awareness, unfairness'), 
        ('active', 'abstractive, attractive, inactive, proactive, reactive, refractive, subtractive'), 
        ('probe', 'globe, lobe, robe, strobe, disrobe, microbe, wardrobe'), 
    ]
    test_examples = [
        ('program', None), 
        ('garpeldorth', None), 
        ('longstwhile', None), 
        ('ruzzurtle', None), 
    ]
    for x, y in test_examples:
        gpt3.few_shot(train_examples, x=x, y=y, temperature=0, prefix=prefix, x_label='Word', y_label='Pronunciation', max_tokens=150)

    # Answers

    # Program: bam, cam, clam, cram, dam, damn, gram, ham, jam, lam, ...
    # Garpeldorth: north, forth, henceforth
    # Longstwhile: aisle, bile, file, guile, isle, mile, ...
    # Ruzzurtle: turtle, fertile, hurtle, kirtle, myrtle, infertile

def main():
    GPT = GPT3 if 'openai' in sys.modules else MockGPT3
    cache_fname = f'cache_chris_{GPT.__name__}.jsonl'
    cache = read_cache(cache_fname)
    gpt3 = GPT(cache)
    run_chris_tasks(gpt3)

if __name__ == '__main__':
    main()
