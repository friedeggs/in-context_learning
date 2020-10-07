import sys

from process import (
    GPT3,
    MockGPT3,
    read_cache,
)

background_knowledge = 'Distinctions among Near-Synonyms\n\nIn the context of near-synonymy, the process of lexical choice becomes profoundly more complicated because of the subtle nuances among near-synonyms. The denotational meaning of two synonyms may differ as one of them is more intentional/accidental, continuous/intermittent, immediate/iterative, sudden/gradual, terminative/non-terminative, emotional/non-emotional, or has different degrees. The connotative meaning of two synonyms may differ as one of them is more formal/informal, abstract/concrete, pejorative/favorable, forceful/weak, or has different emphasis.\n\n'

test_examples = [
    {
        's': "She arrived with a stack of glistening stopboxes containing sushi, sashimi, oysters in their shells, and Terran vegetables fresh plucked from their hydroponic beds.",
        'w': "vegetables",
        'wprimes': ['veggies', 'produce', 'greens', 'plants', 'leaves', 'herbs', 'salads', 'legumes', 'edibles', 'edible plants', 'herbaceous plants'],

        # Good: veggies
        # Acceptable: produce, produces, greens
        # Hyponym: plants, leaves, herbs, salads, legumes, edibles
        # Awkward: edible plants
        # Bad: herbaceous plants
    }
]


def background_knowledge_test(gpt3, prefix=''):
    gpt3.complete(
        prompt=f'{prefix}Q. What is near synonymy?\nA.',
        max_tokens=300,
        top_p=0.1,
        n=3,
    )

    gpt3.complete(
        prompt=f'''{prefix}She stared at him through the window.
She glimpsed him through the window.

Q. What is the difference between "stared at" and "glimpsed" in the sentences?
A.''',
        top_p=0.1,
        n=3,
    )

    gpt3.complete(
        prompt=f'''{prefix}She stared at him through the window.
She glimpsed him through the window.

In these sentences, "stared at" and "glimpsed" have the same meaning because''',
        top_p=0.1,
        n=3,
    )

    gpt3.complete(
        prompt=f'''{prefix}She stared at him through the window.
She glimpsed him through the window.

In these sentences, "stared at" and "glimpsed" have different meanings because''',
        top_p=0.1,
        n=3,
    )


def generation(gpt3, prefix=''):
    for example in test_examples:
        s = example['s']
        w = example['w']
        prompt = f'{prefix}{s}\n\nQ. What are good synonyms for "{w}" in this sentence?\nA.'

        gpt3.complete(
            prompt=prompt,
            logprobs=50,
            temperature=0,
            n=1,
        )

        gpt3.complete(
            prompt=prompt,
            logprobs=50,
            top_p=0.1,
            n=3,
        )

        gpt3.complete(
            prompt=prompt,
            logprobs=50,
            temperature=0.5,
            n=3,
        )


def binary_classification(gpt3, prefix=''):
    for example in test_examples:
        s = example['s']
        w = example['w']
        for wprime in example['wprimes']:
            prompt = f'{prefix}{s}\n\nQ. Is "{wprime}" a good synonym for "{w}" in this sentence?\nA.'
            gpt3.complete(
                prompt=prompt,
                max_tokens=10,
                logprobs=10,
                temperature=0,
            )


def question_answering(gpt3, prefix=''):
    for example in test_examples:
        s = example['s']
        w = example['w']
        for wprime in example['wprimes']:
            prompt = f'{prefix}{s}\n\nQ. Is "{wprime}" a good synonym for "{w}" in this sentence? Why or why not?\nA.'
            gpt3.complete(
                prompt=prompt,
            )


def run_tasks(gpt3, prefix=''):
    tasks = [
        background_knowledge_test,
        generation,
        binary_classification,
        question_answering,
    ]

    # Zero-shot without background knowledge in prompt
    for task in tasks:
        task(gpt3)

    # Zero-shot with background knowledge in prompt
    for task in tasks:
        task(gpt3, prefix=background_knowledge)


def main():
    GPT = GPT3 if 'openai' in sys.modules else MockGPT3
    cache_fname = f'cache_mina_{GPT.__name__}.jsonl'
    cache = read_cache(cache_fname)
    gpt3 = GPT(cache)

    run_tasks(gpt3)


if __name__ == '__main__':
    main()
