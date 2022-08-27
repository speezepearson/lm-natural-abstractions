import argparse
import json
from pathlib import Path
from typing import Iterator, Optional, Tuple
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

def step(sentence: str, tokenizer: RobertaTokenizer, model: RobertaForMaskedLM, rng: np.random.RandomState):
    inputs = tokenizer(sentence, return_tensors='pt')
    input_toks = inputs['input_ids']
    assert input_toks.shape[0] == 1
    replace_tok = rng.randint(1, input_toks.size()[1] - 1)
    input_toks[:, replace_tok] = tokenizer.mask_token_id

    outputs = model(**inputs)
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == input_toks.size()[1]
    probs: torch.Tensor = outputs.logits[0, replace_tok].flatten().softmax(0)

    sample = rng.choice(len(probs), p=probs.detach().numpy())
    while sample in [tokenizer.mask_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
        sample = rng.choice(len(probs), p=probs.detach().numpy())
    new_toks = input_toks.flatten()[:]
    new_toks[replace_tok] = sample
    new_str = tokenizer.decode(new_toks.tolist()[1:-1])
    return new_str

def iter_seeds(start_seed: int) -> Iterator[int]:
    seed = start_seed
    while True:
        yield seed
        seed = np.random.RandomState(seed).randint(0, 2**32-1)

def main(args):
    n_iters: int = args.n_iters
    seeds: Iterator[int] = iter_seeds(args.seed)
    init_content: str = args.init_content.read_text()

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    roberta_model = RobertaForMaskedLM.from_pretrained('roberta-large')

    sentence = init_content
    rng = np.random.RandomState()

    for seed, _ in zip(seeds, range(n_iters)):
        print(json.dumps({'sentence': sentence, 'next_seed': seed}))
        rng.seed(seed)
        sentence = step(sentence, roberta_tokenizer, roberta_model, rng)

parser = argparse.ArgumentParser()
parser.add_argument('--n-iters', type=int, default=1000)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('init_content', type=Path)

if __name__ == '__main__':
    main(parser.parse_args())
