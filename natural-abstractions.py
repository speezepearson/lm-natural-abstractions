import argparse
import json
from pathlib import Path
from typing import Optional
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


def main(args):
    seed: Optional[int] = args.seed
    init_content_path: Path = args.init_content

    rng = np.random.RandomState(seed)

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    roberta_model = RobertaForMaskedLM.from_pretrained('roberta-large')

    sentence: str = init_content_path.read_text()

    iter = 0
    while True:
        print(json.dumps({"iter": iter, "str": sentence}))
        sentence = step(sentence, roberta_tokenizer, roberta_model, rng)
        iter += 1

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('init_content', type=Path)

if __name__ == '__main__':
    main(parser.parse_args())
