import os
from model import *
from plotting import *
from dataset import *
from tokenization import *

import sentencepiece as spm

import click

import matplotlib.pyplot as plt

from datasets import load_dataset

from tqdm import tqdm


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    ctx.obj['DEBUG'] = debug

    data_dir = os.path.join(os.path.dirname(__file__), '.data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)



@cli.command()
@click.pass_context
def train(ctx): 
    d_model = 512
    token_count_src = 15
    token_count_tgt = 10

    transformer = Transformer(d_model, 2048, 8, 36000, 36000, 6)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


    src_tokens = torch.randint(0, 36000, (1, token_count_src))
    tgt_tokens = torch.randint(0, 36000, (1, token_count_tgt))

    transformer.forward(tgt_tokens, src_tokens)

    transformer.apply(init_weights)

    transformer.forward(tgt_tokens, src_tokens)

    attention = transformer.decoder.get_self_attention_weights()
    plot_attention("cross_attention.png", attention[0][0][1].detach().numpy())
    plt.show()

@cli.command()
@click.pass_context
def tokenize(ctx):
    tokenizer_dir = os.path.join(os.path.dirname(__file__), '.data/tokenizers')
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)

    spm.SentencePieceTrainer.train(input=".data/source.txt", model_prefix=".data/tokenizers/source", vocab_size=10000, input_sentence_size=10000)
    spm.SentencePieceTrainer.train(input=".data/target.txt", model_prefix=".data/tokenizers/target", vocab_size=10000, input_sentence_size=10000)

@cli.command()
@click.pass_context
def get_dataset(ctx):
    # Load dataset
    ds = load_dataset("ai4bharat/samanantar", "te")
    ds = ds["train"]
    ds = ds.rename_column("idx", "id")
    ds = ds.rename_column("src", "en")
    ds = ds.rename_column("tgt", "te")

    print("Writing source dataset to .data/source.txt...")
    with open(".data/source.txt", "w", encoding="utf-8") as f:
        for entry in tqdm(ds):
            f.write(entry["te"] + "\n")

    print("Writing target dataset to .data/target.txt...")
    with open(".data/target.txt", "w", encoding="utf-8") as f:
        for entry in tqdm(ds):
            f.write(entry["en"] + "\n")


if __name__ == '__main__':
    cli(obj={})