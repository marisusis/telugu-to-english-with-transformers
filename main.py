import os

import sklearn.model_selection
from model import *
from plotting import *
from dataset import *
from tokenization import *

import tomllib

import sentencepiece as spm

import click

import matplotlib.pyplot as plt

from datasets import load_dataset

from tqdm import tqdm

import polars as pl

import sklearn


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)

    print(f"Using config {config}.")


    ctx.obj['DEBUG'] = debug
    ctx.obj['config'] = config

    data_dir = os.path.join(os.path.dirname(__file__), '.data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

@cli.command()
@click.pass_context
def test_data(ctx): 
    config = ctx.obj['config']


@cli.command()
@click.pass_context
def train(ctx): 
    """
    Train the model.
    """
    config = ctx.obj['config']

    sp_source = spm.SentencePieceProcessor(model_file='.data/tokenizers/source.model')
    sp_target = spm.SentencePieceProcessor(model_file='.data/tokenizers/target.model')

    src_vocab_size = sp_source.GetPieceSize()
    tgt_vocab_size = sp_target.GetPieceSize()

    dataset = PairedTextDataset(f"{config['data_root']}/source_train.txt", 
                                f"{config['data_root']}/target_train.txt")
    


    transformer = Transformer(config["model"]["d_model"], config["model"]["d_ff"], config["model"]["h"], src_vocab_size, tgt_vocab_size, config["model"]["N"])


@cli.command()
@click.pass_context
def test_model(ctx): 
    sp_source = spm.SentencePieceProcessor(model_file='.data/tokenizers/source.model')
    sp_target = spm.SentencePieceProcessor(model_file='.data/tokenizers/target.model')

    d_model = 512
    token_count_src = 15
    token_count_tgt = 10

    transformer = Transformer(d_model, 2048, 8, 36000, 36000, 6)

    src_tokens = torch.randint(0, 36000, (1, token_count_src))
    tgt_tokens = torch.randint(0, 36000, (1, token_count_tgt))

    target = "Hello everybody, my name is Markiplier!"
    source = "అందరికీ నమస్కారం, నా పేరు మార్కిప్లియర్!"

    tgt_tokens = sp_target.encode(target)
    src_tokens = sp_source.encode(source)

    transformer.forward(torch.tensor(tgt_tokens).unsqueeze(0), 
                        torch.tensor(src_tokens).unsqueeze(0))

    attention = transformer.decoder.get_cross_attention_weights()
    plot_attention("cross_attention.png", 
                   attention[0][0][1].detach().numpy(),
                   query_tokens=[sp_target.Decode([x]) for x in tgt_tokens],
                     key_tokens=[sp_source.Decode([x]) for x in src_tokens])
    plt.show()

@cli.command()
@click.option('--src_vocab_size', default=16000)
@click.pass_context
def tokenize(ctx, src_vocab_size=16000):
    config = ctx.obj['config']

    """Tokenize the source and target dataset using sentencepiece."""
    tokenizer_dir = os.path.join(os.path.dirname(__file__), '.data/tokenizers')
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)

    spm.SentencePieceTrainer.train(
        input=os.path.join(config["data_root"], "source.txt"),
        model_prefix=os.path.join(config["data_root"], "tokenizers", "source"),
        vocab_size=config["tokenizer"]["source_vocab_size"],
        input_sentence_size=config["tokenizer"]["input_sentence_size"],
        max_sentence_length=config["tokenizer"]["max_sentence_length"],
        character_coverage=1.0
    )
    spm.SentencePieceTrainer.train(
        input=os.path.join(config["data_root"], "target.txt"),
        model_prefix=os.path.join(config["data_root"], "tokenizers", "target"),
        vocab_size=config["tokenizer"]["target_vocab_size"],
        input_sentence_size=config["tokenizer"]["input_sentence_size"],
        max_sentence_length=config["tokenizer"]["max_sentence_length"],
        character_coverage=1.0
    )

@cli.command()
@click.pass_context
def get_dataset(ctx):
    config = ctx.obj['config']

    # Load dataset
    ds = load_dataset("ai4bharat/samanantar", "te")
    ds = ds["train"]
    ds = ds.rename_column("idx", "id")
    ds = ds.rename_column("src", "en")
    ds = ds.rename_column("tgt", "te")
    

    path = f"{config['data_root']}/source.txt"
    print(f"Writing source dataset to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds):
            f.write(entry["te"] + "\n")

    print(f"Writing target dataset to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds):
            f.write(entry["en"] + "\n")

@cli.command()
@click.pass_context
def prepare_dataset(ctx):
    config = ctx.obj['config']
    print("Loading data from disk...")
    # with open(f"{config['data_root']}/source.txt", 'r', encoding='utf-8') as f:
    #     lines = []
    #     for line in tqdm(f, desc="Reading source lines"):
    #         lines.append(line.strip())
            
    # source_ds = pl.DataFrame({'text': lines})

    source_ds = pl.read_csv(f"{config['data_root']}/source.txt", 
                            has_header=False, 
                            separator="\n",
                            schema_overrides={"text": pl.Utf8},
                            quote_char=None)
    source_ds = source_ds.with_row_index()
    print(f"Source shape: {source_ds.shape}")

    target_ds = pl.read_csv(f"{config['data_root']}/target.txt",
                            has_header=False,
                            separator="\n",
                            schema_overrides={"text": pl.Utf8},
                            quote_char=None)
    target_ds = target_ds.with_row_index()
    print(f"Target shape: {target_ds.shape}")

    print("Checking for null values...")
    source_null_indices = source_ds.filter(pl.col('text').is_null()).get_column('index').to_numpy()
    target_null_indices = target_ds.filter(pl.col('text').is_null()).get_column('index').to_numpy()

    print(f"Source null indices: {source_null_indices}")
    print(f"Target null indices: {target_null_indices}")

    # Remove rows with nulls in either source or target
    null_indices = set(source_null_indices).union(set(target_null_indices))
    source_ds = source_ds.filter(~pl.col("index").is_in(list(null_indices)))
    target_ds = target_ds.filter(~pl.col("index").is_in(list(null_indices)))

    print(f"Source shape after filtering: {source_ds.shape}")
    print(f"Target shape after filtering: {target_ds.shape}")

    if source_ds.shape[0] != target_ds.shape[0]:
        raise ValueError("Source and target datasets have different lengths")

    print("Data loaded.")

    print("Splitting source into test and validation sets...")
    source_train, source_val, target_train, target_val = sklearn.model_selection.train_test_split(source_ds, target_ds, test_size=config["data"]["test_split"])

    pairs = [
        (source_train, f"{config['data_root']}/source_train.txt"),
        (source_val, f"{config['data_root']}/source_val.txt"),
        (target_train, f"{config['data_root']}/target_train.txt"),
        (target_val, f"{config['data_root']}/target_val.txt")
    ]

    for ds, path in pairs:
        print(f"Writing {path}...")
        with open(path, "w", encoding="utf-8") as f:
            for entry in tqdm(ds["text"], unit=" lines"):
                f.write(entry + "\n")

if __name__ == '__main__':
    cli(obj={})