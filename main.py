import os

import nltk.translate.bleu_score
import sklearn.model_selection
from model import *
from plotting import *
from dataset import *
from tokenization import *
from translation import *
import get_datasets
from colorama import Fore, Style

import tomllib

import sentencepiece as spm

import click

import matplotlib.pyplot as plt


from tqdm import tqdm

import polars as pl

import sklearn
import traceback
import time
import random


def transformer_from_config(config):
    sp_source = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/source.model")
    sp_target = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/target.model")

    return Transformer(config["model"]["d_model"], 
                       config["model"]["d_ff"], 
                       config["model"]["h"], 
                        sp_source.GetPieceSize(),
                        sp_target.GetPieceSize(),
                       config["model"]["N"],
                       config["model"]["max_context"])

@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--config', default="config.toml")
@click.option('--device', default="cpu")
@click.pass_context
def cli(ctx, config, debug, device):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    with open(config, 'rb') as f:
        config = tomllib.load(f)

    print(f"{Style.DIM}Using config {config}.{Style.RESET_ALL}")

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but not available.")
        
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available.")


    ctx.obj['DEBUG'] = debug
    ctx.obj['config'] = config
    ctx.obj['device'] = torch.device(device)

    # Check for the device




    data_dir = os.path.join(os.path.dirname(__file__), config["data_root"])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = os.path.join(os.path.dirname(__file__), config["data_root"], "checkpoints")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

@cli.command()
@click.pass_context
def model_summary(ctx): 
    config = ctx.obj['config']
    transformer = transformer_from_config(config)
    from torchinfo import summary
    print(summary(transformer))


@cli.command()
@click.pass_context
@click.option('--checkpoint', default=None)
def train(ctx, checkpoint): 
    """
    Train the model.
    """
    config = ctx.obj['config']
    device = ctx.obj['device']

    sp_source = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/source.model")
    sp_target = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/target.model")

    src_vocab_size = sp_source.GetPieceSize()
    tgt_vocab_size = sp_target.GetPieceSize()

    train_dataset = PairedTextDataset(f"{config['data_root']}/source_train.txt", 
                                f"{config['data_root']}/target_train.txt",
                                sp_source, 
                                sp_target)
    
    val_dataset = PairedTextDataset(f"{config['data_root']}/source_val.txt",
                                f"{config['data_root']}/target_val.txt",
                                sp_source,
                                sp_target)

    transformer = Transformer(config["model"]["d_model"], 
                              config["model"]["d_ff"], 
                              config["model"]["h"], 
                              src_vocab_size, 
                              tgt_vocab_size, 
                              config["model"]["N"],
                              config["model"]["max_context"])
    
    # criteron = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    critereon = nn.NLLLoss(ignore_index=0, reduction='mean')
    transformer.to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config["training"]["lr"], betas=(0.9, 0.98), eps=1e-9)
    

    losses = []
    start_epoch = 0
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)
        transformer.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.Adam(transformer.parameters(), lr=config["training"]["lr"], betas=(0.9, 0.98), eps=1e-9)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["current_epoch"]
        if checkpoint.get("losses") is not None:
            losses = checkpoint["losses"]

    
    def save_model(path):
        state_dict = {
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_epoch": epoch,
            "losses": losses
        }
        torch.save(state_dict, path)

    print(f"{Style.DIM}Saving model every {config['training']['save_every']} batches.{Style.RESET_ALL}")
    
    for epoch in range(start_epoch, start_epoch + config["training"]["epochs"]):
        transformer.epoch = epoch
        print(f"{Fore.GREEN}Epoch {epoch+1}/{start_epoch + config["training"]["epochs"]}{Style.RESET_ALL}")
        count = 0

        with tqdm(train_dataset.batch(batch_size=config["training"]["batch_size"]), 
                total=train_dataset.batch_length(batch_size=config["training"]["batch_size"])) as progress:
            progress.colour = "green"
            progress.set_description("Training model")
            for src, tgt_input, tgt_output in progress:

                src = src.to(device)
                tgt_input = tgt_input.to(device)
                tgt_output = tgt_output.to(device)

                tgt_mask = (tgt_input != 0).unsqueeze(-1)
                src_mask = (src != 0).unsqueeze(-1)

                try:
                    # TRAINING CODE GOES HERE
                    output = transformer.forward(tgt_input, src)

                    loss = critereon(output.view(-1, tgt_vocab_size), 
                                    tgt_output.view(-1))
                    losses.append(loss.detach().item())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    progress.set_postfix(
                        train_loss=loss.item()
                    )

                except RuntimeError as e:
                    print(f"{Fore.RED}Error during forward pass: {e}{Style.RESET_ALL}")
                    traceback.print_exc()
                    print(f"Source shape: {src.shape}")
                    print(f"Target input shape: {tgt_input.shape}")
                    print(f"Target output shape: {tgt_output.shape}")
                    continue

                count += 1
                if count % config["training"]["save_every"] == 0:
                    save_model(f"{config['data_root']}/checkpoints/{epoch}_{count}_checkpoint.pth")
        
        state_dict = {
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_epoch": epoch + 1,
            "losses": losses
        }
        torch.save(state_dict, f"{config['data_root']}/checkpoints/{epoch + 1}_checkpoint.pth")
                    

        with tqdm(val_dataset.batch(batch_size=config["training"]["batch_size"]),
                total=val_dataset.batch_length(batch_size=config["training"]["batch_size"])) as progress:
            progress.colour = "blue"
            progress.set_description("Running validation")
            for batch in progress:
                batch

        print(f"{Fore.GREEN}Finished epoch {epoch+1}{Style.RESET_ALL}")


@cli.command()
@click.argument("model")
@click.pass_context
def validate(ctx, model):
    """
    Train the model.
    """
    config = ctx.obj['config']
    device = ctx.obj['device']

    sp_source = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/source.model")
    sp_target = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/target.model")

    src_vocab_size = sp_source.GetPieceSize()
    tgt_vocab_size = sp_target.GetPieceSize()
    
    val_dataset = PairedTextDataset(f"{config['data_root']}/source_val.txt",
                                f"{config['data_root']}/target_val.txt",
                                sp_source,
                                sp_target)

    transformer = Transformer(config["model"]["d_model"], 
                              config["model"]["d_ff"], 
                              config["model"]["h"], 
                              src_vocab_size, 
                              tgt_vocab_size, 
                              config["model"]["N"],
                              config["model"]["max_context"])
    
    transformer.to(device)
    critereon = nn.NLLLoss(ignore_index=0, reduction='mean')

    if model is not None:
        checkpoint = torch.load(model, map_location=device)
        transformer.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["current_epoch"]

    losses = []

    with tqdm(val_dataset.batch(batch_size=32),
            total=val_dataset.batch_length(batch_size=32)) as progress:
        progress.colour = "blue"
        progress.set_description("Running validation")
        count = 0
        for src, tgt_input, tgt_output in progress:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            # TRAINING CODE GOES HERE
            output = transformer.forward(tgt_input, src)

            loss = critereon(output.view(-1, tgt_vocab_size), 
                            tgt_output.view(-1))
            
            losses.append(loss.detach().item())

    losses = np.array(losses)

    plt.hist(losses)
    plt.savefig("losses.png")
    plt.show()
        



@cli.command()
@click.pass_context
def test_model(ctx): 
    config = ctx.obj['config']
    sp_source = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/source.model")
    sp_target = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/target.model")

    d_model = 512
    token_count_src = 15
    token_count_tgt = 10

    transformer = Transformer(d_model, 2048, 8, 36000, 36000, 6, 100)

    src_tokens = torch.randint(0, 36000, (1, token_count_src))
    tgt_tokens = torch.randint(0, 36000, (1, token_count_tgt))

    target = "Hello everybody, my name is Markiplier!"
    source = "అందరికీ నమస్కారం, నా పేరు మార్కిప్లియర్!"

    tgt_tokens = torch.tensor(sp_target.encode(target)).unsqueeze(0)
    src_tokens = torch.tensor(sp_source.encode(source)).unsqueeze(0)

    output = transformer.forward(tgt_tokens, src_tokens)

    attention = transformer.decoder.get_cross_attention_weights()
    print("Shape of attention weights:", attention[0][0][1].detach().numpy().shape)
    print("Shape of target tokens:", tgt_tokens.shape)
    print("Shape of source tokens:", src_tokens.shape)
    print("Shape of model output:", output.shape)
    # plot_attention("cross_attention.png", 
    #                attention[0][0][1].detach().numpy(),
    #                query_tokens=[sp_target.Decode([x]) for x in tgt_tokens],
    #                  key_tokens=[sp_source.Decode([x]) for x in src_tokens])
    # plt.show()

@cli.command()
@click.argument("input")
@click.option("--model")
@click.option("--max_tokens", default=32)
@click.pass_context
def inference(ctx, model, input, max_tokens):
    config = ctx.obj['config']
    device = ctx.obj['device']

    sp_source = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/source.model")
    sp_target = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/target.model")

    transformer = Transformer(config["model"]["d_model"],
                                config["model"]["d_ff"],
                                config["model"]["h"],
                                sp_source.GetPieceSize(),
                                sp_target.GetPieceSize(),
                                config["model"]["N"],
                                config["model"]["max_context"])

    print(f"Loading model {model}...")
    
    checkpoint = torch.load(model, map_location=device, weights_only=True)

    transformer.load_state_dict(checkpoint["model_state_dict"])
    
    translator = Translator(transformer, sp_source, sp_target, device)

    output = translator.inference(input=input, max_tokens=max_tokens)
    print(output)

@cli.command()
@click.option("--model")
@click.option("--count", default=10)
@click.option("--data", default=None)
@click.pass_context
def score(ctx, model, count, data):
    import nltk

    config = ctx.obj['config']
    device = ctx.obj['device']

    print(f"{Style.DIM}Loading model...{Style.RESET_ALL}")

    if model is None:
        raise ValueError("Checkpoint path must be specified.")
    
    if not os.path.exists(model):
        raise ValueError(f"Checkpoint file {model} does not exist.")

    sp_source = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/source.model")
    sp_target = spm.SentencePieceProcessor(model_file=f"{config['data_root']}/tokenizers/target.model")

    transformer = Transformer(config["model"]["d_model"],
                                config["model"]["d_ff"],
                                config["model"]["h"],
                                sp_source.GetPieceSize(),
                                sp_target.GetPieceSize(),
                                config["model"]["N"],
                                config["model"]["max_context"])
    
    checkpoint = torch.load(model, map_location=device, weights_only=True)

    transformer.load_state_dict(checkpoint["model_state_dict"])
    
    translator = Translator(transformer, sp_source, sp_target, device)

    if data is not None:
        src, tgt = data.split(",")
        dataset = PairedTextDataset(src, tgt, sp_source, sp_target)
    else:
        print(f"Testing {count} random samples from the validation dataset...")
        dataset = PairedTextDataset(f"{config['data_root']}/source_val.txt",
                                    f"{config['data_root']}/target_val.txt",
                                    sp_source,
                                    sp_target)

    df: pl.DataFrame = dataset.get_polars()

    sample = df.sample(count)
    scores = []

    for row in tqdm(sample.iter_rows(), total=count):
        hypothesis = translator.inference(row[1])
        reference = row[2]
        bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis) * 100
        scores.append(bleu_score)

    print(f"{Fore.GREEN}Average BLEU score: {np.mean(scores):.2f}, median score: {np.median(scores):.2f}{Style.RESET_ALL}")


@cli.command()
@click.option('--src_vocab_size', default=16000)
@click.pass_context
def tokenize(ctx, src_vocab_size=16000):
    config = ctx.obj['config']

    """Tokenize the source and target dataset using sentencepiece."""
    tokenizer_dir = os.path.join(os.path.dirname(__file__), f"{config['data_root']}/tokenizers")
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)

    spm.SentencePieceTrainer.train(
        input=os.path.join(config["data_root"], "source.txt"),
        model_prefix=os.path.join(config["data_root"], "tokenizers", "source"),
        vocab_size=config["tokenizer"]["source_vocab_size"],
        input_sentence_size=config["tokenizer"]["input_sentence_size"],
        max_sentence_length=config["tokenizer"]["max_sentence_length"],
        character_coverage=1.0,
        bos_id=config["tokenizer"]["start_id"],
        eos_id=config["tokenizer"]["end_id"],
        pad_id=config["tokenizer"]["pad_id"],
        unk_id=config["tokenizer"]["unk_id"]
    )
    spm.SentencePieceTrainer.train(
        input=os.path.join(config["data_root"], "target.txt"),
        model_prefix=os.path.join(config["data_root"], "tokenizers", "target"),
        vocab_size=config["tokenizer"]["target_vocab_size"],
        input_sentence_size=config["tokenizer"]["input_sentence_size"],
        max_sentence_length=config["tokenizer"]["max_sentence_length"],
        character_coverage=1.0,
        bos_id=config["tokenizer"]["start_id"],
        eos_id=config["tokenizer"]["end_id"],
        pad_id=config["tokenizer"]["pad_id"],
        unk_id=config["tokenizer"]["unk_id"]
    )

@cli.command()
@click.argument("dataset")
@click.pass_context
def get_dataset(ctx, dataset):
    config = ctx.obj['config']

    datasets = {
        "en_te_1": get_datasets.get_semanantar,
        "en_te_2": get_datasets.get_en_te_pairs,
        "en_es_1": get_datasets.get_en_es_1,
        "en_te_3": get_datasets.get_en_te_kaggle
    }

    if dataset is None:
        print(f"{Fore.RED}No dataset specified.{Style.RESET_ALL}")
        print("Available datasets:")
        for key in datasets.keys():
            print(key)
        return
    
    if dataset not in datasets: 
        print(f"{Fore.RED}Unknown dataset \"{dataset}\".{Style.RESET_ALL}")
        print("Available datasets:")
        for key in datasets.keys():
            print("-", key)
        return

    get_function = datasets[dataset]
    get_function(config["data_root"])

    # check line count

    source_lines = 0
    with open(f"{config['data_root']}/source.txt", 'r', encoding='utf-8') as f:
        for _ in tqdm(f, desc="Counting source lines"):
            source_lines += 1

    target_lines = 0
    with open(f"{config['data_root']}/target.txt", 'r', encoding='utf-8') as f:
        for _ in tqdm(f, desc="Counting target lines"):
            target_lines += 1

    if (target_lines != source_lines):
        print(f"{Fore.RED}Source and target datasets have different lengths.{Style.RESET_ALL}")
        print(f"Source lines: {source_lines}")
        print(f"Target lines: {target_lines}")
    else:
        print(f"{Fore.GREEN}Source and target datasets have the same length.{Style.RESET_ALL}")



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

@cli.command()
@click.pass_context
def test_positional_encoding(ctx):
    config = ctx.obj['config']
    d_model = config["model"]["d_model"]
    max_len = config["model"]["max_context"]

    pe = PositionalEncoding(d_model, max_len)
    pe_output = pe.forward(torch.zeros(1, max_len, d_model))

    plt.figure(figsize=(15, 5))
    plt.pcolormesh(pe_output[0, :, :].detach().numpy())
    plt.title("Positional Encoding")
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.show()

if __name__ == '__main__':
    cli(obj={})