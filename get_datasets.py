from datasets import load_dataset
from tqdm import tqdm

def get_semanantar(dir):

    # Load dataset
    ds = load_dataset("ai4bharat/samanantar", "te")
    ds = ds["train"]
    ds = ds.rename_column("idx", "id")
    ds = ds.rename_column("src", "en")
    ds = ds.rename_column("tgt", "te")
    
    path = f"{dir}/source.txt"
    print(f"Writing source dataset to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds):
            f.write(entry["te"] + "\n")

    path = f"{dir}/target.txt"
    print(f"Writing target dataset to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds):
            f.write(entry["en"] + "\n")

def get_en_te_pairs(dir):
    ds = load_dataset("MRR24/English_to_Telugu_Bilingual_Sentence_Pairs")
    ds = ds["train"]

    path = f"{dir}/source.txt"
    print(f"Writing source dataset to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds):
            f.write(entry["Output"] + "\n")

    path = f"{dir}/target.txt"
    print(f"Writing target dataset to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for entry in tqdm(ds):
            f.write(entry["Input"] + "\n")

def get_en_te_kaggle(dir):
    import mlcroissant as mlc
    import pandas as pd

    src_path = f"{dir}/source.txt"
    tgt_path = f"{dir}/target.txt"
    
    # Fetch the Croissant JSON-LD
    croissant_dataset = mlc.Dataset('http://www.kaggle.com/datasets/klu2000030172/english-telugu-translation-dataset/croissant/download')
    
    # Check what record sets are in the dataset
    record_sets = croissant_dataset.metadata.record_sets
    print(record_sets)
    
    # Fetch the records and put them in a DataFrame
    df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
    df = df.set_axis(["en","te"], axis=1)

    src = open(src_path, "w", encoding="utf-8")
    tgt = open(tgt_path, "w", encoding="utf-8")
    for idx, series in df.iterrows():
        src.write(series["te"].decode("utf-8") + "\n")
        tgt.write(series["en"].decode("utf-8") + "\n")

    src.close()
    tgt.close()

def get_en_es_1(dir):
    ds = load_dataset("okezieowen/english_to_spanish")
    ds = ds["train"]

    print(f"Dataset row count: {len(ds)}")

    src_path = f"{dir}/source.txt"
    tgt_path = f"{dir}/target.txt"
    print(f"Writing dataset to {src_path} and {tgt_path}")
    with open(src_path, "w", encoding="utf-8") as src:
        with open(tgt_path, "w", encoding="utf-8") as tgt:
            for entry in tqdm(ds):
                if entry["English"] == None or entry["Spanish"] == None:
                    continue

                # Check if either contains a newline
                if "\n" in entry["English"] or "\n" in entry["Spanish"]:
                    split_en = entry["English"].split("\n")
                    split_es = entry["Spanish"].split("\n")

                    if len(split_en) != len(split_es):
                        continue
                    
                    for e, s in zip(split_en, split_es):
                        src.write(e + "\n")
                        tgt.write(s + "\n")
                else:
                    src.write(entry["English"].strip() + "\n")
                    tgt.write(entry["Spanish"].strip() + "\n")