import requests
import tarfile
import io
import numpy as np
import pickle
import sentencepiece as spm
from typing import List
from pathlib import Path

from datafiles import datafiles_path
from models import models_path


def download_data() -> None:
    """
    Download the data files for the English to German wmt16 translation task.
    """

    # Download the tgz file
    url = "https://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz"
    print("Downloading", url)
    response = requests.get(url)
    tgz_file = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")

    # Only save english and german files
    file_names_keep = (
        "news-commentary-v11.de-en.en",
        "news-commentary-v11.de-en.de"
    )
    for member in tgz_file.getmembers():
        if member.name.split("/")[-1] in file_names_keep:
            print("Extracting", member.name.split("/")[-1])
            tgz_file.extract(member, path=datafiles_path)


def split_data(seed: int=42) -> None:
    """
    Split the dataset into train (80%), val (10%) and test splits (10%).

    Parameters:
        seed (int): Seed to use for the random splitting.
    """

    # Load data
    with open(datafiles_path / "training-parallel-nc-v11" / "news-commentary-v11.de-en.en", 'rb') as f:
        data_en = f.readlines()
        data_en = np.array([line.decode()[:-2] for line in data_en])  # remove \n

    with open(datafiles_path / "training-parallel-nc-v11" / "news-commentary-v11.de-en.de", 'rb') as f:
        data_de = f.readlines()
        data_de = np.array([line.decode()[:-2] for line in data_de])  # remove \n

    # Permutate both languages the same
    np.random.seed(seed)
    perm = np.random.permutation(len(data_en))
    data_en = data_en[perm]
    data_de = data_de[perm]

    # Create splits
    num_train = int(len(data_en) * 0.8)
    num_val = int(len(data_en) * 0.1)
    train_en, val_en, test_en = data_en[:num_train], data_en[num_train:(num_train + num_val)], data_en[(num_train + num_val):]
    train_de, val_de, test_de = data_de[:num_train], data_de[num_train:(num_train + num_val)], data_de[(num_train + num_val):]

    # Save splits
    (datafiles_path / "splits").mkdir(parents=True, exist_ok=True)
    for name, array in (
        ("train_en.pkl", train_en),
        ("val_en.pkl", val_en),
        ("test_en.pkl", test_en),
        ("train_de.pkl", train_de),
        ("val_de.pkl", val_de),
        ("test_de.pkl", test_de),
    ):
        with open(datafiles_path / "splits" / name, "wb") as f:
            pickle.dump(array, f)


def get_data(language: str, split: str) -> np.ndarray:
    """
    Load and return the specified data split.

    Parameters:
        language (str): One of ("en", "de").
        split (str): One of ("train", "val", "test").

    Returns:
        np.array: Array containing the text of the specified split.
    """

    # Load the data split from the corresponding pickle file
    with open(datafiles_path / "splits" / (split + "_" + language + ".pkl"), "rb") as f:
        return pickle.load(f)


def train_tokenizer(data: np.ndarray, vocab_size: int=8_000) -> Path:
    """
    Trains a SentencePieceBPE tokenizer on the given text data.

    Parameters:
        data (np.ndarray): 1D array with each entry containing a sentence.
        vocab_size (int): Number of tokens to use.

    Returns:
        Path: Path of the trained tokenizer.
    """

    # Create temporary file for tokenizer training
    np.savetxt(datafiles_path / "temp_tokenizer_training_data.txt", data, fmt="%s", encoding="utf-8")

    # Tokenizer dir
    tokenizer_dir = models_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Train the tokenizer
    spm.SentencePieceTrainer.train(
        input=datafiles_path / "temp_tokenizer_training_data.txt", 
        model_prefix=tokenizer_dir / f"en_de_{vocab_size}", 
        model_type="bpe", 
        vocab_size=vocab_size,
        pad_id=3
    )

    (datafiles_path / "temp_tokenizer_training_data.txt").unlink()

    return tokenizer_dir / f"en_de_{vocab_size}.model"


def encode(language: str, split: str, tokenizer_path: str, add_bos: bool=True, add_eos: bool=True) -> List[List[int]]:
    """
    Encode a data split to token_ids.

    Parameters:
        language (str): Language of the split to use. One of ("en", "de").
        split (str): One of ("train", "val", "test").
        tokenizer_path (str): Path of the tokenizer to use.
        add_bos (bool): Whether to add a begin of sequence token (True) or not (False).
        add_eos (bool): Whether to add an end of sequence token (True) or not (False).

    Returns:
        List[List[int]]: 2D list with each row containing a tokenized example.
    """

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))

    # Load data
    data = get_data(language=language, split=split)

    # Encode the text to token_ids, 
    # adding begin of sequence and end of sequence tokens if wanted
    return sp.Encode(data.tolist(), add_bos=add_bos, add_eos=add_eos)

def decode(data: List[List[int]], tokenizer_path: str) -> List[str]:
    """
    Decode token_ids to text.

    Parameters:
        data (List[List[int]]): 2D list with each row containing a tokenized example.
        tokenizer_path (str): Path of the tokenizer to use.

    Returns:
        List[str]: List containing the decoded texts.
    """

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))

    # Decode the token_ids to texts
    # Ignores bos and eos tokens
    return sp.Decode(data)
