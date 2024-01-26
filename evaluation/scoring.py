from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import sentencepiece as spm
from typing import List
import torch
import torch.nn as nn
import pickle

from evaluation import evaluation_path
from data import data_preparation
from generation import generation_strategies


def sentence_bleu_from_token_ids(references: List[List[int]], hypotheses: List[List[int]], tokenizer_path: str) -> List[float]:
    """
    Calculate sentence level BLEU score rounded to 6 decimal places.

    Parameters:
        references (List[List[int]]): Tokenized references serving as ground truth.
        hypotheses (List[List[int]]): Tokenized hypotheses to evaluate.
        tokenizer_path (str): Path of the tokenizer to use.

    Returns:
        List[float]: List containing the BLEU score rounded to 6 decimal places for each hypotheses.
    """

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))
    references_encoded = [[[sp.IdToPiece(token_id) for token_id in reference]] for reference in references]  # Additional brackets because multiple references can be supplied
    hypotheses_encoded = [[sp.IdToPiece(token_id) for token_id in hypothesis] for hypothesis in hypotheses]
    return [round(sentence_bleu(references=reference, hypothesis=hypothesis), 6) for reference, hypothesis in zip(references_encoded, hypotheses_encoded)]


def corpus_bleu_from_token_ids(references: List[List[int]], hypotheses: List[List[int]], tokenizer_path: str) -> float:
    """
    Calculate corpus level BLEU score rounded to 6 decimal places.

    Parameters:
        references (List[List[int]]): Tokenized references serving as ground truth.
        hypotheses (List[List[int]]): Tokenized hypotheses to evaluate.
        tokenizer_path (str): Path of the tokenizer to use.

    Returns:
        float: Corpus BLEU score rounded to 6 decimal places.
    """

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))
    references_encoded = [[[sp.IdToPiece(token_id) for token_id in reference]] for reference in references]  # Additional brackets because multiple references can be supplied
    hypotheses_encoded = [[sp.IdToPiece(token_id) for token_id in hypothesis] for hypothesis in hypotheses]
    return round(corpus_bleu(list_of_references=references_encoded, hypotheses=hypotheses_encoded), 6)

def corpus_bleu_model(model: nn.Module, tokenizer_path: str, model_name: str, batch_size: int) -> float:
    """
    Make predictions using the specified model and calculate their corpus level BLEU score.

    Parameters:
        model (nn.Module): Model to use to make predictions.
        tokenizer_path (str): Path of the tokenizer to use.
        model_name (str): Name of the model to use for saving the predictions.
        batch_size (int): Batch size to use for inference.

    Returns:
        float: Corpus level BLEU score of the predictions of the model.
    """

    # Load test data
    en_test = data_preparation.encode(language="en", split="test", tokenizer_path=tokenizer_path, add_bos=True, add_eos=True)
    padded_en_test = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in en_test], batch_first=True, padding_value=3)
    de_test = data_preparation.encode(language="de", split="test", tokenizer_path=tokenizer_path, add_bos=False, add_eos=False)
    max_translation_len = len(max(de_test, key=lambda x: len(x)))

    # Generate translations for test data
    generator = generation_strategies.Generator(model=model)
    output_tokens = generator.generate_argmax(
        inputs=padded_en_test,
        max_generation_length=max_translation_len,
        batch_size=batch_size
    )
    # Remove bos token (which is always there) and eos token (if it was generated)
    output_tokens = [translation[1:-1] if translation[-1]==2 else translation[1:] for translation in output_tokens]

    # Save predictions
    with open(evaluation_path / f"{model_name}_predictions_test_split.pkl", "wb") as f:
        pickle.dump(output_tokens, f)

    # Calculate BLEU score
    return corpus_bleu_from_token_ids(references=de_test, hypotheses=output_tokens, tokenizer_path=tokenizer_path)
