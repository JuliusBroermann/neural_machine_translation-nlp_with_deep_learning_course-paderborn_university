import fire
import pickle
import torch
from data import data_preparation
from datafiles import datafiles_path
from generation import generation_strategies
from models import models_path
from training import lstm, rnn_attention


def predict_test_translations(model_name: str, batch_size: int=4) -> None:
    """
    Use the specified model to generate predictions for the test set.

    Parameters:
        model_name (str): Name of the model to load and use.
        batch_size (int): Number of examples to use per batch.
    """

    if model_name == "encoder-decoder":
        # Initialize model
        model = lstm.LSTMEncoderDecoder(
            num_embeddings=8000,
            embedding_dimension=512,
            hidden_size=700,
            bidirectional=False
        )
        state_dict = torch.load(models_path / "encoder-decoder.pt")
        model.load_state_dict(state_dict)
        model.eval()
    elif model_name == "encoder-decoder-bidirectional":
        # Initialize model
        model = lstm.LSTMEncoderDecoder(
            num_embeddings=8000,
            embedding_dimension=512,
            hidden_size=700,
            bidirectional=True
        )
        state_dict = torch.load(models_path / "encoder-decoder-bidirectional.pt")
        model.load_state_dict(state_dict)
        model.eval()
    elif model_name == "rnn-attention":
        # Initialize model
        model_encoder = rnn_attention.RNNAttentionEncoder(
            num_embeddings=8000,
            embedding_dimension=512,
            hidden_size=512
        )
        model = rnn_attention.RNNAttention(
            encoder=model_encoder,
            decoder=rnn_attention.RNNAttentionDecoder(
                embedding_layer=model_encoder.embedding,
                hidden_size=512,
                attention_dim=256,
                maxout_size=256
            ),
            random_seed=42
        )
        state_dict = torch.load(models_path / "rnn-attention.pt")
        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print("Loaded model")


    # Load test data
    tokenizer_path=(models_path / "tokenizer" / "en_de_8000.model")
    en_test = data_preparation.encode(language="en", split="test", tokenizer_path=tokenizer_path, add_bos=True, add_eos=True)
    padded_en_test = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in en_test], batch_first=True, padding_value=3)
    de_test = data_preparation.encode(language="de", split="test", tokenizer_path=tokenizer_path, add_bos=False, add_eos=False)
    max_translation_len = len(max(de_test, key=lambda x: len(x)))
    print("Loaded data")

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
    with open(datafiles_path / "predictions" / f"{model_name}_predictions_test_split.pkl", "wb") as f:
        pickle.dump(output_tokens, f)


if __name__ == "__main__":
    fire.Fire(predict_test_translations)  # Create a command line script from the method
