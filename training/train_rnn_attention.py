import fire
import sentencepiece as spm
import torch
import dotenv
import wandb
import os

from models import models_path
import rnn_attention
from data import data_preparation

dotenv.load_dotenv()
wandb.login(key=os.environ["WANDB_KEY"])


def train_rnn_attention(
    model_name: str,
    num_epochs=5,
    batch_size=4,
    lr=1e-6,  # following the paper
    teacher_forching_prob=0.5,
    embedding_dimension=512,  # in the paper it is 620
    hidden_size=512,  # in the paper it is 1000
    attention_dim=256,  # in the paper it is 1000
    maxout_size=256,  # in the paper it is 500
    val_steps=8000,
    early_stopping=3
):
    """
    Train a LSTM based Encoder Decoder architecture for translation 
    on the WMT16 translation task from English to German.

    Parameters:
        model_name (str): Name of the model to use when saving the model.
        num_epochs (int): Number of epochs to train for. One epoch is one pass over the training data.
        batch_size (int): Number of samples per mini-batch.
        lr (float): Learning rate.
        teacher_forching_prob (float): Probability with which to use teacher forcing. Applied to each token independently.
        embedding_dimension (int): Size of the embeddings.
        hidden_size (int): Size of the hidden state in the LSTM cells in both the encoder and decoder.
        attention_dim (int): Size of the tensors used for calculating the attention.
        maxout_size (int): Size of the tensor used in the maxout layer.
        val_steps (int): Number of update steps between each validation pass.
        early_stopping_patience (int): Number of validations allowed without the val loss improving before early stopping the training.
    """
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=os.environ["WANDB_PROJECT"],

        # track hyperparameters and run metadata
        config={
            "model_name": model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "teacher_forching_prob": teacher_forching_prob,
            "embedding_dimension": embedding_dimension,
            "hidden_size": hidden_size,
            "attention_dim": attention_dim,
            "maxout_size": maxout_size,
            "val_steps": val_steps,
            "early_stopping": early_stopping
        }
    )
    
    # Load tokenizer
    tokenizer_path = models_path / "tokenizer" / "en_de_8000.model"
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))

    # Initialize model
    rnn_attention_encoder = rnn_attention.RNNAttentionEncoder(
        num_embeddings=sp.vocab_size(),
        embedding_dimension=embedding_dimension,
        hidden_size=hidden_size
    )
    rnn_attention_model = rnn_attention.RNNAttention(
        encoder=rnn_attention_encoder,
        decoder=rnn_attention.RNNAttentionDecoder(
            embedding_layer=rnn_attention_encoder.embedding,
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            maxout_size=maxout_size
        ),
        random_seed=42
    )

    # Print overview
    print("Model loaded")
    print(rnn_attention_model)
    print("Number of parameters:", sum(p.numel() for p in rnn_attention_model.parameters() if p.requires_grad))

    # Load training data
    inputs = data_preparation.encode(language="en", split="train", tokenizer_path=tokenizer_path, add_bos=True, add_eos=True)
    teacher_targets = data_preparation.encode(language="de", split="train", tokenizer_path=tokenizer_path, add_bos=True, add_eos=False)
    loss_targets = data_preparation.encode(language="de", split="train", tokenizer_path=tokenizer_path, add_bos=False, add_eos=True)

    # Padding
    inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in inputs], batch_first=True, padding_value=3)
    teacher_targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in teacher_targets], batch_first=True, padding_value=3)
    loss_targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in loss_targets], batch_first=True, padding_value=3)

    # Load validation data
    val_inputs = data_preparation.encode(language="en", split="val", tokenizer_path=tokenizer_path, add_bos=True, add_eos=True)
    val_teacher_targets = data_preparation.encode(language="de", split="val", tokenizer_path=tokenizer_path, add_bos=True, add_eos=False)
    val_loss_targets = data_preparation.encode(language="de", split="val", tokenizer_path=tokenizer_path, add_bos=False, add_eos=True)

    # Padding
    val_inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in val_inputs], batch_first=True, padding_value=3)
    val_teacher_targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in val_teacher_targets], batch_first=True, padding_value=3)
    val_loss_targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in val_loss_targets], batch_first=True, padding_value=3)
    
    print("Data loaded and prepared")
    print("Start training")

    # Start training
    rnn_attention_model.fit(
        inputs=inputs,
        teacher_targets=teacher_targets,
        loss_targets=loss_targets,
        val_inputs=val_inputs,
        val_teacher_targets=val_teacher_targets,
        val_loss_targets=val_loss_targets,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_steps=val_steps,
        early_stopping_patience=early_stopping,
        model_name=model_name,
        teacher_forching_prob=teacher_forching_prob
    )

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(train_rnn_attention)  # Create a command line script from the method
