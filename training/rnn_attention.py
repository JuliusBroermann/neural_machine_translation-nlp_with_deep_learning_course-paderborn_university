import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Tuple
import random
import numpy as np
import wandb

from models import models_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNAttentionEncoder(nn.Module):
    """
    Encoder of the GRU based Encoder Decoder architecture with attention for translation
    following Bahdanau, Cho and Bengio (2014): Neural Machine Translation by Jointly Learning to Align and Translate. https://arxiv.org/abs/1409.0473.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dimension: int=620,
        hidden_size: int=1000
    ) -> None:
        """
        Constructor.

        Parameters:
            num_embeddings (int): Number of embeddings to use. Equal to the size of the vocabulary.
            embedding_dimension (int): Size of the embeddings.
            hidden_size (int): Size of the hidden state in the GRU cells in both the encoder and decoder.
        """

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dimension)  # used for both encoder and decoder to reduce amount of parameters
        # Bahdanau, Cho and Bengio use gated hidden units by Cho instead of LSTMs, although mentioning them in the appendix as a possible alternative
        self.gru = nn.GRU(input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(
        self, 
        x: torch.Tensor  # [batch_size, sequence_length]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Parameters:
            x (torch.Tensor): Input token ids of the sentences to translate. Shape: [batch_size, sequence_length]

        Returns:
            torch.Tensor: Encoder outputs of the GRU cells. Shape: [batch_size, sequence_length, hidden_size * 2]
            torch.Tensor: Projected last hidden state of the backward direction of the bidirectional GRU layer. Shape: [batch_size, hidden_size]
        """
        x = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]
        outputs, final_hidden_states = self.gru(x)  # [batch_size, sequence_length, hidden_size * 2], [2, batch_size, hidden_size]

        # index 0 is forward pass, index 1 backward pass
        backward_final_hidden_state = final_hidden_states[1]  # [batch_size, hidden_size]

        # Following Bahdanau, Cho and Bengio a linear transformation + tanh of the last hidden state of the backward pass is used 
        # as the initialization of the decoders first hidden state. It might be better to use the concatenation 
        # of both the last states of the forward and backward pass, but I follow the paper (Appendix A.2.2).
        s_0 = F.tanh(self.linear(backward_final_hidden_state))

        return outputs, s_0  # [batch_size, sequence_length, hidden_size * 2], [batch_size, hidden_size]


class Alignment(nn.Module):
    """
    Attention module of the GRU based Encoder Decoder architecture with attention for translation
    following Bahdanau, Cho and Bengio (2014): Neural Machine Translation by Jointly Learning to Align and Translate. https://arxiv.org/abs/1409.0473.
    """

    def __init__(self, attention_dim: int, hidden_size: int) -> None:
        """
        Constructor.

        Parameters:
            attention_dim (int): Size of the tensors used for calculating the attention.
            hidden_size (int): Size of the hidden state in the GRU cells in both the encoder and decoder.
        """

        super().__init__()

        self.hidden_projection = nn.Linear(in_features=hidden_size, out_features=attention_dim)  # called Wa in the paper
        self.outputs_projection = nn.Linear(in_features=hidden_size * 2, out_features=attention_dim)  # called Ua in the paper
        self.energy_scale = nn.Linear(in_features=attention_dim, out_features=1, bias=False)  # called va in the paper

    def forward(
        self, 
        hidden: torch.Tensor,  # [batch_size, hidden_size]
        outputs: torch.Tensor  # [batch_size, sequence_length, hidden_size * 2]
    ) -> torch.Tensor:
        """
        Calculation of attention weights.

        Parameters:
            hidden (torch.Tensor): Hidden state of the decoder GRU cell to calculate the attention for. Shape: [batch_size, hidden_size]
            outputs (torch.Tensor): Outputs of the encoder GRU layer to attend to. Shape: [batch_size, sequence_length, hidden_size * 2]

        Returns:
            torch.Tensor: Attention weights for the encoder GRU layer outputs. Shape: [batch_size, sequence_length]
        """
        
        sequence_length = outputs.shape[1]
        hidden_repeated = hidden.unsqueeze(dim=1).repeat(1, sequence_length, 1)  # [batch_size, sequence_length, hidden_size]
        
        energy: torch.Tensor = self.energy_scale(F.tanh(self.hidden_projection(hidden_repeated) + self.outputs_projection(outputs)))  # [batch_size, sequence_length, 1]
        energy = energy.squeeze(2)  # [batch_size, sequence_length]

        return F.softmax(energy, dim=1)  # [batch_size, sequence_length]
    

class RNNAttentionDecoder(nn.Module):
    """
    Decoder of the GRU based Encoder Decoder architecture with attention for translation
    following Bahdanau, Cho and Bengio (2014): Neural Machine Translation by Jointly Learning to Align and Translate. https://arxiv.org/abs/1409.0473.
    """

    def __init__(
        self,
        embedding_layer: nn.Embedding,  # shared embeddings because of sentencepiece, deviation from the paper
        hidden_size: int=1000,  # has to be the same as in the encoder in this implementation and in the paper, in general could be different
        attention_dim: int=1000,
        maxout_size: int=500
    ) -> None:
        """
        Constructor.

        Parameters:
            embedding_layer (nn.Embedding): Embedding layer of the encoder to reuse. Simplification for parameter saving. If you have the resources, use a second embeddinglayer.
            hidden_size (int): Size of the hidden state in the GRU cells in both the encoder and decoder.
            attention_dim (int): Size of the tensors used for calculating the attention.
            maxout_size (int): Size of the tensor used in the maxout layer.
        """

        super().__init__()

        self.attention = Alignment(attention_dim=attention_dim, hidden_size=hidden_size)
        self.embedding = embedding_layer  # used for both encoder and decoder because of sentencepiece
        # Bahdanau, Cho and Bengio use gated hidden units by Cho instead of LSTMs, although mentioning them in the appendix as a possible alternative
        self.gru = nn.GRU(input_size=hidden_size * 2 + self.embedding.embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.linear_weighted_values = nn.Linear(in_features=hidden_size * 2, out_features=maxout_size * 2)
        self.linear_hidden = nn.Linear(in_features=hidden_size, out_features=maxout_size * 2)
        self.linear_embeddings = nn.Linear(in_features=self.embedding.embedding_dim, out_features=maxout_size * 2)
        self.linear_prediction = nn.Linear(in_features=maxout_size, out_features=self.embedding.num_embeddings)

    def forward(
        self, 
        x: torch.Tensor,  # [batch_size, 1]
        hidden: torch.Tensor,  # [batch_size, hidden_size]
        outputs: torch.Tensor  # [batch_size, sequence_length, hidden_size * 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts only the next output given one current decoding input, the last hidden state and the encoder outputs.

        Parameters:
            x (torch.Tensor): Input token ids of the last predicted token used for the autoregressive decoding. Shape: [batch_size, 1]
            hidden (torch.Tensor): Hidden state passed to the current GRU cell. Shape: [batch_size, hidden_size]
            ouputs (torch.Tensor): Outputs of the encoder GRU layer to attend to. Shape: [batch_size, sequence_length, hidden_size * 2]

        Returns:
            torch.Tensor: Logits for the next token. Shape: [batch_size, vocab_size]
            torch.Tensor: Hidden state to pass to the next GRU cell. Shape: [batch_size, hidden_size]
        """

        # Attention/Alignment
        attention_weights = self.attention(hidden=hidden, outputs=outputs)  # [batch_size, sequence_length]

        prepared_attention_weights = attention_weights.unsqueeze(dim=1)  # [batch_size, 1, sequence_length]
        weighted_values = torch.bmm(prepared_attention_weights, outputs)  # batched matrix multiplication, [batch_size, 1, hidden_size * 2]
        
        # GRU
        x = self.embedding(x)  # [batch_size, 1, embedding_dim]
        gru_input = torch.concat((x, weighted_values), dim=2)  # [batch_size, 1, embedding_dim + hidden_size * 2]
        _, decoder_hidden = self.gru(gru_input, hidden.unsqueeze(dim=0))  # [batch_size, 1, hidden_size], [1, batch_size, hidden_size]

        # Deep output with single maxout hidden layer (see paper Appendix A.2.2 at the end)
        maxout_input = self.linear_hidden(hidden) + self.linear_embeddings(x.squeeze(dim=1)) + self.linear_weighted_values(weighted_values.squeeze(dim=1))  # [batch_size, maxout_size * 2]
        maxout_output = torch.reshape(maxout_input, (-1, maxout_input.shape[1]//2, 2)).max(dim=2, keepdim=False).values  # [batch_size, maxout_size]
        prediction = self.linear_prediction(maxout_output)  # [batch_size, num_embeddings] = [batch_size, vocab_size]

        return prediction, decoder_hidden.squeeze(dim=0)  # [batch_size, num_embeddings] = [batch_size, vocab_size], [batch_size, hidden_size]


class RNNAttention(nn.Module):
    """
    Implementing a GRU based Encoder Decoder architecture with attention for translation
    following Bahdanau, Cho and Bengio (2014): Neural Machine Translation by Jointly Learning to Align and Translate. https://arxiv.org/abs/1409.0473.
    """

    def __init__(self, encoder: RNNAttentionEncoder, decoder: RNNAttentionDecoder, random_seed: int=42) -> None:
        """
        Constructor.

        Parameters:
            encoder (RNNAttentionEncoder): Encoder part of the model.
            decoder (RNNAttentionDecoder): Decoder part of the model.
            random_seed (int): Seed for reproducible training.
        """

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.random = random.Random(random_seed)

    def forward(
        self, 
        x: torch.Tensor,  # [batch_size, sequence_length]
        teacher_target: torch.Tensor,  # [batch_size, target_translation_length], includes bos but no eos token
        teacher_forching_prob: float=0.5
    ) -> torch.Tensor:
        """
        Forward pass using teacher forcing with probability 'teacher_forching_prob' for each token.

        Parameters:
            x (torch.Tensor): Input token ids of the sentences to translate. Shape: [batch_size, sequence_length]
            teacher_target (torch.Tensor): Token ids of the reference translation used for teacher forcing. Shape: [batch_size, target_translation_length]
            teacher_forching_prob (float): Probability with which to use teacher forcing. Applied to each token independently.

        Returns:
            torch.Tensor: Logits for each target position and token in the vocab. Shape: [batch_size, target_translation_length, vocab_size]
        """

        encoder_outputs, hidden = self.encoder(x)  # [batch_size, sequence_length, hidden_size * 2], [batch_size, hidden_size]
        decoder_outputs = torch.zeros(teacher_target.shape[1], teacher_target.shape[0], self.decoder.linear_prediction.out_features)  # [target_translation_length, batch_size, vocab_size]
        decoder_outputs = decoder_outputs.to(device)
        inputs = teacher_target[:, 0]  # this are the bos tokens for starting the generation # [batch_size]

        for t in range(0, teacher_target.shape[1]):
            decoder_output_t, hidden = self.decoder(x=inputs.unsqueeze(dim=1), hidden=hidden, outputs=encoder_outputs)  # [batch_size, vocab_size], [batch_size, hidden_size]
            decoder_outputs[t] = decoder_output_t

            # Get next inputs, use teacher forcing with probability teacher_forching_prob
            if t < teacher_target.shape[1] - 1:
                decoder_prediction = torch.argmax(decoder_output_t, dim=1, keepdim=False)  # [batch_size]
                inputs = teacher_target[:, t+1] if self.random.random() < teacher_forching_prob else decoder_prediction  # [batch_size]

        return decoder_outputs.permute(1, 0, 2)  # [batch_size, target_translation_length, vocab_size]
    
    def fit(
        self, 
        inputs: torch.Tensor,  # [dataset_size, sequence_length]
        teacher_targets: torch.Tensor,  # [dataset_size, target_translation_length], includes bos but no eos token
        loss_targets: torch.Tensor,  # [dataset_size, target_translation_length], includes eos but no bos token
        val_inputs: torch.Tensor, 
        val_teacher_targets: torch.Tensor, 
        val_loss_targets: torch.Tensor,
        num_epochs: int, 
        batch_size: int, 
        lr: float,
        val_steps: int, 
        early_stopping_patience: int,
        model_name: str,
        teacher_forching_prob: float=0.5,
    ) -> None:
        """
        Training loop for fitting the model to the provided training data.

        Parameters:
            inputs (torch.Tensor): Token ids of the sentences to translate. Should have both a bos and an eos token.
            teacher_targets (torch.Tensor): Token ids of the reference translations. Should have a bos but no eos token.
            loss_targets (torch.Tensor): Token ids of the reference translations. Should have no bos but an eos token (shifted).
            val_inputs (torch.Tensor): Token ids of the sentences to translate for validation. Should have both a bos and an eos token.
            val_teacher_targets (torch.Tensor): Token ids of the reference translations for valdiation. Should have a bos but no eos token.
            val_loss_targets (torch.Tensor): Token ids of the reference translations for validation. Should have no bos but an eos token (shifted).
            num_epochs (int): Number of epochs to train for. One epoch is one pass over the training data.
            batch_size (int): Number of samples per mini-batch.
            lr (float): Learning rate. 
            val_steps (int): Number of update steps between each validation pass.
            early_stopping_patience (int): Number of validations allowed without the val loss improving before early stopping the training.
            model_name (str): Name of the model to use when saving the model.
            teacher_forching_prob (float): Probability with which to use teacher forcing. Applied to each token independently.
        """

        train_dataset = TensorDataset(inputs, teacher_targets, loss_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(val_inputs, val_teacher_targets, val_loss_targets)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self = self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=3)  # ignore loss at padding positions

        best_val_loss = np.inf
        num_evaluations_without_improvement = 0

        for epoch in range(num_epochs):
            epoch_train_loss = 0.0
            batch_tqdm = tqdm(train_dataloader, desc="Batch")
            for i, (batch_inputs, batch_teacher_targets, batch_loss_targets) in enumerate(batch_tqdm):
                self.train()
                batch_inputs, batch_teacher_targets, batch_loss_targets = batch_inputs.to(device), batch_teacher_targets.to(device), batch_loss_targets.to(device)
                optimizer.zero_grad()

                outputs = self(x=batch_inputs, teacher_target=batch_teacher_targets, teacher_forching_prob=teacher_forching_prob)  # [batch_size, target_translation_length, vocab_size]

                loss = criterion(outputs.permute(0, 2, 1), batch_loss_targets)  # CrossEntropyLoss expects output shape to be (batch_size, num_classes, sequence_length)
                batch_loss = loss.item()
                epoch_train_loss += batch_loss
                batch_tqdm.set_postfix({"Current batch loss": batch_loss, "Average batch loss": epoch_train_loss/(i+1)})

                loss.backward()
                optimizer.step()

                # Validation
                if i > 0 and (i % val_steps) == 0:

                    del batch_inputs, batch_teacher_targets, batch_loss_targets
                    num_evaluations_without_improvement = 0
                    
                    running_val_loss = 0.0
                    self.eval()
                    with torch.no_grad():
                        for j, (batch_val_inputs, batch_val_teacher_targets, batch_val_loss_targets) in enumerate(val_dataloader):
                            batch_val_inputs, batch_val_teacher_targets, batch_val_loss_targets = batch_val_inputs.to(device), batch_val_teacher_targets.to(device), batch_val_loss_targets.to(device)
                            val_outputs = self(x=batch_val_inputs, teacher_target=batch_val_teacher_targets, teacher_forching_prob=0.0)  # no teacher forcing for validation
                            val_loss = criterion(val_outputs.permute(0, 2, 1), batch_val_loss_targets)
                            running_val_loss += val_loss.item()

                    avg_val_loss = running_val_loss / (j + 1)
                    print(f"Validation loss in step {i}:", avg_val_loss)
                    wandb.log({"val/loss": avg_val_loss})

                    # Track best performance, and save the model's state
                    if avg_val_loss < best_val_loss:
                        print("Validation loss improved, saving")
                        best_val_loss = avg_val_loss
                        # Save the model
                        torch.save(self.state_dict(), models_path / f"{model_name}.pt")
                    else:
                        num_evaluations_without_improvement += 1
                        # Stop training early if validation loss didn't improve for 'early_stopping_patience' number of evaluations
                        if num_evaluations_without_improvement >= early_stopping_patience:
                            print("Early stopping")
                            return
                        
                wandb.log({"train/batch_loss": batch_loss, "train/avg_loss_in_epoch": epoch_train_loss/(i+1)})
