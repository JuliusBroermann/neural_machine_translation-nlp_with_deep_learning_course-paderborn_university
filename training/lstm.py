import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import wandb

from models import models_path


class LSTMEncoderDecoder(nn.Module):
    """
    Implementing a LSTM based Encoder Decoder architecture for translation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dimension: int=768,
        hidden_size: int=1024,
        bidirectional: bool=False

    ) -> None:
        """
        Constructor.

        Parameters:
            num_embeddings (int): Number of embeddings to use. Equal to the size of the vocabulary.
            embedding_dimension (int): Size of the embeddings.
            hidden_size (int): Size of the hidden state in the LSTM cells in both the encoder and decoder.
            bidirectional (bool): Whether to use a bidirectional encoder (True) or a unidirectional encoder (False).
        """

        super().__init__()

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dimension)  # used for both encoder and decoder

        if self.bidirectional:
            self.encoder = nn.LSTM(input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            self.hidden_projection = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
            self.cell_projection = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        else:
            self.encoder = nn.LSTM(input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True)
        
        self.decoder = nn.LSTM(input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True)
        self.final_linear = nn.Linear(in_features=hidden_size, out_features=num_embeddings)

    def forward(
        self, 
        x: torch.Tensor,  # [batch_size, sequence_length]
        teacher_target: torch.Tensor  # [batch_size, target_translation_length]
    ) -> torch.Tensor:
        """
        Forward pass of the model using teacher forcing.

        Parameters:
            x (torch.Tensor): Input token ids of the sentences to translate. Shape: [batch_size, sequence_length]
            teacher_target (torch.Tensor): Token ids of the reference translation used for teacher forcing. Shape: [batch_size, target_translation_length]

        Returns:
            torch.Tensor: Logits for each target position and token in the vocab. Shape: [batch_size, target_translation_length, vocab_size]
        """
        x = self.embedding(x)  # [batch_size, sequence_length, embedding_dimension]
        _, (encoder_final_hidden_state, encoder_final_cell_state) = self.encoder(x)  # ([1 or 2, batch_size, hidden_size], [1 or 2, batch_size, hidden_size]), batch is never first dim here

        if self.bidirectional:
            # Concatenation
            _, batch_size, hidden_size = encoder_final_hidden_state.shape  # 2, batch_size, hidden_size
            encoder_final_hidden_state = encoder_final_hidden_state.permute(1, 0, 2).reshape((batch_size, 2 * hidden_size))  # [batch_size, 2 * hidden_size]
            encoder_final_cell_state = encoder_final_cell_state.permute(1, 0, 2).reshape((batch_size, 2 * hidden_size))  # [batch_size, 2 * hidden_size]
            # Projection
            encoder_final_hidden_state = self.hidden_projection(encoder_final_hidden_state)  # [batch_size, hidden_size]
            encoder_final_cell_state = self.hidden_projection(encoder_final_cell_state)  # [batch_size, hidden_size]
            # Adding first dimension to match required LSTM cell shape [num_layers, batch_size, hidden_size]
            encoder_final_hidden_state = encoder_final_hidden_state.unsqueeze(dim=0)  # [1, batch_size, hidden_size]
            encoder_final_cell_state = encoder_final_cell_state.unsqueeze(dim=0)  # [1, batch_size, hidden_size]

        teacher_embeddings = self.embedding(teacher_target)  # [batch_size, target_translation_length, embedding_dimension]
        decoder_outputs, (_, _) = self.decoder(teacher_embeddings, (encoder_final_hidden_state, encoder_final_cell_state))  # [batch_size, target_translation_length, hidden_size]
        projected_decoder_outputs = self.final_linear(decoder_outputs)  # [batch_size, target_translation_length, num_embeddings] = [batch_size, target_translation_length, vocab_size]
        # no softmax during training because cel expects logits

        return projected_decoder_outputs
    
    def next_token_dist(
        self,
        x: torch.Tensor,  # [batch_size, sequence_length]
        previous_generation: torch.Tensor  # [batch_size, previous_generation_length]
    ) -> torch.Tensor:
        """
        Predicts a probability distribution over the vocabulary for the next token following the previous generation.

        Parameters:
            x (torch.Tensor): Input token ids of the sentences to translate. Shape: [batch_size, sequence_length]
            previous_generation (torch.Tensor): Token ids of the already generated part of the translation. Shape: [batch_size, previous_generation_length]

        Returns:
            torch.Tensor: Probability distribution over the vocabulary for the next token following the previous generation. Shape: [batch_size, vocab_size]
        """
        projected_decoder_output = self(x, previous_generation)  # [batch_size, target_translation_length, num_embeddings] = [batch_size, target_translation_length, vocab_size]
        projected_final_decoder_output = projected_decoder_output[:, -1, :]  # [batch_size, num_embeddings] = [batch_size, vocab_size]
        # For next token do apply softmax
        projected_final_decoder_output = F.softmax(projected_final_decoder_output, dim=1)  # [batch_size, num_embeddings] = [batch_size, vocab_size]

        return projected_final_decoder_output  # [batch_size, num_embeddings] = [batch_size, vocab_size]
    
    def fit(
        self, 
        inputs: torch.Tensor, 
        teacher_targets: torch.Tensor, 
        loss_targets: torch.Tensor, 
        val_inputs: torch.Tensor, 
        val_teacher_targets: torch.Tensor, 
        val_loss_targets: torch.Tensor, 
        num_epochs: int, 
        batch_size: int, 
        lr: float, 
        val_steps: int, 
        early_stopping_patience: int,
        model_name: str
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
        """
        train_dataset = TensorDataset(inputs, teacher_targets, loss_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(val_inputs, val_teacher_targets, val_loss_targets)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

                outputs = self(batch_inputs, batch_teacher_targets)

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
                            val_outputs = self(batch_val_inputs, batch_val_teacher_targets)
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
