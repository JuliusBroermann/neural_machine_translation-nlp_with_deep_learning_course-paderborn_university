import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import List
from tqdm import tqdm
import sentencepiece as spm
from training import lstm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator:
    """
    Providing methods to generate predictions from CausalLM models using different decoding strategies.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor.

        Parameters:
            model (nn.Module): Model to use for generation.
        """

        self.model = model

    def generate_argmax(self, inputs: torch.Tensor, max_generation_length: int, batch_size: int=None, bos_token_id: int=1, eos_token_id: int=2) -> List[List[int]]:
        """
        Generate translations using a greedy argmax decoding strategy.

        Parameters:
            inputs (torch.Tensor): 2D Tensor containing the token_ids of the sentences to translate.
            max_generation_length (int): Maximum number of tokens to generate per example if the eos token is not predicted before.
            batch_size (int): Batch size to use for the inference.
            bos_token_id (int): Token id of the begin of sequence token.
            eos_token_id (int): Token id of the end of sequence token.

        Returns:
            List[List[int]]: List containing the generated token_ids for each example.
        """
        
        if isinstance(self.model, lstm.LSTMEncoderDecoder):
            return self._generate_argmax_lstm_encoder_decoder(
                inputs=inputs, max_generation_length=max_generation_length, batch_size=batch_size, bos_token_id=bos_token_id, eos_token_id=eos_token_id
            )
        else:
            return self._generate_argmax_rnn_attention(
                inputs=inputs, max_generation_length=max_generation_length, batch_size=batch_size, bos_token_id=bos_token_id, eos_token_id=eos_token_id
            )
        
    def generate_argmax_from_strings(self, texts: List[str], tokenizer_path: str, max_generation_length: int) -> List[str]:
        """
        Generate translations from texts using a greedy argmax decoding strategy.

        Parameters:
            texts (List[str]): List of texts to translate.
            tokenizer_path (str): Path of the tokenizer to tokenize the texts to translate and detokenize the translations.
            max_generation_length (int): Maximum number of tokens to generate per example if the eos token is not predicted before.

        Returns:
            List[str]: List of generated translations as text.
        """

        sp = spm.SentencePieceProcessor()
        sp.Load(str(tokenizer_path))

        test_inputs = sp.Encode(
            texts,
            add_bos=True,
            add_eos=True
        )
        padded_test_inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample) for sample in test_inputs], batch_first=True, padding_value=3)

        output_tokens = self.generate_argmax(
            inputs=padded_test_inputs,
            max_generation_length=max_generation_length
        )
        return sp.Decode(output_tokens)
    
    def _generate_argmax_lstm_encoder_decoder(self, inputs: torch.Tensor, max_generation_length: int, batch_size: int=64, bos_token_id=1, eos_token_id=2):
        if batch_size is None:
            batch_size = 64

        # Create dataloader
        dataset = TensorDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Set up model
        self.model.eval()
        self.model = self.model.to(device)

        # Generate for all inputs until max_generation_length to allow efficient batch processing
        generations = []
        for (batch_inputs,) in tqdm(dataloader):
            batch_inputs = batch_inputs.to(device)
            previous_generation = torch.full(size=(batch_inputs.shape[0], 1), fill_value=bos_token_id).to(device)  # [batch_size, 1], start generation with bos token
            for _ in range(max_generation_length):
                next_token_distribution: torch.Tensor = self.model.next_token_dist(batch_inputs, previous_generation)  # [batch_size, vocab_size]
                next_token_id = next_token_distribution.argmax(dim=1, keepdim=True)  # [batch_size, 1], use argmax to select token from distribution
                previous_generation = torch.concat(tensors=(previous_generation, next_token_id), dim=1)  # [batch_size, previous+1]
            generations.extend(previous_generation.cpu().tolist())
        
        # Cut away everything after the respective eos token if it is present
        generations: List[List[int]]
        for i, generation in enumerate(generations):
            if eos_token_id in generation:
                first_eos_index = generation.index(eos_token_id)
                generations[i] = generation[:first_eos_index+1]

        return generations
    
    def _generate_argmax_rnn_attention(self, inputs: torch.Tensor, max_generation_length: int, batch_size: int, bos_token_id=1, eos_token_id=2):
        if batch_size is None:
            batch_size = 4

        # Create dataloader
        dataset = TensorDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Set up model
        self.model.eval()
        self.model = self.model.to(device)

        # Generate for all inputs until max_generation_length to allow efficient batch processing
        generations = []
        for (batch_inputs,) in tqdm(dataloader):
            batch_inputs = batch_inputs.to(device)
            teacher_target = torch.ones(size=(batch_inputs.shape[0], max_generation_length), dtype=torch.int64).to(device)  # [batch_size, max_generation_length], dummy to use for controlling the generation length + bos token

            predictions = self.model(x=batch_inputs, teacher_target=teacher_target, teacher_forching_prob=0.0)  # no teacher forcing, teacher targets are just there for controlling the length + starting with bos
            # logits shape: [batch_size, max_generation_length, vocab_size]
            predictions = F.softmax(predictions, dim=2)  # [batch_size, max_generation_length, vocab_size]
            predictions = predictions.argmax(dim=2, keepdim=False)  # [batch_size, max_generation_length]

            generations.extend(predictions.cpu().tolist())
        
        # Cut away everything after the respective eos token if it is present
        generations: List[List[int]]
        for i, generation in enumerate(generations):
            if eos_token_id in generation:
                first_eos_index = generation.index(eos_token_id)
                generations[i] = generation[:first_eos_index+1]

        return generations
