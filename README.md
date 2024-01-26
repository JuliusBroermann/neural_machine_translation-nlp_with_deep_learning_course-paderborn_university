# Neural Machine Translation - NLP with Deep Learning course @ Paderborn University

Homework project on Neural Machine Translation by Julius Broermann for the course "Natural Language Processing with Deep Learning" at Paderborn University, winter term 2023/2024, taught by Prof. Dr. Ivan Habernal (https://www.trusthlt.org/). 
His course contents are available at https://github.com/trusthlt/nlp-with-deep-learning-lectures and build up step by step from the basics of computational graphs to Encoder-Decoder-Transformer-Models.

The task for this homework was:
> - Implement Neural Machine Translation
>     - In **pytorch**
>     - Tokenizers: Might be 3rd party (e.g., HuggingFace **SentencePiece** or BPE)
>     - Evaluation: Might be 3rd party (BLEU, ROUGE)
> - Dataset: **WMT 16 en-de** (https://data.statmt.org/wmt16/translation-task/)
>     - Choose a suitable sub-set
> - Evaluation: BLEU (optional: ROUGE)
> - Oracle models
>     - Take reference translation but re-shuffle all tokens ("very bad")
>     - Take reference translation, replace each token with another random token with probability 0.01 ("almost perfect")
> - First model: **RNN encoder-decoder** (LSTM, left-to-right first, then bidirectional) without attention (use teacher forcing, you might experiment with several decoding strategies)
> - Second model: **RNN with attention (Bahdanau et al., 2014; https://arxiv.org/abs/1409.0473)**
> - (Optionally further models: Transformer encoder-decoder)
> 
> Challenges:
> - Get the training/dev/test data right
> - Same tokenization for model/evaluation
> 
> You can use existing snippets of code, some external low-leverl libraries, etc., but need to understand the code (aka learning)
> - Just copy & pasting "https://pytorch.org/tutorials/beginner/translation_transformer.html" might be a bad idea

(Habernal, 2023)

## Installation
- Navigate to the projects base directory where the setup.py is located and run `pip install .` or `pip install -e .` if you want to do development
- If you want to run training scripts you have to use your own weights and biases credentials by including a .env file on the top level (next to the README.md) including the environment variables WANDB_KEY and WANDB_PROJECT.

## Running the code
The entire process is documented in the notebook "summary.ipynb" located at the top level of the project. It includes all code calls, their outputs and wandb diagrams as solutions to the tasks. It also states external calls of scripts including their parameters, e.g. for training the models and running inference, as these were done on a Linux server in order to run on proper GPUs and for multiple days. The implementations of the different models and their training scripts can be found in the training module, evaluation methods in the evaluation module, generation methods in the generation module and preprocessing methods in the data module.

## License
GNU AFFERO GENERAL PUBLIC LICENSE V3
