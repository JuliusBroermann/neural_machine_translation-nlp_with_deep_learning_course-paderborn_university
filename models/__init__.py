from pathlib import Path
import enum


models_path = Path(__file__).parent

class ModelType(enum.Enum):
    LSTMEncoderDecoder = "lstm_encoder_decoder"
    LSTMBidirectionalEncoderDecoder = "lstm_bidirectional_encoder_decoder"
    
