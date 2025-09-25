from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from src.config import VOCAB_SIZE, Model

def build_midi_model():
    config = GPTNeoXConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=Model.HIDDEN_SIZE,
        num_hidden_layers=Model.N_HIDDEN_LAYERS,
        num_attention_heads=Model.N_ATTENTION_HEADS,
        intermediate_size=Model.INTERMEDIATE_SIZE,
        max_position_embeddings=Model.SEQ_LEN,
        bos_token_id=1,
        eos_token_id=2
    )
    return GPTNeoXForCausalLM(config)