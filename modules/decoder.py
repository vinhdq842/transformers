from torch import nn

class Decoder(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, max_length: int):
        self.embedding = nn.Embedding(vocab_size, d_model)
        