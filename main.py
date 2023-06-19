import torch
from modules.utils import create_sequence_mask, create_subsequent_mask, count_params
from models.transformers import Transformer

vocab_size = 1000
n_heads = 8
max_length = 512
n_blocks = 6
d_model = 512
d_ff = 4 * d_model
d_k = d_v = d_model // n_heads
p_drop = 0.1

batch_size = 4

x = torch.randint(1, vocab_size, (batch_size, max_length))
x_lengths = torch.randint(1, max_length, (batch_size,))
x_mask = create_sequence_mask(x_lengths, max_length=max_length)

y = torch.randint(1, vocab_size, (batch_size, max_length))
y_lengths = torch.randint(1, max_length, (batch_size,))
y_mask = create_subsequent_mask(max_length)


model = Transformer(
    vocab_size, n_heads, max_length, n_blocks, d_model, d_ff, d_k, d_v, p_drop
)
print(model)
count_params(model)

logits = model(x, x_mask, y, y_mask)
print(logits)
print(logits.size())
