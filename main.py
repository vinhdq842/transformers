import torch

from models import Transformer
from utils.helpers import count_params, create_pad_mask, create_subsequent_mask

vocab_size = 100
n_heads = 4
max_length = 10
n_blocks = 3
d_model = 64
d_ff = 4 * d_model
d_k = d_v = d_model // n_heads
p_drop = 0.1

batch_size = 2

x_lengths = torch.randint(1, max_length, (batch_size,))
x_mask = create_pad_mask(x_lengths)
x = torch.randint(1, vocab_size, (batch_size, x_lengths.max()))

y_lengths = torch.randint(1, max_length, (batch_size,))
y_mask = create_subsequent_mask(y_lengths, pad_mask=create_pad_mask(y_lengths))
y = torch.randint(1, vocab_size, (batch_size, y_lengths.max()))


model = Transformer(
    vocab_size, n_heads, max_length, n_blocks, d_model, d_ff, d_k, d_v, p_drop
)

print(model)
count_params(model)

print(x_lengths, x.shape, x_mask.shape, y_lengths, y.shape, y_mask.shape)
print(x_mask == 0, y_mask == 0, sep="\n")
# logits = model(x, x_mask, y, y_mask)
# print(logits)
# print(logits.size())
