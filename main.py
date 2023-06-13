import torch
from modules.encoder import Encoder

vocab_size = 1000
n_head = 8
max_length = 512
n_block = 6
d_model = 512
d_ff = 2048
d_k = d_v = d_model // n_head

encoder = Encoder(vocab_size, n_head, max_length, n_block, d_model, d_ff, d_k, d_v)
print(encoder)


batch_size = 4
max_seq_length = 20

x = (torch.rand(batch_size, max_seq_length) * vocab_size).to(torch.int64)
x_mask = torch.ones(batch_size, max_seq_length)
print(x)
print(x_mask)

print(encoder(x, x_mask))
