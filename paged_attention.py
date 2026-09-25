import torch

B = 16
D = 512
blocks = 8
seq_len = 40

torch.manual_seed(0)

keys_a = torch.randn([B, D])
keys_b = torch.randn([B, D])

values_a = torch.randn([B, D])
values_b = torch.randn([B, D])

query_a = torch.randn([D])
query_b = torch.randn([D])




def init_cache(blocks):
    cache = {}
    cache["keys"] = torch.zeros([blocks, B, D])
    cache["values"] = torch.zeros([blocks, B, D])





cache = init_cache(blocks)
cache["keys"] = torch.tensor([blocks, B, D])
cache["values"] = torch.tensor([blocks, B, D])


