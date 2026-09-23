import torch

TOKENS_PER_BLOCK = 16
D = 512


@torch.compile
def running_softmax_normalizer(query, keys, values, running_max, running_sum, running_out):
    scores = (query @ keys.T) / (D ** 0.5)
    block_max = scores.max()
    new_max = torch.maximum(running_max, block_max)
    alpha = torch.exp(running_max - new_max)
    probs = torch.exp(scores - new_max)
    running_sum = running_sum * alpha + probs.sum()
    running_out = running_out * alpha + probs @ values
    return running_out, running_sum, new_max


def init_cache(num_blocks):
    cache = {}
    cache["keys"] = torch.zeros(num_blocks, TOKENS_PER_BLOCK, D)
    cache["values"] = torch.zeros(num_blocks, TOKENS_PER_BLOCK, D)
    cache["block_tables"] = {}
    cache["lengths"] = {}
    cache["free_blocks"] = list(range(num_blocks))
    return cache


def append_tokens(cache, seq, new_keys, new_values):
    if seq not in cache["block_tables"]:
        cache["block_tables"][seq] = []
        cache["lengths"][seq] = 0
    token_count = cache["lengths"][seq]
    for i in range(new_keys.shape[0]):
        slot = token_count % TOKENS_PER_BLOCK
        if slot == 0:  # block is full, grab a new one
            block_id = cache["free_blocks"].pop(0)
            cache["block_tables"][seq].append(block_id)
        block_id = cache["block_tables"][seq][-1]
        cache["keys"][block_id][slot] = new_keys[i]
        cache["values"][block_id][slot] = new_values[i]
        token_count += 1
    cache["lengths"][seq] = token_count


def paged_attention(cache, seq, query):
    running_max = torch.tensor(float("-inf"))
    running_sum = torch.tensor(0.0)
    running_out = torch.zeros(D)
    tokens_left = cache["lengths"][seq]
    for block_id in cache["block_tables"][seq]:
        if tokens_left < TOKENS_PER_BLOCK:
            tokens_here = tokens_left
        else:
            tokens_here = TOKENS_PER_BLOCK
        keys = cache["keys"][block_id][:tokens_here]
        values = cache["values"][block_id][:tokens_here]
        running_out, running_sum, running_max = running_softmax_normalizer(
            query, keys, values, running_max, running_sum, running_out
        )
        tokens_left -= tokens_here
    return running_out / running_sum


def dense_attention(keys, values, query):
    scores = (query @ keys.T) / (D ** 0.5)
    probs = torch.softmax(scores, dim=0)
    return probs @ values


torch.manual_seed(0)
seq_len = 40

keys_a = torch.randn(seq_len, D)
values_a = torch.randn(seq_len, D)
keys_b = torch.randn(seq_len, D)
values_b = torch.randn(seq_len, D)
query_a = torch.randn(D)
query_b = torch.randn(D)

cache = init_cache(8)
for i in range(seq_len):
    append_tokens(cache, "seq_a", keys_a[i:i+1], values_a[i:i+1])
    append_tokens(cache, "seq_b", keys_b[i:i+1], values_b[i:i+1])

paged_a = paged_attention(cache, "seq_a", query_a)
paged_b = paged_attention(cache, "seq_b", query_b)
dense_a = dense_attention(keys_a, values_a, query_a)
dense_b = dense_attention(keys_b, values_b, query_b)


def short(vector):
    parts = []
    for x in vector[:4].tolist():
        parts.append(f"{x:+.3f}")
    return "[" + " ".join(parts) + "]"


print("seq_a blocks:", cache["block_tables"]["seq_a"])
print("seq_b blocks:", cache["block_tables"]["seq_b"])
print("paged head:", short(paged_a), short(paged_b))
print("dense head:", short(dense_a), short(dense_b))
diff_a = (paged_a - dense_a).abs().max().item()
diff_b = (paged_b - dense_b).abs().max().item()
print("max difference:", max(diff_a, diff_b))
