"""
pagedattention.py -- educational reimplementation of vLLM's PagedAttention.
Kwon et al., "Efficient Memory Management for Large Language Model Serving with
PagedAttention", arXiv:2309.06180.

The paper's ideas, each exercised in main():

  1. Paged KV cache    KV memory is cut into fixed-size physical blocks. A
                       per-sequence block table maps logical block -> physical
                       block, so a sequence never needs one contiguous buffer.
  2. PagedAttention    The kernel walks the block table block by block, gathers
                       K/V from scattered physical blocks, and streams the
                       softmax -- it never materializes the full attention
                       matrix and never gathers the cache into a dense buffer.
  3. Online softmax    `running_softmax` is an opaque black box: it carries a
                       running max/sum/output through the loop, which is exactly
                       the FlashAttention-style trick the real kernel uses.
  4. Memory sharing    Sequences sharing a prefix (parallel sampling, beam
                       search, shared system prompts) point at the same physical
                       blocks. Refcounts + copy-on-write keep them independent
                       once they diverge.
  5. Memory waste      Only each sequence's last block is partly empty (internal
                       fragmentation); blocks are allocated just-in-time, so
                       there is no over-reservation or external fragmentation.
  6. Preemption        Blocks can be swapped out to CPU and back in (the paper's
                       other option is to recompute the prefill).

No classes: the cache and its allocator are a plain dict driven by functions.
Shapes: keys/values are [num_blocks, block_size, num_heads, head_dim], and one
query is [num_heads, head_dim]. vLLM uses fancier per-head layouts so the kernel
can vectorize, but this one is conceptually equivalent.
"""

import math
import torch

BLOCK_SIZE = 4
NUM_BLOCKS = 12
NUM_HEADS = 2
HEAD_DIM = 4


# ---------------------------------------------------------------------------
# The black box: streaming softmax over one block of keys/values.
# ---------------------------------------------------------------------------

def running_softmax(query, keys, values, running_max, running_sum, running_out):
    """Opaque kernel: fold one physical block into the running attention output.

    Contract:
      query        [H, D]      one query vector per head
      keys/values  [T, H, D]   T tokens of one block (T <= BLOCK_SIZE)
      running_max  [H]         max score seen so far (-inf before the first block)
      running_sum  [H]         sum of exp(score - running_max) so far
      running_out  [H, D]      unnormalized output numerator so far
    Returns the updated (running_max, running_sum, running_out).

    It hides three details: the 1/sqrt(D) scaling, the online rescaling of the
    previous partial results when the running max grows, and the exp/sum/matmul
    math. Treat it as a fused kernel and do not peek inside from the caller.
    """
    scores = torch.einsum("hd,thd->ht", query, keys) / math.sqrt(query.shape[-1])
    block_max = scores.max(dim=-1).values
    new_max = torch.maximum(running_max, block_max)
    alpha = torch.exp(running_max - new_max)
    probs = torch.exp(scores - new_max.unsqueeze(-1))
    new_sum = running_sum * alpha + probs.sum(dim=-1)
    new_out = running_out * alpha.unsqueeze(-1) + torch.einsum("ht,thd->hd", probs, values)
    return new_max, new_sum, new_out


# ---------------------------------------------------------------------------
# KV cache manager: block allocator, block tables, reference counts.
# ---------------------------------------------------------------------------

def new_cache(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE):
    return {
        "k": torch.zeros(num_blocks, block_size, NUM_HEADS, HEAD_DIM),
        "v": torch.zeros(num_blocks, block_size, NUM_HEADS, HEAD_DIM),
        "block_size": block_size,
        "free": list(range(num_blocks)),  # global free block pool
        "tables": {},                     # seq -> [physical block ids], logical order
        "lengths": {},                    # seq -> number of valid tokens
        "refcount": [0] * num_blocks,     # sharers per physical block
    }


def new_sequence(cache, seq):
    cache["tables"][seq] = []
    cache["lengths"][seq] = 0


def alloc_block(cache):
    if not cache["free"]:
        raise RuntimeError("out of KV blocks -- vLLM would preempt a sequence here")
    phys = cache["free"].pop()
    cache["refcount"][phys] = 1
    return phys


def free_block(cache, phys):
    cache["refcount"][phys] -= 1
    if cache["refcount"][phys] == 0:
        cache["free"].append(phys)


def free_sequence(cache, seq):
    for phys in cache["tables"].pop(seq):
        free_block(cache, phys)
    cache["lengths"].pop(seq)


def append(cache, seq, key, value):
    """Append one token's K/V; grab a new block every block_size tokens."""
    pos = cache["lengths"][seq]
    slot = pos % cache["block_size"]
    if slot == 0:
        cache["tables"][seq].append(alloc_block(cache))
    logical = pos // cache["block_size"]
    phys = cache["tables"][seq][logical]
    if cache["refcount"][phys] > 1:
        phys = copy_on_write(cache, seq, logical)
    cache["k"][phys, slot] = key
    cache["v"][phys, slot] = value
    cache["lengths"][seq] = pos + 1


def copy_on_write(cache, seq, logical):
    """Shared block: clone it somewhere else and leave the other sharers alone."""
    old = cache["tables"][seq][logical]
    new = alloc_block(cache)
    cache["k"][new] = cache["k"][old]
    cache["v"][new] = cache["v"][old]
    cache["tables"][seq][logical] = new
    free_block(cache, old)
    return new


def fork(cache, src, dst):
    """Share every block of src with dst (parallel sampling / beam search)."""
    cache["tables"][dst] = list(cache["tables"][src])
    cache["lengths"][dst] = cache["lengths"][src]
    for phys in cache["tables"][dst]:
        cache["refcount"][phys] += 1


def cache_stats(cache):
    """Physical memory use; shared blocks are counted once."""
    used = len(cache["refcount"]) - len(cache["free"])
    slots = used * cache["block_size"]
    occupied = {}
    for seq, table in cache["tables"].items():
        length = cache["lengths"][seq]
        for logical, phys in enumerate(table):
            tokens = min(cache["block_size"], length - logical * cache["block_size"])
            occupied[phys] = max(occupied.get(phys, 0), tokens)
    return used, slots, sum(occupied.values())


# ---------------------------------------------------------------------------
# The paged attention kernel (emulated) + a dense reference to check it.
# ---------------------------------------------------------------------------

def paged_attention(cache, seq, query):
    n_heads, head_dim = query.shape
    running_max = torch.full((n_heads,), float("-inf"))
    running_sum = torch.zeros(n_heads)
    running_out = torch.zeros(n_heads, head_dim)
    length = cache["lengths"][seq]
    for logical, phys in enumerate(cache["tables"][seq]):
        start = logical * cache["block_size"]
        tokens = min(cache["block_size"], length - start)  # last block may be partial
        running_max, running_sum, running_out = running_softmax(
            query,
            cache["k"][phys, :tokens],
            cache["v"][phys, :tokens],
            running_max,
            running_sum,
            running_out,
        )
    return running_out / running_sum.unsqueeze(-1)


def dense_attention(query, keys, values):
    scores = torch.einsum("hd,thd->ht", query, keys) / math.sqrt(query.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", probs, values)


# ---------------------------------------------------------------------------
# Preemption: swap blocks out to a CPU pool and back in.
# ---------------------------------------------------------------------------

def swap_out(cache, seq, cpu_pool):
    """Copy whole blocks to CPU, then release the GPU blocks."""
    blocks = cache["tables"][seq]
    cpu_pool[seq] = {
        "k": torch.stack([cache["k"][p].clone() for p in blocks]),
        "v": torch.stack([cache["v"][p].clone() for p in blocks]),
        "length": cache["lengths"][seq],
    }
    free_sequence(cache, seq)


def swap_in(cache, seq, cpu_pool):
    """Restore into freshly allocated blocks -- they can land anywhere."""
    data = cpu_pool.pop(seq)
    new_sequence(cache, seq)
    cache["lengths"][seq] = data["length"]
    for b in range(data["k"].shape[0]):
        phys = alloc_block(cache)
        cache["tables"][seq].append(phys)
        cache["k"][phys] = data["k"][b]
        cache["v"][phys] = data["v"][b]


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def fmt(vector):
    body = " ".join(f"{x:+.3f}" for x in vector.flatten()[:HEAD_DIM])
    return f"[{body} ...]"


def main():
    torch.manual_seed(0)
    cache = new_cache()
    query = torch.randn(NUM_HEADS, HEAD_DIM)

    print("=" * 70)
    print("1. paged KV cache: logical blocks -> physical blocks")
    print("=" * 70)
    keys_a = torch.randn(10, NUM_HEADS, HEAD_DIM)
    values_a = torch.randn(10, NUM_HEADS, HEAD_DIM)
    new_sequence(cache, "A")
    for t in range(10):
        append(cache, "A", keys_a[t], values_a[t])
    print(f"block size    : {BLOCK_SIZE} tokens")
    print(f"sequence A    : {cache['lengths']['A']} tokens")
    print(f"block table A : {cache['tables']['A']}")
    print(f"                last block {10 % BLOCK_SIZE}/{BLOCK_SIZE} used -> internal fragmentation")
    got = paged_attention(cache, "A", query)
    want = dense_attention(query, keys_a, values_a)
    print(f"paged output  : {fmt(got)}")
    print(f"dense output  : {fmt(want)}")
    assert torch.allclose(got, want, atol=1e-5)
    print("match: PagedAttention == full softmax attention")

    print()
    print("=" * 70)
    print("2. free + reuse: the block table hides the scattered layout")
    print("=" * 70)
    free_sequence(cache, "A")
    keys_b = torch.randn(8, NUM_HEADS, HEAD_DIM)
    values_b = torch.randn(8, NUM_HEADS, HEAD_DIM)
    new_sequence(cache, "B")
    for t in range(4):
        append(cache, "B", keys_b[t], values_b[t])
    new_sequence(cache, "C")
    for t in range(4):
        append(cache, "C", keys_b[t], values_b[t])
    for t in range(4, 8):
        append(cache, "B", keys_b[t], values_b[t])
    print(f"B block table : {cache['tables']['B']}  <- physical ids not consecutive")
    got = paged_attention(cache, "B", query)
    want = dense_attention(query, keys_b, values_b)
    assert torch.allclose(got, want, atol=1e-5)
    print("match: attention works across non-contiguous blocks")

    print()
    print("=" * 70)
    print("3. memory sharing + copy-on-write (parallel sampling)")
    print("=" * 70)
    keys_d = torch.randn(6, NUM_HEADS, HEAD_DIM)
    values_d = torch.randn(6, NUM_HEADS, HEAD_DIM)
    new_sequence(cache, "D")
    for t in range(6):
        append(cache, "D", keys_d[t], values_d[t])
    shared = cache["tables"]["D"][-1]  # partly filled, will be shared
    fork(cache, "D", "D2")
    print(f"fork D -> D2")
    print(f"D  table      : {cache['tables']['D']}")
    print(f"D2 table      : {cache['tables']['D2']}  (same physical blocks)")
    print(f"refcount of shared block {shared}: {cache['refcount'][shared]}")
    key_new = torch.randn(NUM_HEADS, HEAD_DIM)
    value_new = torch.randn(NUM_HEADS, HEAD_DIM)
    append(cache, "D2", key_new, value_new)  # diverging token triggers copy-on-write
    print(f"append to D2 (shared block was {shared}, refcount back to {cache['refcount'][shared]})")
    print(f"D  table      : {cache['tables']['D']}  (untouched)")
    print(f"D2 table      : {cache['tables']['D2']}  (tail block copied)")
    assert torch.allclose(paged_attention(cache, "D", query), dense_attention(query, keys_d, values_d), atol=1e-5)
    keys_d2 = torch.cat([keys_d, key_new[None]])
    values_d2 = torch.cat([values_d, value_new[None]])
    assert torch.allclose(paged_attention(cache, "D2", query), dense_attention(query, keys_d2, values_d2), atol=1e-5)
    print("match: both branches see their own KV, sharing stayed correct")

    print()
    print("=" * 70)
    print("4. memory waste: only partial last blocks")
    print("=" * 70)
    used, slots, occupied = cache_stats(cache)
    print(f"physical blocks in use : {used}/{NUM_BLOCKS}")
    print(f"K/V slots reserved     : {slots} (shared blocks counted once)")
    print(f"K/V slots occupied     : {occupied}")
    print(f"wasted slots           : {slots - occupied} (each sequence's last block only)")
    print("a contiguous allocator would reserve max_seq_len per sequence instead")

    print()
    print("=" * 70)
    print("5. preemption: swap blocks to CPU and swap back in")
    print("=" * 70)
    cpu_pool = {}
    before = paged_attention(cache, "D", query)
    free_before = len(cache["free"])
    swap_out(cache, "D", cpu_pool)
    print(f"swap D out    : free blocks {free_before} -> {len(cache['free'])}, 'D' off the GPU")
    swap_in(cache, "D", cpu_pool)
    after = paged_attention(cache, "D", query)
    print(f"swap D in     : block table {cache['tables']['D']} (may differ from before)")
    assert torch.allclose(before, after, atol=1e-6)
    print("match: output identical after the round trip")


if __name__ == "__main__":
    main()
