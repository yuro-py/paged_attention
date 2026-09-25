"""
Minimal PagedAttention (Kwon et al., arXiv:2309.06180) -- paged KV cache +
streaming attention + copy-on-write sharing + swap-based preemption, all in
a handful of dense functions instead of many small ones.
"""

import math
import torch

BLOCK, NUM_BLOCKS, HEADS, DIM = 4, 12, 2, 4


def new_cache(num_blocks=NUM_BLOCKS, block=BLOCK):
    # k/v: [num_blocks, block, heads, dim]. free = unused physical block ids.
    # tables[seq] = [phys_ids...] (logical order); refcount for COW sharing.
    return dict(k=torch.zeros(num_blocks, block, HEADS, DIM),
                v=torch.zeros(num_blocks, block, HEADS, DIM),
                block=block, free=list(range(num_blocks)),
                tables={}, lengths={}, refcount=[0] * num_blocks)


def append(cache, seq, key, value):
    """Write one token's K/V, allocating/COW-ing physical blocks as needed."""
    cache["tables"].setdefault(seq, [])
    pos = cache["lengths"].get(seq, 0)
    logical, slot = divmod(pos, cache["block"])

    if slot == 0:  # need a fresh block for this sequence
        if not cache["free"]:
            raise RuntimeError("out of KV blocks -- vLLM would preempt here")
        phys = cache["free"].pop()
        cache["refcount"][phys] = 1
        cache["tables"][seq].append(phys)

    phys = cache["tables"][seq][logical]
    if cache["refcount"][phys] > 1:  # shared block, about to diverge -> copy-on-write
        new_phys = cache["free"].pop()
        cache["k"][new_phys], cache["v"][new_phys] = cache["k"][phys].clone(), cache["v"][phys].clone()
        cache["refcount"][phys] -= 1
        cache["refcount"][new_phys] = 1
        cache["tables"][seq][logical] = phys = new_phys

    cache["k"][phys, slot], cache["v"][phys, slot] = key, value
    cache["lengths"][seq] = pos + 1


def free_sequence(cache, seq):
    """Release a sequence's blocks, returning any that drop to refcount 0."""
    for phys in cache["tables"].pop(seq):
        cache["refcount"][phys] -= 1
        if cache["refcount"][phys] == 0:
            cache["free"].append(phys)
    cache["lengths"].pop(seq)


def fork(cache, src, dst):
    """Parallel sampling / beam search: dst shares every block of src."""
    cache["tables"][dst] = list(cache["tables"][src])
    cache["lengths"][dst] = cache["lengths"][src]
    for phys in cache["tables"][dst]:
        cache["refcount"][phys] += 1


def paged_attention(cache, seq, query):
    """Stream softmax over a sequence's blocks -- no dense buffer, ever."""
    length, block = cache["lengths"][seq], cache["block"]
    run_max, run_sum = torch.full((HEADS,), -float("inf")), torch.zeros(HEADS)
    run_out = torch.zeros(HEADS, DIM)

    for logical, phys in enumerate(cache["tables"][seq]):
        n = min(block, length - logical * block)  # last block may be partial
        k, v = cache["k"][phys, :n], cache["v"][phys, :n]

        scores = torch.einsum("hd,thd->ht", query, k) / math.sqrt(DIM)
        new_max = torch.maximum(run_max, scores.max(-1).values)
        alpha = torch.exp(run_max - new_max)
        probs = torch.exp(scores - new_max.unsqueeze(-1))
        run_sum = run_sum * alpha + probs.sum(-1)
        run_out = run_out * alpha.unsqueeze(-1) + torch.einsum("ht,thd->hd", probs, v)
        run_max = new_max

    return run_out / run_sum.unsqueeze(-1)


def dense_attention(query, keys, values):
    """Reference (unpaged) attention, for correctness checks."""
    scores = torch.einsum("hd,thd->ht", query, keys) / math.sqrt(DIM)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", probs, values)


def swap(cache, seq, cpu_pool, direction):
    """Preemption: 'out' evicts a sequence's blocks to CPU, 'in' restores them
    into freshly allocated (possibly different) physical blocks."""
    if direction == "out":
        blocks = cache["tables"][seq]
        cpu_pool[seq] = dict(k=torch.stack([cache["k"][p].clone() for p in blocks]),
                              v=torch.stack([cache["v"][p].clone() for p in blocks]),
                              length=cache["lengths"][seq])
        free_sequence(cache, seq)
    else:  # "in"
        data = cpu_pool.pop(seq)
        cache["tables"][seq], cache["lengths"][seq] = [], data["length"]
        for kb, vb in zip(data["k"], data["v"]):
            phys = cache["free"].pop()
            cache["refcount"][phys] = 1
            cache["tables"][seq].append(phys)
            cache["k"][phys], cache["v"][phys] = kb, vb


def cache_stats(cache):
    """Physical usage; shared blocks counted once, only last block per
    sequence is partly wasted."""
    used = len(cache["refcount"]) - len(cache["free"])
    occupied = {}
    for seq, table in cache["tables"].items():
        length = cache["lengths"][seq]
        for logical, phys in enumerate(table):
            n = min(cache["block"], length - logical * cache["block"])
            occupied[phys] = max(occupied.get(phys, 0), n)
    return used, used * cache["block"], sum(occupied.values())


def main():
    torch.manual_seed(0)
    cache = new_cache()
    query = torch.randn(HEADS, DIM)

    # paged vs dense equivalence, including non-contiguous block reuse
    keys, values = torch.randn(10, HEADS, DIM), torch.randn(10, HEADS, DIM)
    for t in range(10):
        append(cache, "A", keys[t], values[t])
    assert torch.allclose(paged_attention(cache, "A", query), dense_attention(query, keys, values), atol=1e-5)
    print(f"A: table={cache['tables']['A']}  paged==dense OK")

    # copy-on-write: fork a sequence, diverge one branch, both stay correct
    fork(cache, "A", "A2")
    k2, v2 = torch.randn(HEADS, DIM), torch.randn(HEADS, DIM)
    append(cache, "A2", k2, v2)
    keys2, values2 = torch.cat([keys, k2[None]]), torch.cat([values, v2[None]])
    assert torch.allclose(paged_attention(cache, "A", query), dense_attention(query, keys, values), atol=1e-5)
    assert torch.allclose(paged_attention(cache, "A2", query), dense_attention(query, keys2, values2), atol=1e-5)
    print(f"fork+COW: A={cache['tables']['A']} A2={cache['tables']['A2']}  both correct")

    # memory stats: only the tail block per sequence is ever partly wasted
    used, slots, occ = cache_stats(cache)
    print(f"blocks used={used}/{NUM_BLOCKS}  slots={slots}  occupied={occ}  wasted={slots - occ}")

    # preemption: swap out to CPU and back, output must round-trip exactly
    pool, before = {}, paged_attention(cache, "A2", query)
    swap(cache, "A2", pool, "out")
    swap(cache, "A2", pool, "in")
    assert torch.allclose(before, paged_attention(cache, "A2", query), atol=1e-6)
    print(f"swap round-trip: table={cache['tables']['A2']}  output unchanged")


if __name__ == "__main__":
    main()
