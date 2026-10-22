"""
Reproduce the results table in the README.

    python benchmarks/run_benchmarks.py

Setup: 100,000 Zipf(0.9) requests over 10,000 objects, 8 edge nodes, 100 objects
per node (the edge holds 8% of the catalog), averaged over seeds 1-3.
Takes about a minute on a laptop.
"""

import contextlib
import hashlib
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import CDNSimulator, SimulationConfig  # noqa: E402
from src.trace_generator import TraceGenerator  # noqa: E402
from src.utils.hashing import ConsistentHasher  # noqa: E402

REQUESTS, CATALOG, SKEW, NODES, CAPACITY = 100_000, 10_000, 0.9, 8, 100
SEEDS = (1, 2, 3)


def run(policy, seed, capacity=CAPACITY, ttl=300.0):
    """Run one simulation quietly and return (aggregate metrics, node metrics, seconds)."""
    trace = TraceGenerator(seed=seed).generate_zipf_trace(REQUESTS, CATALOG, skew=SKEW)
    sim = CDNSimulator(SimulationConfig(num_nodes=NODES, cache_capacity=capacity,
                                        policy=policy, ttl_seconds=ttl))
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = sim.run(trace)
    return result.aggregate_metrics, result.node_metrics, time.perf_counter() - start


def mean(xs):
    return float(np.mean(xs))


def main():
    print(f"{REQUESTS:,} Zipf({SKEW}) requests, {CATALOG:,} objects, {NODES} nodes x "
          f"{CAPACITY} objects (edge holds {NODES * CAPACITY / CATALOG:.0%} of catalog), "
          f"seeds {SEEDS}\n")

    print("| Policy | Edge hit ratio | Bytes served from edge | Simulator speed |")
    print("|---|---|---|---|")
    for policy in ("LRU", "LFU", "TTL"):
        runs = [run(policy, s) for s in SEEDS]
        hit = mean([m["avg_hit_ratio"] for m, _, _ in runs])
        lo = min(m["avg_hit_ratio"] for m, _, _ in runs)
        hi = max(m["avg_hit_ratio"] for m, _, _ in runs)
        bw = mean([m["bandwidth_saved_pct"] for m, _, _ in runs])
        rps = mean([REQUESTS / t for _, _, t in runs])
        print(f"| {policy} | {hit:.1%} ({lo:.1%} to {hi:.1%}) | {bw:.1f}% | {rps:,.0f} req/s |")

    print("\n| LRU, objects per node | 25 | 50 | 100 | 200 | 400 |")
    print("|---|---|---|---|---|---|")
    hits = [run("LRU", 1, capacity=c)[0]["avg_hit_ratio"] for c in (25, 50, 100, 200, 400)]
    print("| Edge hit ratio | " + " | ".join(f"{h:.1%}" for h in hits) + " |")

    print("\n| TTL (seconds) | 1 | 5 | 10 | 20 | 60 |")
    print("|---|---|---|---|---|---|")
    row_hit, row_exp = [], []
    for ttl in (1, 5, 10, 20, 60):
        runs = [run("TTL", s, ttl=float(ttl)) for s in SEEDS]
        row_hit.append(mean([m["avg_hit_ratio"] for m, _, _ in runs]))
        expired = mean([sum(n.get("expired_count", 0) for n in nodes) for _, nodes, _ in runs])
        evicted = mean([sum(n.get("evictions", 0) for n in nodes) for _, nodes, _ in runs])
        row_exp.append(expired / (expired + evicted) if expired + evicted else 0.0)
    print("| Edge hit ratio | " + " | ".join(f"{h:.1%}" for h in row_hit) + " |")
    print("| Removals caused by expiry | " + " | ".join(f"{e:.0%}" for e in row_exp) + " |")

    class Node:
        def __init__(self, node_id):
            self.node_id = node_id

    keys = [f"obj_{i:06d}" for i in range(CATALOG)]
    nodes = [Node(i) for i in range(NODES)]
    ring_before = ConsistentHasher(nodes, virtual_nodes_per_node=150)
    ring_after = ConsistentHasher(nodes[:-1], virtual_nodes_per_node=150)
    moved_ring = mean([ring_before.get_node(k).node_id != ring_after.get_node(k).node_id for k in keys])

    def modulo(key, n):
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % n

    moved_mod = mean([modulo(k, NODES) != modulo(k, NODES - 1) for k in keys])

    print(f"\nOne of {NODES} nodes removed, objects that move to a different node:")
    print(f"  consistent hashing (150 virtual nodes): {moved_ring:.1%}   "
          f"modulo hashing: {moved_mod:.1%}   ideal: {1 / NODES:.1%}")

    print("Busiest node's load relative to the average:")
    for v in (1, 150):
        ring = ConsistentHasher(nodes, virtual_nodes_per_node=v)
        counts = np.bincount([ring.get_node(k).node_id for k in keys], minlength=NODES)
        print(f"  {v:>3} virtual node(s) per server: {counts.max() / counts.mean():.2f}x")


if __name__ == "__main__":
    main()
