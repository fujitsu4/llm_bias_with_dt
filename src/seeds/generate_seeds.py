"""
generate_seeds.py
Author : Zakaria JOUILIL

Description :
    Generates a timestamped seeds_list.txt containing a sorted list of unique random seeds.

Input parameters :
    --count: Number of unique seeds to generate. Default : 30
    --low: Minimum seed value (inclusive). Default : 1
    --high: Maximum seed value (inclusive). Default : 10**9

Output example (outputs/attention/seeds_list.txt):
    2025-12-02 10:24:07
    SEEDS = [1837291, 52821, 934820, ...]

Usage example :
    !python src.seeds.generate_seeds --count 30 --low 1 -- high 10000 \
        --output outputs/attention/seeds_list.txt
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

def generate_unique_seeds(count: int, low: int, high: int):
    if count > (high - low + 1):
        raise ValueError("Range too small for requested unique seeds")
    seeds = set()
    while len(seeds) < count:
        seeds.add(random.randint(low, high))
    return sorted(seeds)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=30, help="Number of unique seeds to generate")
    p.add_argument("--low", type=int, default=1, help="Minimum seed value (inclusive)")
    p.add_argument("--high", type=int, default=10**9, help="Maximum seed value (inclusive)")
    p.add_argument("--output", type=str, default="outputs/attention/seeds_list.txt", help="Output file path")
    args = p.parse_args()

    seeds = generate_unique_seeds(args.count, args.low, args.high)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{timestamp}\n")
        f.write("SEEDS = [")
        f.write(", ".join(str(s) for s in seeds))
        f.write("]\n")

    print(f"[OK] Wrote {len(seeds)} seeds to {out_path.resolve()}")

if __name__ == "__main__":
    main()