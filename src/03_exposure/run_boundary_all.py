#!/usr/bin/env python3
"""
Module: run_boundary_all.py
Purpose: Orchestrate parallel execution of boundary distance calculations.

This module runs boundary_chunk_dist.py in parallel across multiple chunks
using a thread pool, then merges results using boundary_merge.py. Designed
for efficient processing of large parcel datasets.

Environment Variables
---------------------
BOUNDARY_CHUNKS : int (default 64)
    Number of chunks to split parcels into.
BOUNDARY_WORKERS : int (default 4)
    Number of parallel workers.

Functions
---------
ensure_env : Verify virtual environment is activated
run_chunk : Execute a single chunk processing subprocess
main : Orchestrate parallel execution and merge
"""
from __future__ import annotations
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def run_chunk(idx: int, tot: int) -> int:
    """
    Execute a single chunk processing subprocess.

    Parameters
    ----------
    idx : int
        Chunk index (0-based).
    tot : int
        Total number of chunks.

    Returns
    -------
    int
        Subprocess return code.
    """
    cmd = [sys.executable, 'src/03_exposure/boundary_chunk_dist.py', '--chunk-index', str(idx), '--chunk-total', str(tot)]
    print('Running', ' '.join(cmd), flush=True)
    res = subprocess.run(cmd)
    return res.returncode

def main():
    """
    Orchestrate parallel chunk execution and merge.

    Launches chunk workers in parallel using ThreadPoolExecutor, waits for
    completion, then runs merge to consolidate results. Exits with error
    if any chunks fail.
    """
    ensure_env()
    tot = int(os.getenv('BOUNDARY_CHUNKS', '64'))
    workers = int(os.getenv('BOUNDARY_WORKERS', '4'))
    fails = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_chunk, i, tot): i for i in range(tot)}
        for fut in as_completed(futures):
            rc = fut.result()
            if rc != 0:
                fails += 1
    if fails:
        print(f'{fails} chunks failed')
        sys.exit(1)
    # Merge
    rc = subprocess.run([sys.executable, 'src/03_exposure/boundary_merge.py']).returncode
    sys.exit(rc)

if __name__ == '__main__':
    main()

