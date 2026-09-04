from __future__ import annotations
import numpy as np
def chronological_split(samples, train=.70, validation=.15):
    n=len(samples); a=int(n*train); b=int(n*(train+validation)); return samples[:a],samples[a:b],samples[b:]
def generate_hindcast_seeds(n=30, first_seed=26081): return np.arange(first_seed,first_seed+n)
