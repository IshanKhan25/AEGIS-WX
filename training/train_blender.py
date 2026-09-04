from __future__ import annotations
import json
from pathlib import Path
from core.config import load_config
from inference.pipeline import AegisPipeline
from training.dataset import generate_hindcast_seeds, chronological_split
from verification.benchmarks import benchmark
def main():
    config=load_config(); seeds=generate_hindcast_seeds(20,config["demo"]["random_seed"]); train,val,test=chronological_split(seeds)
    # The deterministic RAM-STA parameters are a transparent baseline; benchmark actual held-out chronological scenarios.
    metrics=[]
    for seed in test: metrics.append(benchmark(AegisPipeline(config).run(int(seed)),"precipitation").query("Model == 'AEGIS-WX'").iloc[0].to_dict())
    Path("checkpoints").mkdir(exist_ok=True); Path("checkpoints/blender_demo_metrics.json").write_text(json.dumps({"split":{"train":len(train),"validation":len(val),"test":len(test)},"test_metrics":metrics},indent=2)); print("Saved checkpoints/blender_demo_metrics.json; prototype benchmark uses synthetic hindcast data.")
if __name__ == "__main__": main()
