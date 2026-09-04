from __future__ import annotations
from core.config import load_config
from inference.pipeline import AegisPipeline
def main():
    for regime in ("NORMAL","MONSOON","CYCLONE","HEATWAVE","WESTERN_DISTURBANCE"):
        r=AegisPipeline(load_config()).run(regime=regime); print(f"scenario={regime}, detected={r['detected_regime']}")
    print("Rule-based regime baseline evaluated; replace MLRegimeClassifier with a trained model when real labels exist.")
if __name__ == "__main__": main()
