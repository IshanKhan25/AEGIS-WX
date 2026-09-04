from __future__ import annotations
import argparse, time
import numpy as np
from core.config import load_config
from core.logger import get_logger
from data.loaders import load_scenario
from data.validators import assert_valid
from data.synthetic_generator import MODELS
from models.regime_classifier import RuleBasedRegimeClassifier
from models.ram_sta import RAMSTABlender
from models.bias_correction import RollingBiasCorrector
from models.uncertainty import DisagreementUncertainty
from models.fallback import SafetyFallback

class AegisPipeline:
    def __init__(self, config: dict | None=None): self.config=config or load_config(); self.log=get_logger()
    def run(self, seed: int | None=None, regime: str | None=None, mode: str="DEMO") -> dict:
        started=time.perf_counter(); scenario, data_mode=load_scenario(self.config,mode,seed=seed,regime=regime)
        if not hasattr(scenario,"forecasts"): raise RuntimeError("Real data adapter loading is available but normalization into internal schema is pending")
        assert_valid(scenario); detected, probs, features=RuleBasedRegimeClassifier().predict(scenario.forecasts)
        active=regime if regime and regime != "AUTO" else detected
        ram=RAMSTABlender(); outputs={}; threshold=self.config["thresholds"]["uncertainty"]
        for variable in ("precipitation","temperature_2m","wind_speed_10m"):
            weights=ram.weights(scenario.forecasts,variable,active,scenario.leads)
            blended=ram.blend(scenario.forecasts,variable,weights)
            corrector=RollingBiasCorrector().fit(blended,scenario.truth[variable]); corrected=corrector.transform(blended,variable)
            members=np.stack([scenario.forecasts[m][variable] for m in MODELS],axis=1)
            unc,conf=DisagreementUncertainty().calculate(members,weights,corrector.residual_std)
            final, fallback, status=SafetyFallback(threshold).apply(corrected,members,weights,unc)
            outputs[variable]={"weights":weights,"raw_blended":blended,"bias_corrected":corrected,"final":final,"uncertainty":unc,"confidence":conf,"fallback_mask":fallback,"fallback_status":status,"members":members,"exceedance_probability":DisagreementUncertainty().exceedance_probability(members,self.config["thresholds"]["heavy_rain"] if variable=="precipitation" else self.config["thresholds"]["severe_wind"])}
        alerts={"heavy_rain_cells":int((outputs["precipitation"]["final"]>=self.config["thresholds"]["heavy_rain"]).sum()),"high_wind_cells":int((outputs["wind_speed_10m"]["final"]>=self.config["thresholds"]["severe_wind"]).sum()),"heatwave_cells":int((outputs["temperature_2m"]["final"]>=self.config["thresholds"]["heatwave"]).sum())}
        result={"scenario":scenario,"data_mode":data_mode,"detected_regime":detected,"active_regime":active,"regime_probabilities":probs,"regime_features":features,"outputs":outputs,"alerts":alerts,"runtime_seconds":time.perf_counter()-started}
        self.log.info("inference_complete mode=%s regime=%s runtime=%.3fs",data_mode,active,result["runtime_seconds"]); return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--seed",type=int,default=None); parser.add_argument("--regime",default=None); args=parser.parse_args()
    result=AegisPipeline().run(args.seed,args.regime); print(f"{result['data_mode']} | regime={result['active_regime']} | {result['runtime_seconds']:.2f}s")
if __name__ == "__main__": main()
