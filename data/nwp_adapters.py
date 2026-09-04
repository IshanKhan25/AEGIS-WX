from __future__ import annotations
from pathlib import Path
from .synthetic_generator import SyntheticNWPGenerator

class BaseAdapter:
    """Common real-data contract: load, validate, normalize, and regrid a source."""
    model_name = "NWP"
    def load_forecast(self, path: str):
        p = Path(path)
        if not p.exists(): raise FileNotFoundError(p)
        try:
            import xarray as xr
            return xr.open_dataset(p, engine="cfgrib" if p.suffix.lower() in {".grib", ".grib2"} else None)
        except ImportError as exc: raise RuntimeError("Install xarray and cfgrib/eccodes for real-data mode") from exc
    def validate(self, dataset):
        if not getattr(dataset, "data_vars", None): raise ValueError("Dataset contains no forecast variables")
        return dataset
    def normalize(self, dataset): return dataset
    def regrid(self, dataset, target_grid): return dataset.interp(latitude=target_grid[0], longitude=target_grid[1])
class GFSAdapter(BaseAdapter): model_name = "GFS"
class ECMWFAdapter(BaseAdapter): model_name = "ECMWF"
class NCUMAdapter(BaseAdapter): model_name = "NCUM"
class DemoAdapter:
    def __init__(self, config): self.generator = SyntheticNWPGenerator(config)
    def load_forecast(self, **kwargs): return self.generator.generate(**kwargs)
