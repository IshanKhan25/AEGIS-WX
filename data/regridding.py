from __future__ import annotations
import numpy as np

class Regridder:
    """Dependency-free regular-grid regridding. xESMF can replace this interface in production."""
    def __init__(self, source_lat, source_lon, target_lat, target_lon, method: str = "linear"):
        if method not in {"linear", "nearest", "conservative"}: raise ValueError("method must be linear, nearest or conservative")
        self.sl, self.so, self.tl, self.to, self.method = map(np.asarray,(source_lat,source_lon,target_lat,target_lon)) + (method,) if False else (np.asarray(source_lat),np.asarray(source_lon),np.asarray(target_lat),np.asarray(target_lon),method)
    def regrid(self, field: np.ndarray) -> np.ndarray:
        """Bilinear sequential interpolation; conservative is a documented linear fallback."""
        if field.shape[-2:] != (len(self.sl), len(self.so)): raise ValueError("field grid does not match source coordinates")
        nearest = self.method == "nearest"
        # Interpolate longitude then latitude for every leading dimension.
        flat = field.reshape((-1, len(self.sl), len(self.so)))
        out = np.empty((flat.shape[0], len(self.tl), len(self.to)))
        for n, item in enumerate(flat):
            if nearest:
                ix = np.abs(self.so[None,:]-self.to[:,None]).argmin(1); iy = np.abs(self.sl[None,:]-self.tl[:,None]).argmin(1)
                out[n] = item[np.ix_(iy,ix)]
            else:
                lon_done = np.array([np.interp(self.to,self.so,row) for row in item])
                out[n] = np.array([np.interp(self.tl,self.sl,lon_done[:,j]) for j in range(len(self.to))]).T
        return out.reshape(field.shape[:-2] + (len(self.tl),len(self.to)))
