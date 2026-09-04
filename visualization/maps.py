from __future__ import annotations
import numpy as np
def heatmap(field, latitude, longitude, title, colorscale="Viridis", unit="", zmin=None, zmax=None, height=360):
    import plotly.graph_objects as go
    suffix = f" {unit}" if unit else ""
    trace = go.Heatmap(
        z=np.asarray(field), x=longitude, y=latitude, colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar={"title": unit},
        hovertemplate=f"Latitude: %{{y:.1f}}°N<br>Longitude: %{{x:.1f}}°E<br>{title}: %{{z:.2f}}{suffix}<extra></extra>",
    )
    return go.Figure(trace).update_layout(title=title,xaxis_title="Longitude (°E)",yaxis_title="Latitude (°N)",height=height,margin=dict(l=20,r=20,t=45,b=30),paper_bgcolor="#0b1c2d",plot_bgcolor="#0b1c2d",font={"color":"#eaf3fb"})
