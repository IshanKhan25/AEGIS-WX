from __future__ import annotations
import numpy as np
def heatmap(field, latitude, longitude, title, colorscale="Viridis"):
    import plotly.graph_objects as go
    return go.Figure(go.Heatmap(z=np.asarray(field),x=longitude,y=latitude,colorscale=colorscale,colorbar={"title":title})).update_layout(title=title,xaxis_title="Longitude (°E)",yaxis_title="Latitude (°N)",height=440,margin=dict(l=20,r=20,t=45,b=30))
