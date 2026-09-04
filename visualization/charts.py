from __future__ import annotations


def line_comparison(frame, x="lead_time", y="RMSE", color="Model", title="Error by lead time"):
    import numpy as np
    import pandas as pd
    import plotly.express as px

    clean = frame.copy()
    clean[y] = pd.to_numeric(clean[y], errors="coerce")
    clean.loc[~np.isfinite(clean[y]), y] = np.nan
    if clean[y].notna().sum() == 0:
        import plotly.graph_objects as go
        return go.Figure().update_layout(title=title, annotations=[{"text": "N/A — metric undefined for this demo case", "showarrow": False}])
    return px.line(clean, x=x, y=y, color=color, markers=True, title=title)
def weights_chart(weights, models):
    import plotly.graph_objects as go
    return go.Figure([go.Bar(name=m,y=[float(weights[i])]) for i,m in enumerate(models)]).update_layout(barmode="group",title="Adaptive weights at selected grid point",yaxis_range=[0,1])
