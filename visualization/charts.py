from __future__ import annotations
def line_comparison(frame, x="lead_time", y="RMSE", color="Model", title="Error by lead time"):
    import plotly.express as px
    return px.line(frame,x=x,y=y,color=color,markers=True,title=title)
def weights_chart(weights, models):
    import plotly.graph_objects as go
    return go.Figure([go.Bar(name=m,y=[float(weights[i])]) for i,m in enumerate(models)]).update_layout(barmode="group",title="Adaptive weights at selected grid point",yaxis_range=[0,1])
