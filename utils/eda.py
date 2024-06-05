import os
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set default template:
pio.templates.default = "plotly_dark"

def load_sample(size=10_000, cache=True):
    # Check if sample file exists:
    if (not os.path.exists(f"data/sample_{size}.csv")) or (not cache):
        print("Creating sample file...")
        df = pd.read_csv('data/SummaryIndividuals.csv')
        sample = df.sample(size)
        sample.to_csv(f"data/sample_{size}.csv", index=False)
    else:
        print("Sample file found. Loading sample file...")
        sample = pd.read_csv(f"data/sample_{size}.csv")
    return sample

def ridgeline(df, quantVars=None):
    if quantVars is None:
        quantVars = df.select_dtypes(include='number').columns
    colors = px.colors.qualitative.Plotly
    # Create subplots for each quantitative variable:
    fig = go.Figure()
    for i, q in enumerate(quantVars):
        fig.add_trace(go.Violin(
            x=df[q],
            opacity=0.6,
            name=q,
            side='positive',
            hoverinfo='text',
            text=df[q],
            marker=dict(size=2, color=colors[i]),
            meanline_visible=True,
            hoveron='points',
            ))
        # Add annotation at median:
        fig.add_annotation(
            x=df[q].median(),
            y=i,
            text=f"{q}<br>{df[q].median():0.0f}",
            yshift=20,
            xshift=50,
            align='left',
            showarrow=False,
            font=dict(color=colors[i]),
        )
    fig.update_layout(
        height=300, width=400,
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
        violingap=0, violinmode='overlay'
    )
    fig.update_traces(
        orientation='h', side='positive',
        width=3
        )
    fig.update_yaxes(
        showticklabels=False
    )
    return fig

def scatter_3d(df, quantVars=None, x=None, y=None, z=None, color=None):
    if quantVars is None:
        quantVars = df.select_dtypes(include='number').columns
    
    varMap = {'x': x, 'y': y, 'z': z, 'color': color}
    
    # Initialize the axes with default values if not provided
    for i, axis in enumerate(varMap.keys()):
        if varMap[axis] is None:
            varMap[axis] = quantVars[i]
    
    # Create the initial 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df[varMap['x']],
        y=df[varMap['y']],
        z=df[varMap['z']],
        mode='markers',
        marker=dict(
            size=2,
            color=df[varMap['color']],
        )
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title=varMap['x'],
            yaxis_title=varMap['y'],
            zaxis_title=varMap['z']
        )
    )
    
    return fig