import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_waveforms(streams, time_range=None, normalize=True,
                   p_picks=None, s_picks=None, height=700):
    """绘制多台站波形图"""
    station_names = list(streams.keys())
    num_stations = len(station_names)
    
    if num_stations == 0:
        fig = go.Figure()
        fig.update_layout(title="无数据", height=height)
        return fig
    
    fig = make_subplots(
        rows=num_stations, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=station_names
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, name in enumerate(station_names):
        tr = streams[name]
        t = tr.times()
        data = tr.data
        
        if time_range:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t_plot = t[mask]
            data_plot = data[mask]
        else:
            t_plot = t
            data_plot = data
        
        if normalize and len(data_plot) > 0:
            max_val = np.max(np.abs(data_plot))
            if max_val > 0:
                data_plot = data_plot / max_val
        
        color = colors[i % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=t_plot, y=data_plot,
                mode='lines',
                name=name,
                line=dict(color=color, width=1),
                showlegend=False
            ),
            row=i + 1, col=1
        )
        
        if p_picks and name in p_picks:
            p_time = p_picks[name]
            if time_range is None or (time_range[0] <= p_time <= time_range[1]):
                y_min = min(data_plot) if len(data_plot) > 0 else -1
                y_max = max(data_plot) if len(data_plot) > 0 else 1
                fig.add_vline(
                    x=p_time,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text="P",
                    annotation_position="top right",
                    annotation_font_color="red",
                    row=i + 1, col=1
                )
        
        if s_picks and name in s_picks:
            s_time = s_picks[name]
            if time_range is None or (time_range[0] <= s_time <= time_range[1]):
                y_min = min(data_plot) if len(data_plot) > 0 else -1
                y_max = max(data_plot) if len(data_plot) > 0 else 1
                fig.add_vline(
                    x=s_time,
                    line_dash="dash",
                    line_color="blue",
                    line_width=2,
                    annotation_text="S",
                    annotation_position="top right",
                    annotation_font_color="blue",
                    row=i + 1, col=1
                )
        
        y_label = "归一化振幅" if normalize else "振幅"
        fig.update_yaxes(title_text=y_label, row=i + 1, col=1)
    
    fig.update_xaxes(title_text="时间 (秒)", row=num_stations, col=1)
    
    fig.update_layout(
        height=height,
        title_text="地震波形图",
        hovermode="x unified",
        template="plotly_white"
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=10)
    
    return fig


def plot_single_waveform(trace, title="波形图", p_pick=None, s_pick=None):
    """绘制单道波形"""
    t = trace.times()
    data = trace.data
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=data, mode='lines', name='波形',
                  line=dict(color='#1f77b4', width=1.5))
    )
    
    if p_pick is not None:
        fig.add_vline(x=p_pick, line_dash="dash", line_color="red",
                     line_width=2, annotation_text="P波",
                     annotation_font_color="red")
    
    if s_pick is not None:
        fig.add_vline(x=s_pick, line_dash="dash", line_color="blue",
                     line_width=2, annotation_text="S波",
                     annotation_font_color="blue")
    
    fig.update_layout(
        title=title,
        xaxis_title="时间 (秒)",
        yaxis_title="振幅",
        template="plotly_white",
        height=300
    )
    
    return fig
