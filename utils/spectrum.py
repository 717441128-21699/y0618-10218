import numpy as np
import plotly.graph_objects as go
from scipy.signal import get_window


def compute_spectrum(trace, freq_range=(0.1, 10.0), window_type="hann",
                     detect_anomaly=True, anomaly_threshold=3.0):
    """
    计算波形的频谱
    
    参数:
        trace: ObsPy Trace 对象
        freq_range: 频率范围 (min_freq, max_freq)
        window_type: 窗函数类型
        detect_anomaly: 是否检测异常
        anomaly_threshold: 异常阈值（倍标准差）
    
    返回:
        freqs: 频率数组
        spectrum: 幅值谱
        peak_freq: 主频
        anomalies: 异常频率点列表 [(freq, amplitude), ...]
    """
    data = trace.data
    sampling_rate = trace.stats.sampling_rate
    npts = len(data)
    
    if window_type and window_type != "none":
        window = get_window(window_type, npts)
        data_windowed = data * window
    else:
        data_windowed = data
    
    fft_result = np.fft.fft(data_windowed)
    freqs = np.fft.fftfreq(npts, 1.0 / sampling_rate)
    
    spectrum = np.abs(fft_result) / npts * 2
    spectrum[0] = spectrum[0] / 2
    
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    spectrum_pos = spectrum[pos_mask]
    
    freq_mask = (freqs_pos >= freq_range[0]) & (freqs_pos <= freq_range[1])
    freqs_filtered = freqs_pos[freq_mask]
    spectrum_filtered = spectrum_pos[freq_mask]
    
    if len(spectrum_filtered) > 0:
        peak_idx = np.argmax(spectrum_filtered)
        peak_freq = freqs_filtered[peak_idx]
    else:
        peak_freq = 0.0
    
    anomalies = []
    if detect_anomaly and len(spectrum_filtered) > 0:
        spec_log = np.log10(spectrum_filtered + 1e-10)
        mean_spec = np.mean(spec_log)
        std_spec = np.std(spec_log)
        
        upper_bound = mean_spec + anomaly_threshold * std_spec
        
        for i, (f, s) in enumerate(zip(freqs_filtered, spectrum_filtered)):
            if np.log10(s + 1e-10) > upper_bound:
                if i > 0 and i < len(spectrum_filtered) - 1:
                    if (s > spectrum_filtered[i-1] and s > spectrum_filtered[i+1]):
                        anomalies.append((f, s))
        
        anomalies = anomalies[:10]
    
    return freqs_filtered, spectrum_filtered, peak_freq, anomalies


def plot_spectrum(freqs, spectrum, freq_range, peak_freq, anomalies=None,
                  log_scale=True, station_name=""):
    """绘制频谱图"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=spectrum,
            mode='lines',
            name='频谱',
            line=dict(color='#1f77b4', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        )
    )
    
    fig.add_vline(
        x=peak_freq,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"主频: {peak_freq:.2f} Hz",
        annotation_position="top right",
        annotation_font_color="red"
    )
    
    if anomalies:
        anomaly_freqs = [a[0] for a in anomalies]
        anomaly_amps = [a[1] for a in anomalies]
        fig.add_trace(
            go.Scatter(
                x=anomaly_freqs,
                y=anomaly_amps,
                mode='markers',
                name='异常点',
                marker=dict(color='red', size=8, symbol='circle'),
                showlegend=True
            )
        )
    
    y_axis_type = "log" if log_scale else "linear"
    
    fig.update_layout(
        title=f"{station_name} 频谱分析" if station_name else "频谱分析",
        xaxis_title="频率 (Hz)",
        yaxis_title="幅值",
        yaxis_type=y_axis_type,
        xaxis_range=list(freq_range),
        template="plotly_white",
        height=500,
        hovermode="x unified"
    )
    
    return fig


def compute_spectrogram(trace, window_length=2.0, overlap=0.5):
    """计算时频谱（短时傅里叶变换）"""
    data = trace.data
    sampling_rate = trace.stats.sampling_rate
    
    nperseg = int(window_length * sampling_rate)
    noverlap = int(nperseg * overlap)
    
    from scipy.signal import stft
    
    f, t, Zxx = stft(
        data, fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann'
    )
    
    Sxx = np.abs(Zxx)
    
    return f, t, Sxx


def plot_spectrogram(freqs, times, Sxx, station_name=""):
    """绘制时频谱图"""
    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx + 1e-10),
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title='功率 (dB)')
    ))
    
    fig.update_layout(
        title=f"{station_name} 时频谱" if station_name else "时频谱",
        xaxis_title="时间 (秒)",
        yaxis_title="频率 (Hz)",
        height=400,
        template="plotly_white"
    )
    
    return fig
