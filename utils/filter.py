import numpy as np
from scipy.signal import butter, filtfilt, iirfilter
import copy


def apply_filter(streams, filter_params):
    """
    对多个台站数据应用滤波
    
    参数:
        streams: 字典 {station_name: obspy.Trace}
        filter_params: 滤波参数字典
            - type: 'bandpass', 'highpass', 'lowpass'
            - freqmin/freqmax: 带通的高低截止频率
            - freq: 高通/低通的截止频率
            - corners: 滤波器阶数
    
    返回:
        filtered_streams: 滤波后的流字典
    """
    filtered_streams = {}
    
    for name, tr in streams.items():
        filtered_tr = copy.deepcopy(tr)
        data = tr.data.copy()
        sampling_rate = tr.stats.sampling_rate
        nyquist = sampling_rate / 2.0
        
        filter_type = filter_params.get("type", "bandpass")
        corners = filter_params.get("corners", 4)
        
        if filter_type == "bandpass":
            freqmin = filter_params.get("freqmin", 1.0)
            freqmax = filter_params.get("freqmax", 10.0)
            
            low = freqmin / nyquist
            high = freqmax / nyquist
            
            if low >= 1.0:
                low = 0.99
            if high >= 1.0:
                high = 0.99
            if low <= 0:
                low = 0.001
            if high <= low:
                high = low + 0.01
            
            b, a = butter(corners, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data)
            
        elif filter_type == "highpass":
            freq = filter_params.get("freq", 1.0)
            
            high = freq / nyquist
            if high >= 1.0:
                high = 0.99
            if high <= 0:
                high = 0.001
            
            b, a = butter(corners, high, btype='high')
            filtered_data = filtfilt(b, a, data)
            
        elif filter_type == "lowpass":
            freq = filter_params.get("freq", 5.0)
            
            low = freq / nyquist
            if low >= 1.0:
                low = 0.99
            if low <= 0:
                low = 0.001
            
            b, a = butter(corners, low, btype='low')
            filtered_data = filtfilt(b, a, data)
        else:
            filtered_data = data
        
        filtered_tr.data = filtered_data
        
        filtered_streams[name] = filtered_tr
    
    return filtered_streams


def design_filter(filter_type, freq, sampling_rate, corners=4):
    """设计滤波器并返回频率响应"""
    nyquist = sampling_rate / 2.0
    
    if filter_type == "bandpass":
        freqmin, freqmax = freq
        low = freqmin / nyquist
        high = freqmax / nyquist
        b, a = butter(corners, [low, high], btype='band')
    elif filter_type == "highpass":
        high = freq / nyquist
        b, a = butter(corners, high, btype='high')
    elif filter_type == "lowpass":
        low = freq / nyquist
        b, a = butter(corners, low, btype='low')
    else:
        b, a = np.array([1.0]), np.array([1.0])
    
    from scipy.signal import freqz
    w, h = freqz(b, a, worN=1024)
    freqs = w * sampling_rate / (2 * np.pi)
    response = np.abs(h)
    
    return freqs, response


def running_average(data, window_size):
    """滑动平均滤波"""
    if window_size <= 1:
        return data
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')
    
    return smoothed


def detrend_trace(trace, type='linear'):
    """去趋势"""
    import copy
    tr = copy.deepcopy(trace)
    data = tr.data
    
    if type == 'linear':
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        tr.data = data - trend
    elif type == 'mean':
        tr.data = data - np.mean(data)
    
    return tr


def taper_trace(trace, type='cosine', max_percentage=0.05):
    """加窗（窗函数）"""
    import copy
    tr = copy.deepcopy(trace)
    data = tr.data
    npts = len(data)
    
    taper_length = int(npts * max_percentage)
    
    if type == 'cosine':
        taper = np.ones(npts)
        for i in range(taper_length):
            val = 0.5 * (1 - np.cos(np.pi * i / taper_length))
            taper[i] = val
            taper[npts - 1 - i] = val
        tr.data = data * taper
    
    return tr
