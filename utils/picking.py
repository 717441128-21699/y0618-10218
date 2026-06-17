import numpy as np
from scipy.signal import find_peaks


def sta_lta(data, sampling_rate, sta_window, lta_window):
    """STA/LTA 算法计算特征函数"""
    nsta = int(sta_window * sampling_rate)
    nlta = int(lta_window * sampling_rate)
    
    if nsta < 1:
        nsta = 1
    if nlta < nsta:
        nlta = nsta
    
    npts = len(data)
    char_func = np.zeros(npts)
    
    abs_data = np.abs(data)
    
    sta = np.zeros(npts)
    lta = np.zeros(npts)
    
    cumsum = np.cumsum(abs_data)
    
    for i in range(npts):
        if i >= nsta:
            sta[i] = (cumsum[i] - cumsum[i - nsta]) / nsta
        else:
            sta[i] = cumsum[i] / (i + 1) if i > 0 else abs_data[0]
        
        if i >= nlta:
            lta[i] = (cumsum[i] - cumsum[i - nlta]) / nlta
        else:
            lta[i] = cumsum[i] / (i + 1) if i > 0 else abs_data[0]
    
    char_func = np.where(lta > 0, sta / lta, 0)
    
    return char_func


def find_first_peak(char_func, threshold, min_distance=50):
    """找到第一个超过阈值的峰值"""
    peaks, properties = find_peaks(
        char_func,
        height=threshold,
        distance=min_distance
    )
    
    if len(peaks) > 0:
        return peaks[0]
    return None


def auto_pick_ps(streams, sta_window=1.0, lta_window=10.0, threshold=4.0):
    """自动拾取P波和S波到时"""
    p_picks = {}
    s_picks = {}
    
    for name, tr in streams.items():
        data = tr.data
        sampling_rate = tr.stats.sampling_rate
        
        char_func = sta_lta(data, sampling_rate, sta_window, lta_window)
        
        p_idx = find_first_peak(char_func, threshold, min_distance=int(sampling_rate))
        
        if p_idx is not None:
            p_time = p_idx / sampling_rate
            
            search_start = p_idx + int(sampling_rate * 2)
            search_end = min(len(char_func), p_idx + int(sampling_rate * 30))
            
            if search_start < search_end:
                s_char = char_func[search_start:search_end]
                s_idx_rel = find_first_peak(s_char, threshold * 0.7, 
                                           min_distance=int(sampling_rate * 2))
                
                if s_idx_rel is not None:
                    s_idx = search_start + s_idx_rel
                    s_time = s_idx / sampling_rate
                else:
                    s_time = p_time + 5.0
            else:
                s_time = p_time + 5.0
        else:
            p_time = 10.0
            s_time = 20.0
        
        p_picks[name] = p_time
        s_picks[name] = s_time
    
    return p_picks, s_picks


def calculate_epicentral_distance(p_time, s_time, vp=6.0, vs=3.5):
    """
    根据P波S波走时差计算震源距离
    vp: P波速度 (km/s)
    vs: S波速度 (km/s)
    """
    ts_minus_tp = s_time - p_time
    
    if ts_minus_tp <= 0:
        return 0.0
    
    distance = ts_minus_tp * vp * vs / (vp - vs)
    
    return distance


def estimate_magnitude(amplitude, distance, correction=0.0):
    """估算震级（简化的里克特公式）"""
    if amplitude <= 0 or distance <= 0:
        return 0.0
    
    A = amplitude * 1e6
    mag = np.log10(A) + 3.0 * np.log10(distance) - 2.92 + correction
    
    return mag
