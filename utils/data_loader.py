import numpy as np
import pandas as pd
from obspy import read, Stream
from io import BytesIO
import tempfile
import os


def generate_synthetic_waveform(duration, sampling_rate, station_name, 
                                p_arrival=None, s_arrival=None, noise_level=0.1):
    """生成合成地震波形数据"""
    npts = int(duration * sampling_rate)
    t = np.arange(npts) / sampling_rate
    
    data = np.random.randn(npts) * noise_level
    
    if p_arrival is None:
        p_arrival = duration * 0.3
    if s_arrival is None:
        s_arrival = p_arrival + duration * 0.15
    
    p_duration = 2.0
    p_mask = (t >= p_arrival) & (t < p_arrival + p_duration)
    p_env = np.zeros_like(t)
    p_window = np.arange(int(p_duration * sampling_rate)) / (p_duration * sampling_rate)
    p_env[p_mask] = np.sin(np.pi * p_window) * (1 - 0.3 * p_window)
    
    p_freq = 3.0 + np.random.rand() * 2.0
    p_wave = np.sin(2 * np.pi * p_freq * (t - p_arrival))
    data += p_wave * p_env * 2.0
    
    s_duration = 4.0
    s_mask = (t >= s_arrival) & (t < s_arrival + s_duration)
    s_env = np.zeros_like(t)
    s_window = np.arange(int(s_duration * sampling_rate)) / (s_duration * sampling_rate)
    s_env[s_mask] = np.sin(np.pi * s_window) * (1 - 0.5 * s_window)
    
    s_freq = 1.5 + np.random.rand() * 1.0
    s_wave = np.sin(2 * np.pi * s_freq * (t - s_arrival))
    data += s_wave * s_env * 3.0
    
    coda_start = s_arrival + s_duration * 0.5
    coda_duration = duration - coda_start
    if coda_duration > 0:
        coda_mask = t >= coda_start
        coda_env = np.zeros_like(t)
        coda_t = np.arange(np.sum(coda_mask)) / sampling_rate
        coda_env[coda_mask] = np.exp(-coda_t / coda_duration * 2)
        
        coda_freq = 2.0 + np.random.rand() * 1.5
        coda_wave = np.sin(2 * np.pi * coda_freq * (t - coda_start))
        data += coda_wave * coda_env * 1.5
    
    from obspy import Trace
    from obspy.core.utcdatetime import UTCDateTime
    from obspy.core.trace import Stats
    
    stats = Stats()
    stats.station = station_name
    stats.network = "XX"
    stats.channel = "BHZ"
    stats.sampling_rate = sampling_rate
    stats.starttime = UTCDateTime("2024-01-01T00:00:00")
    
    tr = Trace(data=data.astype(np.float32), header=stats)
    
    return tr


def load_example_data(num_stations=6):
    """生成示例地震数据"""
    streams = {}
    stations_data = []
    
    duration = 60.0
    sampling_rate = 100.0
    
    base_lat = 30.0
    base_lon = 103.0
    
    station_names = [
        "AAA", "BBB", "CCC", "DDD", "EEE", "FFF",
        "GGG", "HHH", "III", "JJJ"
    ]
    
    for i in range(num_stations):
        name = station_names[i]
        noise = 0.05 + np.random.rand() * 0.1
        distance_km = 10 + i * 25
        
        p_time = 5.0 + distance_km / 6.0 + np.random.randn() * 0.5
        s_time = p_time + distance_km / 3.5 + np.random.randn() * 0.8
        
        amplitude_factor = 1.0 / (1.0 + distance_km / 50.0)
        
        tr = generate_synthetic_waveform(
            duration, sampling_rate, name,
            p_arrival=p_time, s_arrival=s_time,
            noise_level=noise
        )
        tr.data *= amplitude_factor
        
        streams[name] = tr
        
        lat = base_lat + (np.random.rand() - 0.5) * 3.0
        lon = base_lon + (np.random.rand() - 0.5) * 4.0
        stations_data.append({
            "station": name,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "elevation": int(np.random.rand() * 2000 + 500),
            "distance_km": round(distance_km, 1)
        })
    
    stations_df = pd.DataFrame(stations_data)
    stations_df.set_index("station", inplace=True)
    
    return streams, stations_df


def load_seed_file(file_obj):
    """加载SEED/MiniSEED文件"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mseed') as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name
    
    try:
        st = read(tmp_path)
        
        streams = {}
        stations_data = []
        
        for tr in st:
            station_name = tr.stats.station
            if station_name not in streams:
                streams[station_name] = tr
                
                lat = tr.stats.get('latitude', 0.0)
                lon = tr.stats.get('longitude', 0.0)
                elev = tr.stats.get('elevation', 0.0)
                
                stations_data.append({
                    "station": station_name,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elev,
                    "distance_km": 0.0
                })
        
        stations_df = pd.DataFrame(stations_data)
        stations_df.set_index("station", inplace=True)
        
        return streams, stations_df
    finally:
        os.unlink(tmp_path)


def load_sac_file(file_objects):
    """加载SAC文件列表"""
    streams = {}
    stations_data = []
    
    for file_obj in file_objects:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sac') as tmp:
            tmp.write(file_obj.getvalue())
            tmp_path = tmp.name
        
        try:
            st = read(tmp_path, format='SAC')
            
            for tr in st:
                station_name = tr.stats.station or file_obj.name.replace('.sac', '')
                if station_name not in streams:
                    streams[station_name] = tr
                    
                    sac_header = tr.stats.get('sac', {})
                    lat = sac_header.get('stla', 0.0)
                    lon = sac_header.get('stlo', 0.0)
                    elev = sac_header.get('stel', 0.0)
                    dist = sac_header.get('dist', 0.0)
                    
                    stations_data.append({
                        "station": station_name,
                        "latitude": lat,
                        "longitude": lon,
                        "elevation": elev,
                        "distance_km": dist
                    })
        finally:
            os.unlink(tmp_path)
    
    stations_df = pd.DataFrame(stations_data)
    stations_df.set_index("station", inplace=True)
    
    return streams, stations_df
