import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path
import pandas as pd


def generate_sample_moment_tensor():
    """生成示例矩张量 (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)"""
    m0 = 1.5e18
    mt = np.array([
        m0 * 0.6,
        -m0 * 0.4,
        -m0 * 0.2,
        m0 * 0.5,
        -m0 * 0.2,
        m0 * 0.3
    ])
    return mt


def moment_tensor_to_t_axes(mt):
    """从矩张量计算T轴、P轴、B轴"""
    mt_matrix = np.array([
        [mt[0], mt[3], mt[4]],
        [mt[3], mt[1], mt[5]],
        [mt[4], mt[5], mt[2]]
    ])
    
    eigenvalues, eigenvectors = np.linalg.eigh(mt_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    t_axis = eigenvectors[:, 0]
    b_axis = eigenvectors[:, 1]
    p_axis = eigenvectors[:, 2]
    
    return t_axis, p_axis, b_axis, eigenvalues


def calculate_scalar_moment(mt):
    """计算标量地震矩"""
    mt = np.array(mt)
    m0 = np.sqrt(np.sum(mt**2) / 2.0)
    return m0


def calculate_moment_magnitude(mt):
    """计算矩震级 Mw"""
    m0 = calculate_scalar_moment(mt)
    mw = (2.0 / 3.0) * (np.log10(m0) - 9.1)
    return mw


def equal_area_projection(azimuth, plunge):
    """
    等面积投影（下半球投影）
    azimuth: 方位角 (度), 从正北顺时针
    plunge: 倾角 (度), 0=水平, 90=垂直向下
    返回 (x, y) 坐标，在单位圆内
    """
    az_rad = np.radians(azimuth)
    pl_rad = np.radians(plunge)
    
    r = np.sqrt(2) * np.sin(pl_rad / 2.0)
    
    x = r * np.sin(az_rad)
    y = -r * np.cos(az_rad)
    
    return x, y


def equal_area_inverse(x, y):
    """等面积投影的逆变换，返回 (方位角, 倾角)"""
    r = np.sqrt(x**2 + y**2)
    if r > np.sqrt(2):
        return None, None
    
    plunge = 2 * np.degrees(np.arcsin(r / np.sqrt(2)))
    
    azimuth = np.degrees(np.arctan2(x, -y))
    if azimuth < 0:
        azimuth += 360
    
    return azimuth, plunge


def vector_to_azimuth_plunge(vector):
    """将三维单位向量转换为方位角和倾角（下半球）"""
    x, y, z = vector
    
    if z > 0:
        x, y, z = -x, -y, -z
    
    plunge = np.degrees(np.arccos(np.abs(z)))
    azimuth = np.degrees(np.arctan2(y, x))
    
    if azimuth < 0:
        azimuth += 360
    
    return azimuth, plunge


def compute_radiation_pattern(mt, npts_angular=360, npts_radial=100):
    """
    计算下半球的P波辐射花样
    返回: angles, radii, radiation_grid, X, Y
    """
    mt = np.array(mt)
    mt_matrix = np.array([
        [mt[0], mt[3], mt[4]],
        [mt[3], mt[1], mt[5]],
        [mt[4], mt[5], mt[2]]
    ])
    
    angles = np.linspace(0, 2 * np.pi, npts_angular)
    radii = np.linspace(0, 1, npts_radial)
    
    radiation_grid = np.zeros((npts_angular, npts_radial))
    
    for i, angle in enumerate(angles):
        for j, r in enumerate(radii):
            plunge = 2 * np.degrees(np.arcsin(r / np.sqrt(2)))
            
            n = np.array([
                np.sin(np.radians(plunge)) * np.cos(angle),
                np.sin(np.radians(plunge)) * np.sin(angle),
                np.cos(np.radians(plunge))
            ])
            
            radiation_grid[i, j] = np.dot(n, np.dot(mt_matrix, n))
    
    X = np.outer(np.cos(angles), radii)
    Y = np.outer(np.sin(angles), radii)
    
    return angles, radii, radiation_grid, X, Y


def find_nodal_lines(mt, npts=720):
    """
    找到节线（辐射花样为零的位置）
    返回节线的两个大圆弧的点列表
    """
    mt = np.array(mt)
    mt_matrix = np.array([
        [mt[0], mt[3], mt[4]],
        [mt[3], mt[1], mt[5]],
        [mt[4], mt[5], mt[2]]
    ])
    
    eigenvalues, eigenvectors = np.linalg.eigh(mt_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    t_axis = eigenvectors[:, 0]
    p_axis = eigenvectors[:, 2]
    
    null_axis = np.cross(t_axis, p_axis)
    null_axis = null_axis / np.linalg.norm(null_axis)
    
    def great_circle_points(normal, n_pts=360):
        """计算下半球上大圆弧的点"""
        normal = normal / np.linalg.norm(normal)
        
        if normal[2] < 0:
            normal = -normal
        
        theta = np.linspace(0, 2 * np.pi, n_pts)
        
        v1 = np.array([-normal[1], normal[0], 0.0])
        if np.linalg.norm(v1) < 1e-10:
            v1 = np.array([1.0, 0.0, 0.0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        points = []
        for angle in theta:
            p = np.cos(angle) * v1 + np.sin(angle) * v2
            
            if p[2] > 0:
                p = -p
            
            az, pl = vector_to_azimuth_plunge(p)
            x, y = equal_area_projection(az, pl)
            points.append((x, y))
        
        return np.array(points)
    
    plane1_normal = (t_axis + p_axis) / np.linalg.norm(t_axis + p_axis)
    plane2_normal = (t_axis - p_axis) / np.linalg.norm(t_axis - p_axis)
    
    points1 = great_circle_points(plane1_normal)
    points2 = great_circle_points(plane2_normal)
    
    return points1, points2


def plot_beach_ball_moment_tensor(mt, show_axes=True, size=6):
    """
    根据矩张量绘制完整的沙滩球图（带辐射花样填充）
    
    参数:
        mt: 矩张量 [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
        show_axes: 是否显示T轴和P轴
        size: 图大小 (英寸)
    
    返回:
        matplotlib Figure 对象
    """
    mt = np.array(mt)
    
    fig, ax = plt.subplots(figsize=(size, size))
    
    angles, radii, radiation, X, Y = compute_radiation_pattern(mt)
    
    max_val = np.max(np.abs(radiation))
    
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=-max_val, vmax=max_val)
    cmap = plt.cm.RdBu_r
    
    ax.pcolormesh(X, Y, radiation, cmap=cmap, norm=norm, 
                  shading='auto', alpha=0.9)
    
    try:
        points1, points2 = find_nodal_lines(mt)
        ax.plot(points1[:, 0], points1[:, 1], 'k-', linewidth=2.0, zorder=3)
        ax.plot(points2[:, 0], points2[:, 1], 'k-', linewidth=2.0, zorder=3)
    except Exception:
        pass
    
    circle = Circle((0, 0), 1, fill=False, linewidth=2.5, color='black', zorder=5)
    ax.add_patch(circle)
    
    if show_axes:
        t_axis, p_axis, b_axis, _ = moment_tensor_to_t_axes(mt)
        
        t_az, t_pl = vector_to_azimuth_plunge(t_axis)
        p_az, p_pl = vector_to_azimuth_plunge(p_axis)
        
        t_x, t_y = equal_area_projection(t_az, t_pl)
        p_x, p_y = equal_area_projection(p_az, p_pl)
        
        ax.plot(t_x, t_y, 'o', color='white', markersize=10,
               markeredgecolor='black', markeredgewidth=2, 
               label='T轴 (拉伸)', zorder=10)
        ax.plot(p_x, p_y, 's', color='black', markersize=10,
               markeredgecolor='white', markeredgewidth=1.5,
               label='P轴 (压缩)', zorder=10)
        
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    ax.plot([-1, 1], [0, 0], 'k-', linewidth=0.5, alpha=0.3, zorder=2)
    ax.plot([0, 0], [-1, 1], 'k-', linewidth=0.5, alpha=0.3, zorder=2)
    
    m0 = calculate_scalar_moment(mt)
    mw = calculate_moment_magnitude(mt)
    
    info_text = f'矩震级 Mw = {mw:.2f}\n标量矩 M₀ = {m0:.2e} N·m'
    ax.text(0, -1.18, info_text, ha='center', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
    
    ax.text(0, 1.08, 'N', ha='center', fontsize=10, fontweight='bold')
    ax.arrow(0, 0.95, 0, 0.15, head_width=0.05, head_length=0.08, 
             fc='black', ec='black', zorder=4)
    
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('震源机制解 (矩张量)', fontsize=13, fontweight='bold', pad=10)
    
    return fig


def plot_beach_ball_first_motion(polarities, show_border=True, size=6):
    """
    根据初动极性绘制沙滩球图
    
    参数:
        polarities: 列表 [(station_name, azimuth, plunge, polarity), ...]
                   polarity: 'compressional' (压缩) 或 'dilational' (拉张)
        show_border: 是否显示边框
        size: 图大小
    
    返回:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=(size, size))
    
    if show_border:
        circle = Circle((0, 0), 1, fill=True, facecolor='#f0f0f0', 
                       edgecolor='black', linewidth=2.5, zorder=1)
        ax.add_patch(circle)
    
    for item in polarities:
        if len(item) == 4:
            name, az, pl, pol = item
        else:
            az, pl, pol = item
            name = ''
        
        x, y = equal_area_projection(az, pl)
        
        if x**2 + y**2 > 1.05:
            continue
        
        if pol in ['compressional', 'C', 'c', '+', 1, 'up']:
            ax.plot(x, y, 'o', color='black', markersize=9,
                   markeredgecolor='black', markeredgewidth=1, zorder=5)
        else:
            ax.plot(x, y, 'o', color='white', markersize=9,
                   markeredgecolor='black', markeredgewidth=2, zorder=5)
    
    ax.plot([-1, 1], [0, 0], 'k-', linewidth=0.5, alpha=0.3, zorder=2)
    ax.plot([0, 0], [-1, 1], 'k-', linewidth=0.5, alpha=0.3, zorder=2)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markeredgecolor='black', markersize=10, label='压缩 (P波初动向上)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=10, label='拉张 (P波初动向下)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    
    ax.text(0, 1.08, 'N', ha='center', fontsize=10, fontweight='bold')
    ax.arrow(0, 0.95, 0, 0.15, head_width=0.05, head_length=0.08, 
             fc='black', ec='black', zorder=4)
    
    num_comp = sum(1 for p in polarities if p[3] in ['compressional', 'C', 'c', '+', 1, 'up'])
    num_dil = sum(1 for p in polarities if p[3] in ['dilational', 'D', 'd', '-', -1, 'down'])
    
    info_text = f'总初动数: {len(polarities)}  (压缩: {num_comp}, 拉张: {num_dil})'
    ax.text(0, -1.18, info_text, ha='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
    
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('震源机制解 (初动极性)', fontsize=13, fontweight='bold', pad=10)
    
    return fig


def compute_station_azimuth(station_lat, station_lon, source_lat, source_lon):
    """
    计算台站相对于震源的方位角（从正北顺时针）
    
    参数:
        station_lat, station_lon: 台站经纬度
        source_lat, source_lon: 震源经纬度
    
    返回:
        azimuth: 方位角 (度)
    """
    from math import radians, sin, cos, atan2
    
    lat1 = radians(source_lat)
    lon1 = radians(source_lon)
    lat2 = radians(station_lat)
    lon2 = radians(station_lon)
    
    dlon = lon2 - lon1
    
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    azimuth = atan2(y, x)
    azimuth = np.degrees(azimuth)
    
    if azimuth < 0:
        azimuth += 360
    
    return azimuth


def compute_distance_km(lat1, lon1, lat2, lon2):
    """计算两点之间的大圆距离（km）"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371.0
    
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    
    return distance


def compute_takeoff_angle(epicentral_distance_km, source_depth_km, vp=6.0):
    """
    计算P波出射角（从震源向下测量，0度=垂直向下）
    
    使用简单的直线近似，实际应使用射线追踪
    
    参数:
        epicentral_distance_km: 震中距 (km)
        source_depth_km: 震源深度 (km)
        vp: P波速度 (km/s)
    
    返回:
        takeoff_angle: 出射角 (度), 0=垂直向下, 90=水平
    """
    ray_length = np.sqrt(epicentral_distance_km**2 + source_depth_km**2)
    
    if ray_length < 1e-6:
        return 0.0
    
    cos_angle = source_depth_km / ray_length
    takeoff = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    return takeoff


def estimate_source_location(stations_info, p_picks, vp=6.0):
    """
    简单估算震源位置（基于P波到时的近似）
    
    返回:
        (lat, lon, depth_km)
    """
    if stations_info is None or len(stations_info) < 3:
        return 30.0, 103.0, 10.0
    
    avg_lat = stations_info["latitude"].mean()
    avg_lon = stations_info["longitude"].mean()
    
    min_p_time = float('inf')
    for name in stations_info.index:
        if name in p_picks:
            if p_picks[name] < min_p_time:
                min_p_time = p_picks[name]
    
    depth = 10.0 + min_p_time * 2.0
    
    return avg_lat, avg_lon, depth


def extract_first_motions(streams, p_picks, stations_info=None):
    """
    从波形和P波到时提取初动极性
    
    参数:
        streams: 字典 {station_name: obspy.Trace}
        p_picks: 字典 {station_name: p_arrival_time_seconds}
        stations_info: 可选，台站位置信息 DataFrame
    
    返回:
        polarities: 列表 [(station_name, azimuth, plunge, polarity, quality)]
    """
    polarities = []
    
    source_lat, source_lon, source_depth = 30.0, 103.0, 10.0
    if stations_info is not None and len(stations_info) >= 3:
        source_lat, source_lon, source_depth = estimate_source_location(
            stations_info, p_picks
        )
    
    for station_name, tr in streams.items():
        if station_name not in p_picks:
            continue
        
        p_time = p_picks[station_name]
        sampling_rate = tr.stats.sampling_rate
        data = tr.data
        
        p_idx = int(p_time * sampling_rate)
        if p_idx < 5 or p_idx >= len(data) - 10:
            continue
        
        window_before_start = max(0, p_idx - int(1.0 * sampling_rate))
        window_before_end = max(0, p_idx - int(0.1 * sampling_rate))
        
        if window_before_end > window_before_start:
            noise_std = np.std(data[window_before_start:window_before_end])
        else:
            noise_std = np.std(data[:p_idx]) if p_idx > 10 else 1e-10
        
        if noise_std < 1e-10:
            noise_std = 1e-10
        
        search_start = p_idx - int(0.05 * sampling_rate)
        search_end = min(len(data), p_idx + int(0.5 * sampling_rate))
        
        if search_start < 0:
            search_start = 0
        if search_end <= search_start + 2:
            continue
        
        search_data = data[search_start:search_end]
        
        first_peak_idx = None
        first_peak_val = -float('inf')
        first_trough_idx = None
        first_trough_val = float('inf')
        
        threshold = noise_std * 1.5
        
        for i in range(1, len(search_data) - 1):
            val = search_data[i]
            
            if val > threshold and val > search_data[i-1] and val > search_data[i+1]:
                if first_peak_idx is None:
                    first_peak_idx = i
                    first_peak_val = val
            
            if val < -threshold and val < search_data[i-1] and val < search_data[i+1]:
                if first_trough_idx is None:
                    first_trough_idx = i
                    first_trough_val = val
            
            if first_peak_idx is not None and first_trough_idx is not None:
                break
        
        if first_peak_idx is None and first_trough_idx is None:
            if search_data[0] < search_data[-1]:
                polarity = 'compressional'
                amplitude = abs(search_data[-1])
            else:
                polarity = 'dilational'
                amplitude = abs(search_data[0])
            quality = 'poor'
        elif first_peak_idx is not None and first_trough_idx is not None:
            if first_peak_idx < first_trough_idx:
                polarity = 'compressional'
                amplitude = abs(first_peak_val)
            else:
                polarity = 'dilational'
                amplitude = abs(first_trough_val)
        elif first_peak_idx is not None:
            polarity = 'compressional'
            amplitude = abs(first_peak_val)
        else:
            polarity = 'dilational'
            amplitude = abs(first_trough_val)
        
        snr = amplitude / noise_std if noise_std > 0 else 0
        
        if snr > 5:
            quality = 'good'
        elif snr > 2:
            quality = 'medium'
        else:
            quality = 'poor'
        
        if stations_info is not None and station_name in stations_info.index:
            info = stations_info.loc[station_name]
            st_lat = info["latitude"]
            st_lon = info["longitude"]
            
            azimuth = compute_station_azimuth(st_lat, st_lon, source_lat, source_lon)
            
            dist_km = info.get("distance_km", None)
            if dist_km is None or dist_km == 0:
                dist_km = compute_distance_km(source_lat, source_lon, st_lat, st_lon)
            
            takeoff = compute_takeoff_angle(dist_km, source_depth)
            plunge = 90 - takeoff
        else:
            station_idx = list(streams.keys()).index(station_name)
            azimuth = (station_idx * 360 / len(streams) + 30) % 360
            plunge = 30 + (station_idx % 3) * 20
        
        if plunge < 0:
            plunge = 0
        if plunge > 90:
            plunge = 90
        
        polarities.append((station_name, azimuth, plunge, polarity, quality))
    
    return polarities


def load_moment_tensor_file(file_obj):
    """
    从文件加载矩张量
    
    支持格式:
    - 纯文本: 6个数字（Mrr Mtt Mpp Mrt Mrp Mtp）
    - CSV格式: 带表头或不带表头
    - JSON格式
    
    参数:
        file_obj: 文件对象 (streamlit UploadedFile)
    
    返回:
        mt: 矩张量数组 [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
        info: 附加信息字典
    """
    import json
    
    content = file_obj.getvalue().decode('utf-8', errors='ignore')
    content = content.strip()
    
    info = {}
    mt = None
    
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            keys_lower = {k.lower(): v for k, v in data.items()}
            
            component_keys = ['mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp']
            mt_vals = []
            for key in component_keys:
                if key in keys_lower:
                    mt_vals.append(float(keys_lower[key]))
            
            if len(mt_vals) == 6:
                mt = np.array(mt_vals)
            
            for k, v in data.items():
                if k.lower() not in component_keys:
                    info[k] = v
            
            info['format'] = 'JSON'
    except (json.JSONDecodeError, ValueError):
        pass
    
    if mt is None:
        try:
            lines = content.split('\n')
            
            data_lines = []
            header_parts = None
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('%'):
                    continue
                
                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                else:
                    parts = line.split()
                
                try:
                    nums = [float(p) for p in parts]
                    data_lines.append(nums)
                except ValueError:
                    if header_parts is None:
                        header_parts = parts
            
            if len(data_lines) > 0:
                all_nums = []
                for row in data_lines:
                    all_nums.extend(row)
                
                if len(all_nums) >= 6:
                    mt = np.array(all_nums[:6])
                    info['format'] = 'Text/CSV'
                    
                    if header_parts and len(header_parts) >= 6:
                        info['columns'] = header_parts[:6]
        except Exception:
            pass
    
    if mt is None:
        raise ValueError("无法解析文件中的矩张量数据，请确保文件包含6个分量")
    
    return mt, info


def first_motion_to_dataframe(polarities):
    """将初动极性列表转换为DataFrame"""
    data = []
    for item in polarities:
        if len(item) >= 5:
            name, az, pl, pol, qual = item[:5]
        elif len(item) == 4:
            name, az, pl, pol = item
            qual = 'unknown'
        else:
            name = ''
            az, pl, pol = item[:3]
            qual = 'unknown'
        
        pol_cn = '压缩' if pol in ['compressional', 'C', 'c', '+', 1, 'up'] else '拉张'
        qual_cn = {'good': '优', 'medium': '中', 'poor': '差', 'unknown': '未知'}.get(qual, qual)
        
        data.append({
            '台站': name,
            '方位角 (°)': round(az, 1),
            '倾角 (°)': round(pl, 1),
            '极性': pol_cn,
            '质量': qual_cn
        })
    
    return pd.DataFrame(data)
