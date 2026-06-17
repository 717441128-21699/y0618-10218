import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd


def create_station_map(stations_info, quality_metrics=None,
                       map_style="OpenStreetMap", show_quality=True,
                       marker_size=12):
    """
    创建台站地图
    
    参数:
        stations_info: 台站信息 DataFrame，包含 latitude, longitude, elevation 等
        quality_metrics: 质量指标字典 {station_name: snr_value}
        map_style: 地图样式
        show_quality: 是否显示信号质量
        marker_size: 标记大小
    
    返回:
        folium.Map 对象
    """
    if stations_info.empty:
        return folium.Map(location=[0, 0], zoom_start=2)
    
    center_lat = stations_info["latitude"].mean()
    center_lon = stations_info["longitude"].mean()
    
    tiles = {
        "OpenStreetMap": "OpenStreetMap",
        "Stamen Terrain": "Stamen Terrain",
        "Stamen Toner": "Stamen Toner"
    }.get(map_style, "OpenStreetMap")
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles=tiles
    )
    
    marker_cluster = MarkerCluster().add_to(m)
    
    for station_name, info in stations_info.iterrows():
        lat = info["latitude"]
        lon = info["longitude"]
        elev = info.get("elevation", 0)
        dist = info.get("distance_km", 0)
        
        if quality_metrics and station_name in quality_metrics:
            snr = quality_metrics[station_name]
            if snr > 5:
                color = "green"
                quality = "优"
            elif snr >= 2:
                color = "orange"
                quality = "中"
            else:
                color = "red"
                quality = "差"
        else:
            color = "blue"
            quality = "未知"
            snr = 0
        
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <h4 style="margin: 0 0 8px 0;">台站: {station_name}</h4>
            <p style="margin: 4px 0;"><b>纬度:</b> {lat:.4f}°</p>
            <p style="margin: 4px 0;"><b>经度:</b> {lon:.4f}°</p>
            <p style="margin: 4px 0;"><b>海拔:</b> {elev:.0f} m</p>
            <p style="margin: 4px 0;"><b>震中距:</b> {dist:.1f} km</p>
            <p style="margin: 4px 0;"><b>信噪比:</b> {snr:.2f} dB</p>
            <p style="margin: 4px 0;"><b>信号质量:</b> {quality}</p>
        </div>
        """
        
        popup = folium.Popup(popup_html, max_width=300)
        
        if show_quality and quality_metrics:
            icon = folium.Icon(color=color, icon='signal', prefix='fa')
        else:
            icon = folium.Icon(color='blue', icon='info-sign')
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_size,
            popup=popup,
            color=color if show_quality else 'blue',
            fill=True,
            fill_color=color if show_quality else 'blue',
            fill_opacity=0.7,
            weight=2
        ).add_to(marker_cluster)
        
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 10pt; font-weight: bold; '
                     f'color: black; text-shadow: 1px 1px 1px white;">'
                     f'{station_name}</div>'
            )
        ).add_to(marker_cluster)
    
    if quality_metrics and show_quality:
        legend_html = """
        <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            z-index: 1000;
            background-color: white;
            padding: 10px;
            border: 2px solid grey;
            border-radius: 5px;
            font-family: Arial;
            font-size: 12px;
        ">
            <p style="margin: 0 0 5px 0;"><b>信号质量图例</b></p>
            <p style="margin: 3px 0;">
                <span style="color: green; font-size: 16px;">●</span> 优 (SNR > 5)
            </p>
            <p style="margin: 3px 0;">
                <span style="color: orange; font-size: 16px;">●</span> 中 (2 ≤ SNR ≤ 5)
            </p>
            <p style="margin: 3px 0;">
                <span style="color: red; font-size: 16px;">●</span> 差 (SNR < 2)
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    
    folium.LayerControl().add_to(m)
    
    return m


def create_heatmap(stations_info, quality_metrics=None):
    """创建台站信号质量热力图"""
    if stations_info.empty:
        return folium.Map(location=[0, 0], zoom_start=2)
    
    center_lat = stations_info["latitude"].mean()
    center_lon = stations_info["longitude"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap"
    )
    
    heat_data = []
    for station_name, info in stations_info.iterrows():
        lat = info["latitude"]
        lon = info["longitude"]
        
        if quality_metrics and station_name in quality_metrics:
            weight = min(quality_metrics[station_name] / 10.0, 1.0)
        else:
            weight = 0.5
        
        heat_data.append([lat, lon, weight])
    
    HeatMap(heat_data, radius=50, blur=30).add_to(m)
    
    return m


def calculate_station_distance(lat1, lon1, lat2, lon2):
    """计算两个坐标点之间的距离（km）"""
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
