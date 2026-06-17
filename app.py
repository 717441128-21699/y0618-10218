import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="地震波形数据分析工具",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.data_loader import load_example_data, load_seed_file, load_sac_file
from utils.waveform_plot import plot_waveforms
from utils.picking import auto_pick_ps, calculate_epicentral_distance
from utils.spectrum import compute_spectrum, plot_spectrum
from utils.focal_mechanism import (
    plot_beach_ball_moment_tensor,
    plot_beach_ball_first_motion,
    generate_sample_moment_tensor,
    extract_first_motions,
    load_moment_tensor_file,
    first_motion_to_dataframe,
    calculate_scalar_moment,
    calculate_moment_magnitude,
    save_beachball_to_png,
    create_first_motion_csv,
    create_event_report_json,
    create_complete_report_figure,
    save_report_to_pdf,
    create_report_zip_package
)
from utils.filter import apply_filter
from utils.station_map import create_station_map
from utils.waveform_plot import plot_single_waveform

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import numpy as np

st.title("🌍 地震波形数据分析工具")
st.markdown("---")

if "streams" not in st.session_state:
    st.session_state.streams = None
    st.session_state.stations_info = None
    st.session_state.p_picks = {}
    st.session_state.s_picks = {}
    st.session_state.filtered_streams = None
    st.session_state.polarities = []
    st.session_state.custom_mt = None
    st.session_state.mt_file_info = None
    st.session_state.selected_station = None
    st.session_state.excluded_stations = []
    st.session_state.review_notes = {}


def clear_report_cache():
    """清除报告预览缓存，保证排除/复核后立即刷新"""
    for k in list(st.session_state.keys()):
        if k.startswith("_report"):
            del st.session_state[k]


def parse_clicked_station(map_ret, station_names):
    """从 streamlit_folium 返回值中提取被点击的台站名
    
    兼容点圆点/tooltip/popup 等多种点击情况，
    多层解析返回字段，不同版本 streamlit-folium 键名不同
    """
    if not isinstance(map_ret, dict):
        return None

    candidate_strings = []

    # 第一层常见键
    for key in [
        "last_object_clicked_tooltip",
        "last_object_clicked_popup",
        "last_active_tooltip",
        "last_active_popup",
    ]:
        v = map_ret.get(key)
        if isinstance(v, str):
            candidate_strings.append(v)
        elif isinstance(v, dict):
            for sk in ["tooltip", "popup", "text", "station", "name", "id"]:
                if isinstance(v.get(sk), str):
                    candidate_strings.append(v[sk])

    # last_object_clicked 内部字段
    for key in ["last_object_clicked", "last_active_drawing"]:
        v = map_ret.get(key)
        if isinstance(v, dict):
            props = v.get("properties", {}) if isinstance(v.get("properties"), dict) else {}
            for sk in [
                "tooltip", "popup", "station", "stations", "name", "id",
                "text", "Station", "tooltip_content", "popup_content"
            ]:
                val = props.get(sk) or v.get(sk)
                if isinstance(val, str):
                    candidate_strings.append(val)
            # properties 里的值有时是 dict 再带 html
            for sk in ["tooltip", "popup"]:
                val = props.get(sk) or v.get(sk)
                if isinstance(val, dict):
                    for ssk in ["value", "text", "content"]:
                        if isinstance(val.get(ssk), str):
                            candidate_strings.append(val[ssk])

    # 其它键的字符串值
    for key, v in map_ret.items():
        if isinstance(v, str) and len(v) > 0 and len(v) < 50:
            candidate_strings.append(v)

    # 匹配：精确匹配优先，否则包含匹配
    for cs in candidate_strings:
        if cs in station_names:
            return cs
    for cs in candidate_strings:
        for sn in station_names:
            if sn and (sn in cs or cs in sn):
                return sn

    return None


with st.sidebar:
    st.header("📂 数据加载")
    data_source = st.radio(
        "数据源",
        ["示例数据", "上传SEED文件", "上传SAC文件"],
        index=0
    )

    if data_source == "示例数据":
        num_stations = st.slider("台站数量", 3, 10, 6)
        if st.button("生成示例数据", type="primary"):
            streams, stations_info = load_example_data(num_stations)
            st.session_state.streams = streams
            st.session_state.stations_info = stations_info
            st.session_state.p_picks = {}
            st.session_state.s_picks = {}
            st.session_state.filtered_streams = None
            st.success(f"已生成 {num_stations} 个台站的示例数据")
    elif data_source == "上传SEED文件":
        seed_file = st.file_uploader("选择SEED文件", type=["seed", "mseed", "miniseed"])
        if seed_file is not None and st.button("加载SEED文件", type="primary"):
            try:
                streams, stations_info = load_seed_file(seed_file)
                st.session_state.streams = streams
                st.session_state.stations_info = stations_info
                st.session_state.p_picks = {}
                st.session_state.s_picks = {}
                st.session_state.filtered_streams = None
                st.success("SEED文件加载成功")
            except Exception as e:
                st.error(f"加载失败: {str(e)}")
    elif data_source == "上传SAC文件":
        sac_files = st.file_uploader("选择SAC文件（可多选）", type=["sac"], accept_multiple_files=True)
        if sac_files and st.button("加载SAC文件", type="primary"):
            try:
                streams, stations_info = load_sac_file(sac_files)
                st.session_state.streams = streams
                st.session_state.stations_info = stations_info
                st.session_state.p_picks = {}
                st.session_state.s_picks = {}
                st.session_state.filtered_streams = None
                st.success(f"成功加载 {len(streams)} 个SAC文件")
            except Exception as e:
                st.error(f"加载失败: {str(e)}")

    st.markdown("---")
    st.header("🔧 功能模块")
    page = st.selectbox(
        "选择功能",
        [
            "波形展示",
            "P/S波拾取与震源距离",
            "频谱分析",
            "频谱-滤波对比",
            "台站质量检查",
            "震源机制解",
            "滤波工具",
            "台站地图"
        ]
    )

if st.session_state.streams is None:
    st.info("👈 请在左侧边栏加载数据或生成示例数据开始使用")
    
    st.markdown("### 功能概览")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **波形展示**\n\n多台站波形并排展示，时间轴同步联动")
    with col2:
        st.info("🔍 **P/S波拾取**\n\n自动拾取P波S波到时，人工校正，计算震源距离")
    with col3:
        st.info("📈 **频谱分析**\n\n傅里叶变换，展示频率成分分布")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.info("🎯 **震源机制解**\n\n沙滩球图，支持矩张量导入")
    with col5:
        st.info("🎛️ **滤波工具**\n\n带通/高通/低通滤波，实时预览")
    with col6:
        st.info("🗺️ **台站地图**\n\n台站位置和信号质量分布")
else:
    streams = st.session_state.streams
    stations_info = st.session_state.stations_info
    
    display_streams = st.session_state.filtered_streams if st.session_state.filtered_streams else streams

    if page == "波形展示":
        st.header("📊 多台站波形展示")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("显示设置")
            all_stations = list(display_streams.keys())
            
            if st.session_state.selected_station and st.session_state.selected_station in all_stations:
                st.info(f"📌 当前选中台站: **{st.session_state.selected_station}**（来自地图联动）")
                if st.button("清除选中"):
                    st.session_state.selected_station = None
            
            show_stations = st.multiselect(
                "选择台站",
                all_stations,
                default=all_stations
            )
            first_tr = display_streams[list(display_streams.keys())[0]]
            total_duration = first_tr.stats.npts * first_tr.stats.delta
            time_range = st.slider(
                "时间范围 (秒)",
                0.0,
                float(total_duration),
                (0.0, float(total_duration))
            )
            normalize = st.checkbox("归一化显示", value=True)
            fig_height = st.slider("图高度", 400, 1200, 700)
            
            st.markdown("---")
            st.subheader("台站排除管理")
            st.caption("排除质量差的台站，避免影响机制解")
            excluded = st.multiselect(
                "排除的台站",
                all_stations,
                default=st.session_state.excluded_stations,
                key="exclude_stations_wave"
            )
            if excluded != st.session_state.excluded_stations:
                st.session_state.excluded_stations = excluded
            if st.session_state.excluded_stations:
                st.warning(f"已排除 {len(st.session_state.excluded_stations)} 个台站")
        
        with col1:
            active_stations = [s for s in show_stations if s not in st.session_state.excluded_stations]
            selected_streams = {k: v for k, v in display_streams.items() if k in active_stations}
            if selected_streams:
                fig = plot_waveforms(
                    selected_streams,
                    time_range=time_range,
                    normalize=normalize,
                    p_picks=st.session_state.p_picks,
                    s_picks=st.session_state.s_picks,
                    height=fig_height
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("没有可显示的台站，请取消排除或选择台站")
        
        st.markdown("### 台站信息")
        st.dataframe(stations_info, use_container_width=True)

    elif page == "P/S波拾取与震源距离":
        st.header("🔍 P波/S波到时拾取")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("自动拾取设置")
            sta_window = st.slider("STA窗口 (秒)", 0.5, 5.0, 1.0, 0.5)
            lta_window = st.slider("LTA窗口 (秒)", 5.0, 30.0, 10.0, 1.0)
            threshold = st.slider("触发阈值", 2.0, 10.0, 4.0, 0.5)
            
            if st.button("🔄 自动拾取P/S波", type="primary"):
                p_picks, s_picks = auto_pick_ps(display_streams, sta_window, lta_window, threshold)
                st.session_state.p_picks = p_picks
                st.session_state.s_picks = s_picks
                st.success(f"自动拾取完成，共 {len(p_picks)} 个台站")
                st.rerun()
            
            st.markdown("---")
            st.subheader("人工校正")
            all_stations = list(display_streams.keys())
            
            default_idx = 0
            if st.session_state.selected_station and st.session_state.selected_station in all_stations:
                default_idx = all_stations.index(st.session_state.selected_station)
                st.info(f"📌 联动选中: **{st.session_state.selected_station}**")
            
            station_select = st.selectbox("选择台站", all_stations, index=default_idx)
            tr_sel = display_streams[station_select]
            dur_sel = tr_sel.stats.npts * tr_sel.stats.delta
            p_time = st.number_input(
                "P波到时 (秒)",
                0.0,
                float(dur_sel),
                float(st.session_state.p_picks.get(station_select, 10.0))
            )
            s_time = st.number_input(
                "S波到时 (秒)",
                0.0,
                float(dur_sel),
                float(st.session_state.s_picks.get(station_select, 20.0))
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("保存校正"):
                    st.session_state.p_picks[station_select] = p_time
                    st.session_state.s_picks[station_select] = s_time
                    st.success("已保存")
                    st.rerun()
            with col_b:
                if st.button("重置"):
                    st.session_state.p_picks = {}
                    st.session_state.s_picks = {}
                    st.session_state.polarities = []
                    st.success("已重置")
                    st.rerun()
            
            if st.session_state.excluded_stations:
                st.markdown("---")
                st.warning(f"⚠️ 已排除 {len(st.session_state.excluded_stations)} 个台站: {', '.join(st.session_state.excluded_stations)}")
            
            st.markdown("---")
            st.subheader("震源参数")
            vp = st.number_input("P波速度 (km/s)", 5.0, 10.0, 6.0, 0.5)
            vs = st.number_input("S波速度 (km/s)", 2.0, 5.0, 3.5, 0.5)
        
        with col1:
            fig = plot_waveforms(
                display_streams,
                p_picks=st.session_state.p_picks,
                s_picks=st.session_state.s_picks,
                height=800
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📐 震源距离估算")
        
        if st.session_state.p_picks and st.session_state.s_picks:
            results = []
            for station in display_streams.keys():
                if station in st.session_state.p_picks and station in st.session_state.s_picks:
                    dist = calculate_epicentral_distance(
                        st.session_state.p_picks[station],
                        st.session_state.s_picks[station],
                        vp, vs
                    )
                    results.append({
                        "台站": station,
                        "P波到时 (s)": round(st.session_state.p_picks[station], 3),
                        "S波到时 (s)": round(st.session_state.s_picks[station], 3),
                        "走时差 (s)": round(st.session_state.s_picks[station] - st.session_state.p_picks[station], 3),
                        "震源距 (km)": round(dist, 2),
                        "已排除": "是" if station in st.session_state.excluded_stations else "否"
                    })
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            active_count = len([r for r in results if r["已排除"] == "否"])
            if active_count >= 3:
                st.info(f"✅ 可用于后续定位的台站数: {active_count}（排除 {len(results) - active_count} 个）")
            else:
                st.warning(f"⚠️ 至少需要3个有效台站才能定位，当前有 {active_count} 个有效")
        else:
            st.info("请先进行P/S波拾取")

    elif page == "频谱分析":
        st.header("📈 频谱分析")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("分析设置")
            all_stations = list(display_streams.keys())
            
            default_idx = 0
            if st.session_state.selected_station and st.session_state.selected_station in all_stations:
                default_idx = all_stations.index(st.session_state.selected_station)
                st.info(f"📌 联动选中: **{st.session_state.selected_station}**")
            
            station_select = st.selectbox("选择台站", all_stations, index=default_idx)
            freq_range = st.slider("频率范围 (Hz)", 0.0, 20.0, (0.1, 10.0), 0.1)
            log_scale = st.checkbox("对数坐标 (Y轴)", value=True)
            window_type = st.selectbox(
                "窗函数",
                ["hann", "hamming", "blackman", "none"]
            )
            
            st.markdown("---")
            st.subheader("异常检测")
            detect_anomaly = st.checkbox("启用异常检测", value=True)
            anomaly_threshold = st.slider("异常阈值 (倍标准差)", 2.0, 5.0, 3.0, 0.5)
        
        with col1:
            tr = display_streams[station_select]
            freqs, spectrum, peak_freq, anomalies = compute_spectrum(
                tr, freq_range, window_type, detect_anomaly, anomaly_threshold
            )
            
            fig = plot_spectrum(
                freqs, spectrum, freq_range, peak_freq,
                anomalies, log_scale, station_select
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 频谱特征")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("主频 (Hz)", f"{peak_freq:.2f}")
        with col_b:
            st.metric("频带宽度 (Hz)", f"{freq_range[1] - freq_range[0]:.1f}")
        with col_c:
            st.metric("采样率 (Hz)", f"{display_streams[station_select].stats.sampling_rate:.1f}")
        with col_d:
            st.metric("异常数量", len(anomalies))
        
        if anomalies and detect_anomaly:
            st.markdown("### 🔴 检测到的频率异常")
            anomaly_df = {
                "频率 (Hz)": [f"{a[0]:.2f}" for a in anomalies],
                "幅值": [f"{a[1]:.2e}" for a in anomalies],
                "类型": ["高频异常" if a[0] > 5 else "低频异常" for a in anomalies]
            }
            import pandas as pd
            st.dataframe(pd.DataFrame(anomaly_df), use_container_width=True)

    elif page == "频谱-滤波对比":
        st.header("📊 频谱-滤波对比分析")
        st.caption("同一台站上对比原始波形/频谱与滤波后效果，直观查看主频、异常频段和波形变化")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("设置")
            station_select = st.selectbox("选择台站", list(streams.keys()))
            
            if st.session_state.selected_station and st.session_state.selected_station in streams:
                station_select = st.session_state.selected_station
                st.info(f"📌 联动选中: **{station_select}**")
            
            st.markdown("##### 滤波参数")
            filter_type = st.selectbox(
                "滤波类型",
                ["带通", "高通", "低通"],
                key="sf_filter_type"
            )
            
            if filter_type == "带通":
                lowcut = st.slider("低频截止 (Hz)", 0.01, 10.0, 1.0, 0.1, key="sf_low")
                highcut = st.slider("高频截止 (Hz)", 0.1, 20.0, 5.0, 0.1, key="sf_high")
                corners = st.slider("阶数", 2, 8, 4, key="sf_corners")
                filter_params = {"type": "bandpass", "freqmin": lowcut, "freqmax": highcut, "corners": corners}
            elif filter_type == "高通":
                freq = st.slider("截止频率 (Hz)", 0.01, 10.0, 1.0, 0.1, key="sf_hp_freq")
                corners = st.slider("阶数", 2, 8, 4, key="sf_hp_corners")
                filter_params = {"type": "highpass", "freq": freq, "corners": corners}
            else:
                freq = st.slider("截止频率 (Hz)", 0.1, 20.0, 5.0, 0.1, key="sf_lp_freq")
                corners = st.slider("阶数", 2, 8, 4, key="sf_lp_corners")
                filter_params = {"type": "lowpass", "freq": freq, "corners": corners}
            
            freq_range_spec = st.slider("频谱显示范围 (Hz)", 0.0, 20.0, (0.1, 10.0), 0.1, key="sf_freq_range")
            log_scale = st.checkbox("频谱对数坐标", value=True, key="sf_log")
        
        with col1:
            original = streams[station_select]
            filtered_single = apply_filter({station_select: original}, filter_params)
            filtered = filtered_single[station_select]
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            st.subheader("📐 波形对比")
            fig_wave = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     vertical_spacing=0.08,
                                     subplot_titles=("原始波形", f"{filter_type}滤波后波形"))
            
            t_orig = original.times()
            t_filt = filtered.times()
            
            max_orig = np.max(np.abs(original.data)) or 1.0
            max_filt = np.max(np.abs(filtered.data)) or 1.0
            
            fig_wave.add_trace(
                go.Scatter(x=t_orig, y=original.data / max_orig, mode='lines',
                          name='原始', line=dict(color='#1f77b4', width=1)),
                row=1, col=1
            )
            fig_wave.add_trace(
                go.Scatter(x=t_filt, y=filtered.data / max_filt, mode='lines',
                          name='滤波后', line=dict(color='#d62728', width=1)),
                row=2, col=1
            )
            
            if station_select in st.session_state.p_picks:
                fig_wave.add_vline(x=st.session_state.p_picks[station_select],
                                  line_dash="dash", line_color="green", line_width=1.5,
                                  annotation_text="P", annotation_font_color="green",
                                  row=1, col=1)
                fig_wave.add_vline(x=st.session_state.p_picks[station_select],
                                  line_dash="dash", line_color="green", line_width=1.5,
                                  annotation_text="P", annotation_font_color="green",
                                  row=2, col=1)
            
            fig_wave.update_layout(height=450, showlegend=False, template="plotly_white")
            fig_wave.update_xaxes(title_text="时间 (秒)", row=2, col=1)
            fig_wave.update_yaxes(title_text="归一化振幅", row=1, col=1)
            fig_wave.update_yaxes(title_text="归一化振幅", row=2, col=1)
            st.plotly_chart(fig_wave, use_container_width=True)
            
            st.subheader("📈 频谱对比")
            freqs_o, spec_o, peak_o, anom_o = compute_spectrum(original, freq_range_spec, "hann", True, 3.0)
            freqs_f, spec_f, peak_f, anom_f = compute_spectrum(filtered, freq_range_spec, "hann", True, 3.0)
            
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(x=freqs_o, y=spec_o, mode='lines',
                                         name='原始频谱', line=dict(color='#1f77b4', width=1.5)))
            fig_spec.add_trace(go.Scatter(x=freqs_f, y=spec_f, mode='lines',
                                         name='滤波后频谱', line=dict(color='#d62728', width=1.5)))
            fig_spec.add_vline(x=peak_o, line_dash="dot", line_color="#1f77b4",
                              annotation_text=f"原始主频 {peak_o:.2f}Hz",
                              annotation_font_color="#1f77b4")
            fig_spec.add_vline(x=peak_f, line_dash="dot", line_color="#d62728",
                              annotation_text=f"滤波主频 {peak_f:.2f}Hz",
                              annotation_font_color="#d62728")
            
            y_axis_type = "log" if log_scale else "linear"
            fig_spec.update_layout(
                title=f"{station_select} 频谱对比",
                xaxis_title="频率 (Hz)",
                yaxis_title="幅值",
                yaxis_type=y_axis_type,
                xaxis_range=list(freq_range_spec),
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig_spec, use_container_width=True)
            
            st.markdown("### 📊 对比指标")
            col_a, col_b, col_c, col_d, col_e = st.columns(5)
            with col_a:
                st.metric("原始主频", f"{peak_o:.2f} Hz")
            with col_b:
                st.metric("滤波后主频", f"{peak_f:.2f} Hz")
            with col_c:
                st.metric("原始异常数", len(anom_o))
            with col_d:
                st.metric("滤波后异常数", len(anom_f))
            with col_e:
                rms_change = np.sqrt(np.mean((np.array(original.data) - np.array(filtered.data))**2))
                st.metric("RMS差异", f"{rms_change:.4f}")
            
            if anom_o or anom_f:
                st.markdown("### 🔴 异常频段对比")
                comp_data = []
                orig_freqs = {round(a[0], 2): a[1] for a in anom_o}
                filt_freqs = {round(a[0], 2): a[1] for a in anom_f}
                all_freqs = set(list(orig_freqs.keys()) + list(filt_freqs.keys()))
                for f in sorted(all_freqs):
                    comp_data.append({
                        "频率 (Hz)": f"{f:.2f}",
                        "原始幅值": f"{orig_freqs.get(f, 0):.2e}" if f in orig_freqs else "-",
                        "滤波后幅值": f"{filt_freqs.get(f, 0):.2e}" if f in filt_freqs else "-",
                        "状态": "滤波后保留" if f in filt_freqs and f in orig_freqs else
                               ("滤波后新增" if f in filt_freqs else "滤波后消除")
                    })
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    elif page == "台站质量检查":
        st.header("🔍 台站质量检查 · 对照视图 + 批量复核")
        st.caption("选中台站后，波形、拾取、频谱和初动信息集中展示；批量复核按SNR+P拾取+初动质量给建议，一键应用，立即同步到机制解和报告")

        all_stations = list(display_streams.keys())

        col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])
        with col_sel1:
            default_idx = 0
            if st.session_state.selected_station and st.session_state.selected_station in all_stations:
                default_idx = all_stations.index(st.session_state.selected_station)
                st.info(f"📌 联动选中台站（来自地图或其他页面）: **{st.session_state.selected_station}**")

            station_check = st.selectbox(
                "选择要检查的台站",
                all_stations,
                index=default_idx,
                key="qc_station_select"
            )
            st.session_state.selected_station = station_check

        with col_sel2:
            is_excluded = station_check in st.session_state.excluded_stations
            if is_excluded:
                st.warning(f"⚠️ 该台站已被排除")
                if st.button("✅ 恢复该台站", use_container_width=True):
                    st.session_state.excluded_stations.remove(station_check)
                    clear_report_cache()
                    st.success(f"已恢复: {station_check}")
                    st.rerun()
            else:
                if st.button("🚫 排除该台站", use_container_width=True, type="secondary"):
                    st.session_state.excluded_stations.append(station_check)
                    st.session_state.review_notes[station_check] = {
                        "status": "exclude",
                        "reason": st.session_state.review_notes.get(station_check, {}).get("reason", "人工排除")
                    }
                    clear_report_cache()
                    st.warning(f"已排除: {station_check}，报告和机制解已同步更新")
                    st.rerun()

        with col_sel3:
            tr_qc = display_streams[station_check]
            snr_qc = float(abs(tr_qc.data).max() / (tr_qc.data.std() + 1e-10))
            if snr_qc > 5:
                st.success(f"SNR: {snr_qc:.1f}  质量优 🟢")
            elif snr_qc >= 2:
                st.warning(f"SNR: {snr_qc:.1f}  质量中 🟡")
            else:
                st.error(f"SNR: {snr_qc:.1f}  质量差 🔴")

        st.markdown("---")

        col_qc1, col_qc2 = st.columns([2, 1])

        with col_qc1:
            st.subheader(f"📐 波形 & 拾取 ({station_check})")

            t = np.array(tr_qc.times())
            d = np.array(tr_qc.data)

            import plotly.graph_objects as go
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=t, y=d, mode='lines', name=station_check,
                                     line=dict(color='#1f77b4', width=1.2)))

            p_qc = st.session_state.p_picks.get(station_check)
            s_qc = st.session_state.s_picks.get(station_check)
            if p_qc:
                fig_w.add_vline(x=p_qc, line_dash="dash", line_color="green", line_width=2,
                               annotation_text=f"P = {p_qc:.2f}s", annotation_font_color="green")
            if s_qc:
                fig_w.add_vline(x=s_qc, line_dash="dash", line_color="red", line_width=2,
                               annotation_text=f"S = {s_qc:.2f}s", annotation_font_color="red")

            fig_w.update_layout(
                xaxis_title="时间 (秒)",
                yaxis_title="振幅",
                height=320,
                margin=dict(l=10, r=10, t=10, b=40),
                showlegend=False,
                template="plotly_white"
            )
            st.plotly_chart(fig_w, use_container_width=True)

            st.subheader(f"📈 频谱 ({station_check})")
            freq_r = (0.1, 10.0)
            freqs_qc, spec_qc, peak_qc, anom_qc = compute_spectrum(
                tr_qc, freq_r, "hann", True, 3.0
            )
            fig_s = plot_spectrum(freqs_qc, spec_qc, freq_r, peak_qc,
                                  anom_qc, True, station_check)
            fig_s.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=40))
            st.plotly_chart(fig_s, use_container_width=True)

        with col_qc2:
            st.subheader("📋 台站详情 & 复核批注")

            st.markdown("##### 基础信息")
            det_cols1, det_cols2 = st.columns(2)
            with det_cols1:
                st.metric("采样率", f"{tr_qc.stats.sampling_rate:.0f} Hz")
                st.metric("数据点数", f"{tr_qc.stats.npts}")
                st.metric("主频", f"{peak_qc:.2f} Hz")
            with det_cols2:
                st.metric("P到时", f"{p_qc:.2f}s" if p_qc else "未拾取")
                st.metric("S到时", f"{s_qc:.2f}s" if s_qc else "未拾取")
                st.metric("异常数", f"{len(anom_qc)} 个")

            st.markdown("---")
            st.markdown("##### 初动极性")

            pol_info_qc = None
            for p in st.session_state.polarities:
                if p[0] == station_check:
                    pol_info_qc = p
                    break

            if pol_info_qc:
                pol_cn_qc = "压缩 (● 黑点)" if pol_info_qc[3] in ['compressional', 'C', 'c', '+', 1, 'up'] else "拉张 (○ 白圈)"
                qual_map_qc = {'good': '优 🟢', 'medium': '中 🟡', 'poor': '差 🔴',
                              'manual': '人工 ✏️'}
                qual_disp_qc = qual_map_qc.get(pol_info_qc[4], str(pol_info_qc[4]))
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    st.metric("极性", pol_cn_qc)
                with pc2:
                    st.metric("方位角", f"{float(pol_info_qc[1]):.1f}°")
                with pc3:
                    st.metric("倾角", f"{float(pol_info_qc[2]):.1f}°")
                st.info(f"初动质量: **{qual_disp_qc}**")
            else:
                st.info("⚠️ 该台站尚未提取初动极性")
                st.caption("请去「震源机制解 → 初动方向」页面点击自动提取")

            st.markdown("---")
            st.markdown("##### ✏️ 复核批注")
            cur_note = st.session_state.review_notes.get(station_check, {"status": "", "reason": ""})
            status_options = ["", "keep", "exclude", "review"]
            status_labels = {"": "— (未设置)", "keep": "✅ 保留",
                           "exclude": "🚫 排除", "review": "⚠️ 待复核"}
            cur_idx = status_options.index(cur_note.get("status", "")) if cur_note.get("status") in status_options else 0
            new_status = st.radio(
                "复核结论",
                status_options,
                index=cur_idx,
                format_func=lambda x: status_labels[x],
                key=f"review_status_{station_check}",
                horizontal=True
            )
            new_reason = st.text_input(
                "备注原因",
                value=cur_note.get("reason", ""),
                placeholder="例如：SNR太低、初动不可靠、波形异常...",
                key=f"review_reason_{station_check}"
            )

            col_rv1, col_rv2 = st.columns(2)
            with col_rv1:
                changed = (new_status != cur_note.get("status", "")) or (new_reason != cur_note.get("reason", ""))
                if st.button("💾 保存批注", disabled=not changed, use_container_width=True):
                    st.session_state.review_notes[station_check] = {
                        "status": new_status,
                        "reason": new_reason
                    }
                    if new_status == "exclude" and station_check not in st.session_state.excluded_stations:
                        st.session_state.excluded_stations.append(station_check)
                    elif new_status in ("keep", "review") and station_check in st.session_state.excluded_stations:
                        st.session_state.excluded_stations.remove(station_check)
                    clear_report_cache()
                    st.success("已保存批注，机制解和报告已同步更新")
                    st.rerun()
            with col_rv2:
                if st.button("🗑️ 清除批注", use_container_width=True):
                    if station_check in st.session_state.review_notes:
                        del st.session_state.review_notes[station_check]
                    clear_report_cache()
                    st.success("已清除该台站批注")
                    st.rerun()

            st.markdown("---")
            st.markdown("##### 机制解参与状态")
            if is_excluded:
                st.error("❌ 已排除 — 不会出现在沙滩球图和报告中")
            else:
                st.success("✅ 已参与 — 用于机制解计算")

            if len(st.session_state.excluded_stations) > 0:
                with st.expander(f"查看已排除的台站 ({len(st.session_state.excluded_stations)})"):
                    for ex in list(st.session_state.excluded_stations):
                        col_r1, col_r2 = st.columns([3, 1])
                        with col_r1:
                            note = st.session_state.review_notes.get(ex, {})
                            reason = note.get("reason", "")
                            disp = ex + (f"（{reason}）" if reason else "")
                            st.text(disp)
                        with col_r2:
                            if st.button("恢复", key=f"qc_restore_{ex}"):
                                st.session_state.excluded_stations.remove(ex)
                                clear_report_cache()
                                st.success(f"已恢复: {ex}")
                                st.rerun()

        st.markdown("---")
        st.subheader("📊 批量复核 · 自动建议")
        st.caption("根据SNR、P波拾取是否存在、初动质量三项指标自动给出复核建议，可一键应用再逐个调整")

        def compute_suggestion(stn):
            """基于SNR+P拾取+初动质量给出建议"""
            tr_s = display_streams[stn]
            snr = float(abs(tr_s.data).max() / (tr_s.data.std() + 1e-10))

            p_picked = stn in st.session_state.p_picks

            pol_q = None
            for p in st.session_state.polarities:
                if p[0] == stn:
                    pol_q = p[4] if len(p) > 4 else None
                    break

            score = 0
            reasons = []

            if snr > 5:
                score += 2
            elif snr >= 2:
                score += 1
            else:
                score -= 1
                reasons.append(f"SNR低({snr:.1f})")

            if p_picked:
                score += 1
            else:
                score -= 1
                reasons.append("无P波拾取")

            if pol_q == "good":
                score += 2
            elif pol_q == "medium":
                score += 1
            elif pol_q == "poor":
                score -= 1
                reasons.append("初动质量差")
            elif pol_q is None:
                score -= 0
                reasons.append("尚无初动提取")

            if score >= 3:
                status = "keep"
                if not reasons:
                    reasons.append("各项指标良好")
            elif score >= 1:
                status = "review"
                if not reasons:
                    reasons.append("指标中等，建议人工确认")
            else:
                status = "exclude"
                if not reasons:
                    reasons.append("综合指标较差")

            return status, "、".join(reasons[:3])

        col_ba1, col_ba2, col_ba3 = st.columns([1, 1, 1])
        with col_ba1:
            if st.button("🤖 生成自动建议", use_container_width=True, type="primary"):
                suggestions = {}
                for stn in all_stations:
                    s, r = compute_suggestion(stn)
                    suggestions[stn] = {"status": s, "reason": r,
                                       "auto": True}
                st.session_state["_qc_suggestions"] = suggestions
                st.success(f"已为 {len(all_stations)} 个台站生成建议")
                st.rerun()
        with col_ba2:
            if st.button("✅ 一键应用建议（保留/排除/待复核）", use_container_width=True):
                if "_qc_suggestions" in st.session_state:
                    applied = 0
                    for stn, s in st.session_state["_qc_suggestions"].items():
                        st.session_state.review_notes[stn] = {
                            "status": s["status"],
                            "reason": s["reason"]
                        }
                        if s["status"] == "exclude" and stn not in st.session_state.excluded_stations:
                            st.session_state.excluded_stations.append(stn)
                        elif s["status"] in ("keep", "review") and stn in st.session_state.excluded_stations:
                            st.session_state.excluded_stations.remove(stn)
                        applied += 1
                    clear_report_cache()
                    st.success(f"已应用 {applied} 个建议，机制解和报告已同步更新")
                    st.rerun()
                else:
                    st.info("请先生成建议")
        with col_ba3:
            if st.button("🧹 清除所有复核批注", use_container_width=True):
                st.session_state.review_notes = {}
                st.session_state.excluded_stations = []
                clear_report_cache()
                st.success("已清除所有批注和排除设置")
                st.rerun()

        batch_rows = []
        suggestions = st.session_state.get("_qc_suggestions", {})
        keep_t = excl_t = rev_t = 0
        for stn in all_stations:
            tr_b = display_streams[stn]
            snr_b = float(abs(tr_b.data).max() / (tr_b.data.std() + 1e-10))
            if snr_b > 5:
                qc = "优 🟢"
            elif snr_b >= 2:
                qc = "中 🟡"
            else:
                qc = "差 🔴"

            pol_b = "—"
            pol_q = "—"
            for p in st.session_state.polarities:
                if p[0] == stn:
                    pol_b = "压缩" if p[3] in ['compressional', 'C', 'c', '+', 1, 'up'] else "拉张"
                    pol_q = {'good': '优', 'medium': '中', 'poor': '差',
                            'manual': '人工'}.get(p[4] if len(p) > 4 else '-', '-')
                    break

            p_pick = "✅" if stn in st.session_state.p_picks else "❌"
            excl_b = "是" if stn in st.session_state.excluded_stations else "否"

            note = st.session_state.review_notes.get(stn, {})
            rev_st = note.get("status", "")
            rev_reason = note.get("reason", "")
            rev_label = {"keep": "✅ 保留", "exclude": "🚫 排除",
                        "review": "⚠️ 待复核", "": "—"}.get(rev_st, rev_st)
            if rev_st == "keep": keep_t += 1
            elif rev_st == "exclude": excl_t += 1
            elif rev_st == "review": rev_t += 1

            sug = suggestions.get(stn)
            sug_label = {"keep": "→保", "exclude": "→排",
                        "review": "→待"}.get(sug["status"], "") if sug else ""
            sug_reason = sug.get("reason", "") if sug else ""

            batch_rows.append({
                "台站": stn,
                "SNR": f"{snr_b:.1f}",
                "信号": qc,
                "P拾取": p_pick,
                "初动": pol_b,
                "初动质": pol_q,
                "建议": sug_label + ("("+sug_reason+")" if sug_reason else ""),
                "复核结论": rev_label,
                "备注": rev_reason,
                "已排除": excl_b,
            })

        st.info(f"当前复核状态：✅ 保留 {keep_t}  |  🚫 排除 {excl_t}  |  ⚠️ 待复核 {rev_t}  |  未设置 {len(all_stations) - keep_t - excl_t - rev_t}")
        st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)

    elif page == "震源机制解":
        st.header("🎯 震源机制解（沙滩球图）")
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("输入方式")
            input_method = st.radio(
                "选择输入方式",
                ["示例数据", "矩张量分量", "上传矩张量文件", "初动方向"],
                index=0,
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            current_mt = None
            current_mw = None
            
            if input_method == "示例数据":
                current_mt = generate_sample_moment_tensor()
                current_mw = calculate_moment_magnitude(current_mt)
                st.info("✅ 已加载示例矩张量数据")
                st.metric("矩震级 Mw", f"{current_mw:.2f}")
            
            elif input_method == "矩张量分量":
                st.markdown("##### 手动输入矩张量分量")
                st.caption("坐标系: 球坐标系 (r, θ, φ) → Mrr, Mtt, Mpp, Mrt, Mrp, Mtp")
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    mrr = st.number_input("Mrr", value=1.5e17, format="%.2e")
                    mtt = st.number_input("Mtt", value=-1.0e17, format="%.2e")
                    mpp = st.number_input("Mpp", value=-5.0e16, format="%.2e")
                with col_m2:
                    mrt = st.number_input("Mrt", value=8.0e16, format="%.2e")
                    mrp = st.number_input("Mrp", value=-3.0e16, format="%.2e")
                    mtp = st.number_input("Mtp", value=6.0e16, format="%.2e")
                
                current_mt = [mrr, mtt, mpp, mrt, mrp, mtp]
                current_mw = calculate_moment_magnitude(current_mt)
                
                if st.button("应用矩张量", type="primary"):
                    st.session_state.custom_mt = current_mt
                    st.success("矩张量已更新，沙滩球图已刷新")
                    st.rerun()
                
                if st.session_state.custom_mt is not None:
                    current_mt = st.session_state.custom_mt
                    current_mw = calculate_moment_magnitude(current_mt)
                st.metric("矩震级 Mw", f"{current_mw:.2f}")
            
            elif input_method == "上传矩张量文件":
                st.markdown("##### 上传矩张量文件")
                st.caption("支持格式: .txt, .csv, .json (6个分量: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)")
                st.caption("可参考项目目录下的 sample_moment_tensor.json / .txt")
                
                mt_file = st.file_uploader(
                    "选择矩张量文件",
                    type=["txt", "csv", "json", "dat"]
                )
                
                if mt_file is not None:
                    try:
                        mt_loaded, file_info = load_moment_tensor_file(mt_file)
                        current_mt = mt_loaded
                        current_mw = calculate_moment_magnitude(mt_loaded)
                        st.session_state.custom_mt = mt_loaded
                        st.session_state.mt_file_info = file_info
                        
                        st.success(f"✅ 文件加载成功 ({file_info.get('format', '未知格式')})")
                        st.metric("矩震级 Mw", f"{current_mw:.2f}")
                        
                        with st.expander("查看分量详情与文件信息"):
                            st.dataframe(
                                pd.DataFrame({
                                    "分量": ["Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"],
                                    "数值 (N·m)": [f"{v:.3e}" for v in mt_loaded]
                                }),
                                use_container_width=True, hide_index=True
                            )
                            if file_info:
                                for k, v in file_info.items():
                                    if k != 'format':
                                        st.text(f"{k}: {v}")
                    except Exception as e:
                        st.error(f"❌ 加载失败: {str(e)}")
                        st.info("文件应包含6个矩张量分量，空格或逗号分隔")
            
            elif input_method == "初动方向":
                st.markdown("##### P波初动极性")
                
                col_auto1, col_auto2 = st.columns(2)
                with col_auto1:
                    if st.button("🔄 从波形自动提取", type="primary"):
                        streams_for_pick = {
                            k: v for k, v in display_streams.items() 
                            if k not in st.session_state.excluded_stations
                        }
                        if not streams_for_pick:
                            st.error("所有台站已被排除，请取消排除")
                        else:
                            if not st.session_state.p_picks:
                                st.warning("⚠️ 请先在 P/S波拾取 页面完成P波拾取")
                            else:
                                polarities = extract_first_motions(
                                    streams_for_pick,
                                    st.session_state.p_picks,
                                    stations_info
                                )
                                st.session_state.polarities = polarities
                                st.success(f"✅ 提取 {len(polarities)} 个，立即见沙滩球图")
                                st.rerun()
                
                with col_auto2:
                    if st.button("🧹 清空全部"):
                        st.session_state.polarities = []
                        st.warning("已清空所有初动")
                        st.rerun()
                
                active_pols = [
                    p for p in st.session_state.polarities 
                    if p[0] not in st.session_state.excluded_stations
                ]
                excl_count = len(st.session_state.polarities) - len(active_pols)
                
                if active_pols:
                    info_text = f"✅ 有效台站: {len(active_pols)} 个"
                    if excl_count > 0:
                        info_text += f"，排除: {excl_count} 个"
                    st.success(info_text)
                elif st.session_state.polarities:
                    st.warning(f"⚠️ {excl_count} 个台站已全部排除")
                else:
                    st.info("💡 可直接在下方「手动添加台站」录入，无需先做自动提取")
                
                if st.session_state.excluded_stations:
                    st.caption(f"排除台站: {', '.join(st.session_state.excluded_stations)}")
                
                st.markdown("---")
                st.markdown("##### 📝 手动添加台站（无拾取时也可用）")
                
                default_new_name = f"STA{len(st.session_state.polarities) + 1:02d}"
                pol_names_exist = [p[0] for p in st.session_state.polarities]
                if default_new_name in pol_names_exist:
                    default_new_name = f"STA{len(st.session_state.polarities) + 10:02d}"
                
                new_station_name = st.text_input("台站名", default_new_name, key="fm_new_name")
                
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    add_az = st.number_input("方位角 (°)", 0.0, 360.0, 45.0, 5.0, key="fm_add_az")
                    add_pl = st.number_input("倾角 (°)", 0.0, 90.0, 30.0, 5.0, key="fm_add_pl")
                with col_a2:
                    add_pol = st.radio(
                        "初动极性",
                        ["compressional (压缩 ●)", "dilational (拉张 ○)"],
                        key="fm_add_pol",
                        horizontal=True
                    )
                
                add_pol_val = 'compressional' if 'compressional' in add_pol else 'dilational'
                
                if st.button("➕ 添加台站（添加后立即显示）", use_container_width=True):
                    new_entry = (new_station_name, float(add_az), float(add_pl), add_pol_val, 'manual')
                    st.session_state.polarities.append(new_entry)
                    st.success(f"已添加 {new_station_name} (方位{add_az:.0f}°, 倾{add_pl:.0f}°, {add_pol_val})")
                    st.rerun()
                
                if st.session_state.polarities:
                    st.markdown("---")
                    st.markdown("##### 🔧 校正/删除已有台站")
                    
                    edit_idx = st.selectbox(
                        "选择台站",
                        list(range(len(st.session_state.polarities))),
                        format_func=lambda i: f"{st.session_state.polarities[i][0]}  "
                                             f"(az={float(st.session_state.polarities[i][1]):.0f}°, "
                                             f"pl={float(st.session_state.polarities[i][2]):.0f}°, "
                                             f"{'●' if st.session_state.polarities[i][3] in ['compressional', 'C', 'c', '+', 1, 'up'] else '○'})",
                        key="fm_edit_idx"
                    )
                    
                    current_pol = st.session_state.polarities[edit_idx]
                    
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        edit_az = st.number_input(
                            "校正方位角 (°)", 0.0, 360.0,
                            float(current_pol[1]), 1.0, key="fm_edit_az"
                        )
                        edit_pl = st.number_input(
                            "校正倾角 (°)", 0.0, 90.0,
                            float(current_pol[2]), 1.0, key="fm_edit_pl"
                        )
                    with col_e2:
                        edit_pol_display = st.radio(
                            "校正极性",
                            ["compressional (压缩 ●)", "dilational (拉张 ○)"],
                            0 if current_pol[3] in ['compressional', 'C', 'c', '+', 1, 'up'] else 1,
                            key="fm_edit_pol", horizontal=True
                        )
                    
                    edit_pol_val = 'compressional' if 'compressional' in edit_pol_display else 'dilational'
                    qual = current_pol[4] if len(current_pol) > 4 else 'manual'
                    
                    changed = (abs(float(current_pol[1]) - edit_az) > 1e-3 or
                              abs(float(current_pol[2]) - edit_pl) > 1e-3 or
                              current_pol[3] != edit_pol_val)
                    
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        save_disabled = not changed
                        if st.button("💾 保存校正", disabled=save_disabled, use_container_width=True):
                            st.session_state.polarities[edit_idx] = (
                                current_pol[0], float(edit_az), float(edit_pl), edit_pol_val, qual
                            )
                            st.success(f"已更新 {current_pol[0]}")
                            st.rerun()
                    with col_b2:
                        if st.button("❌ 删除该台站", use_container_width=True):
                            removed = st.session_state.polarities.pop(edit_idx)
                            st.warning(f"已删除 {removed[0]}")
                            st.rerun()
            
            st.markdown("---")
            st.subheader("显示设置")
            show_axes = st.checkbox("显示主应力轴 (T/P轴)", value=True)
            show_nodal = st.checkbox("显示节线", value=True)
        
        with col1:
            active_polarities = [
                p for p in st.session_state.polarities 
                if p[0] not in st.session_state.excluded_stations
            ]
            
            if input_method in ["示例数据", "矩张量分量", "上传矩张量文件"]:
                mt_to_plot = None
                if input_method == "示例数据":
                    mt_to_plot = generate_sample_moment_tensor()
                elif input_method == "矩张量分量":
                    if current_mt is not None:
                        mt_to_plot = current_mt
                    elif st.session_state.custom_mt is not None:
                        mt_to_plot = st.session_state.custom_mt
                    else:
                        mt_to_plot = generate_sample_moment_tensor()
                elif input_method == "上传矩张量文件":
                    if current_mt is not None:
                        mt_to_plot = current_mt
                    elif st.session_state.custom_mt is not None:
                        mt_to_plot = st.session_state.custom_mt
                    else:
                        mt_to_plot = generate_sample_moment_tensor()
                
                if mt_to_plot is not None:
                    current_mt = mt_to_plot
                    current_mw = calculate_moment_magnitude(mt_to_plot)
                    try:
                        fig_beach = plot_beach_ball_moment_tensor(
                            mt_to_plot,
                            show_axes=show_axes,
                            size=6
                        )
                        st.pyplot(fig_beach, use_container_width=True)
                    except Exception as e:
                        st.error(f"绘制沙滩球图失败: {str(e)}")
            
            elif input_method == "初动方向":
                if active_polarities:
                    try:
                        fig_beach = plot_beach_ball_first_motion(
                            active_polarities,
                            size=6
                        )
                        st.pyplot(fig_beach, use_container_width=True)
                        
                        st.markdown("### 📋 初动极性列表（未排除）")
                        pol_df = first_motion_to_dataframe(active_polarities)
                        st.dataframe(pol_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"绘制沙滩球图失败: {str(e)}")
                else:
                    if st.session_state.polarities and len(st.session_state.polarities) > 0:
                        st.info("⚠️ 所有台站已被排除，请先取消排除后再查看")
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    circle = Circle((0, 0), 1, fill=True, facecolor='#f0f0f0',
                                   edgecolor='black', linewidth=2.5)
                    ax.add_patch(circle)
                    ax.plot([-1, 1], [0, 0], 'k-', linewidth=0.5, alpha=0.3)
                    ax.plot([0, 0], [-1, 1], 'k-', linewidth=0.5, alpha=0.3)
                    ax.text(0, 1.08, 'N', ha='center', fontsize=10, fontweight='bold')
                    
                    hint = '暂无初动数据\n左侧「手动添加台站」\n或「从波形自动提取」'
                    ax.text(0, -0.05, hint, 
                           ha='center', fontsize=11, color='gray')
                    ax.set_xlim(-1.25, 1.25)
                    ax.set_ylim(-1.25, 1.25)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    ax.set_title('震源机制解 (初动极性)', fontsize=13, fontweight='bold', pad=10)
                    st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        st.header("📄 事件报告 · 复核工作流")
        st.caption("先预览完整报告（按有效台站生成），确认无误后再导出 PDF 或打包下载；修改排除/复核后报告自动失效，需重新生成以保证图表一致")

        report_mt = current_mt if current_mt is not None else st.session_state.custom_mt
        report_mw = current_mw
        report_pols = active_polarities if input_method == "初动方向" else st.session_state.polarities
        report_all_pols = st.session_state.polarities
        report_file_info = st.session_state.mt_file_info
        report_excluded = list(st.session_state.excluded_stations)
        review_notes = st.session_state.review_notes

        col_top1, col_top2, col_top3, col_top4 = st.columns(4)
        with col_top1:
            st.metric("台站总数", len(report_all_pols))
        with col_top2:
            st.metric("有效台站", len(report_pols))
        with col_top3:
            st.metric("排除台站", len(report_excluded))
        with col_top4:
            k_ct = sum(1 for v in review_notes.values() if v.get('status') == 'keep')
            e_ct = sum(1 for v in review_notes.values() if v.get('status') == 'exclude')
            r_ct = sum(1 for v in review_notes.values() if v.get('status') == 'review')
            st.metric("复核结论", f"保{k_ct}/排{e_ct}/待{r_ct}")

        gen_btn = st.button("🔄 生成 / 刷新报告预览", type="primary", use_container_width=True)

        preview_key = "_report_preview_fig"

        if gen_btn or (report_mt is not None and preview_key not in st.session_state):
            try:
                with st.spinner("正在生成报告预览..."):
                    report_fig = create_complete_report_figure(
                        report_mt, report_all_pols, report_pols,
                        report_excluded, report_file_info, report_mw,
                        review_notes, (12, 16)
                    )
                    st.session_state[preview_key] = report_fig
                    st.success("✅ 报告已生成，当前有效台站、排除台站、复核批注均已同步")
            except Exception as e:
                import traceback
                st.error(f"生成报告失败: {e}\n{traceback.format_exc()}")
        
        if st.session_state.get(preview_key) is not None:
            st.subheader("📑 报告预览")
            st.pyplot(st.session_state[preview_key], use_container_width=True)
            
            st.markdown("---")
            st.subheader("💾 导出选项")
            
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            
            with col_d1:
                try:
                    bb_fig = None
                    if report_mt is not None:
                        bb_fig = plot_beach_ball_moment_tensor(report_mt, show_axes=True, size=6)
                    elif report_pols:
                        bb_fig = plot_beach_ball_first_motion(report_pols, size=6)
                    if bb_fig:
                        png_data = save_beachball_to_png(bb_fig)
                        st.download_button(
                            label="📥 沙滩球 PNG",
                            data=png_data,
                            file_name="beachball.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.info("无沙滩球图")
                except Exception as e:
                    st.error(f"PNG失败: {e}")
            
            with col_d2:
                try:
                    pdf_buf = save_report_to_pdf(
                        st.session_state[preview_key], bb_fig
                    )
                    st.download_button(
                        label="📥 完整报告 PDF",
                        data=pdf_buf,
                        file_name="event_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF失败: {e}")
            
            with col_d3:
                try:
                    csv_buf = create_first_motion_csv(
                        report_all_pols, report_mt, report_mw, report_file_info,
                        report_excluded, report_pols, review_notes
                    )
                    st.download_button(
                        label="📥 数据文件 CSV",
                        data=csv_buf.getvalue().encode('utf-8-sig'),
                        file_name="event_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"CSV失败: {e}")
            
            with col_d4:
                try:
                    zip_buf = create_report_zip_package(
                        st.session_state[preview_key], bb_fig,
                        report_all_pols, report_mt,
                        report_excluded, report_mw, report_file_info,
                        report_pols, review_notes
                    )
                    st.download_button(
                        label="📦 打包下载 ZIP",
                        data=zip_buf,
                        file_name="event_report_package.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"ZIP失败: {e}")
            
            with st.expander("查看导出文件清单"):
                st.markdown("""
| 导出方式 | 包含内容 |
|---------|---------|
| 沙滩球 PNG | 单独沙滩球高清图（按有效台站/当前矩张量生成） |
| 完整报告 PDF | 首页：标题/沙滩球/矩张量/来源/复核统计/初动表(含复核结论和原因)；次页：单独沙滩球图 |
| 数据文件 CSV | 概览段(总数/有效/排除/复核统计) + 排除台站段 + 矩张量6分量 + M0/Mw + 文件信息 + 初动极性表(含复核结论和原因) |
| 打包下载 ZIP | PDF + PNG + CSV + JSON + README_package.txt（含复核统计+每台站状态+原因） |
                """)
                if report_excluded:
                    st.warning(f"⚠️ 报告中已标记 {len(report_excluded)} 个排除台站（红色行），{r_ct} 个待复核（黄色行），{k_ct} 个保留（绿色行）")

    elif page == "滤波工具":
        st.header("🎛️ 滤波工具")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("滤波设置")
            filter_type = st.selectbox(
                "滤波类型",
                ["带通", "高通", "低通"]
            )
            
            if filter_type == "带通":
                lowcut = st.slider("低频截止 (Hz)", 0.01, 10.0, 1.0, 0.1)
                highcut = st.slider("高频截止 (Hz)", 0.1, 20.0, 5.0, 0.1)
                corners = st.slider("阶数", 2, 8, 4)
                filter_params = {"type": "bandpass", "freqmin": lowcut, "freqmax": highcut, "corners": corners}
            elif filter_type == "高通":
                freq = st.slider("截止频率 (Hz)", 0.01, 10.0, 1.0, 0.1)
                corners = st.slider("阶数", 2, 8, 4)
                filter_params = {"type": "highpass", "freq": freq, "corners": corners}
            else:
                freq = st.slider("截止频率 (Hz)", 0.1, 20.0, 5.0, 0.1)
                corners = st.slider("阶数", 2, 8, 4)
                filter_params = {"type": "lowpass", "freq": freq, "corners": corners}
            
            station_select = st.selectbox("预览台站", list(streams.keys()))
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("应用滤波", type="primary"):
                    filtered = apply_filter(streams, filter_params)
                    st.session_state.filtered_streams = filtered
                    st.success("滤波已应用到所有台站")
            with col_b:
                if st.button("重置"):
                    st.session_state.filtered_streams = None
                    st.success("已重置")
        
        with col1:
            st.subheader("实时预览")
            original = streams[station_select]
            if st.session_state.filtered_streams:
                filtered = st.session_state.filtered_streams[station_select]
            else:
                filtered = apply_filter({station_select: original}, filter_params)[station_select]
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                               subplot_titles=("原始波形", "滤波后波形"))
            
            t_original = original.times()
            t_filtered = filtered.times()
            
            fig.add_trace(
                go.Scatter(x=t_original, y=original.data, mode='lines',
                          name='原始', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=t_filtered, y=filtered.data, mode='lines',
                          name='滤波后', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(title_text="时间 (秒)", row=2, col=1)
            fig.update_yaxes(title_text="振幅", row=1, col=1)
            fig.update_yaxes(title_text="振幅", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 频谱对比")
            freqs_orig, spec_orig, _, _ = compute_spectrum(original, (0.1, 10.0), "hann", False, 3.0)
            freqs_filt, spec_filt, _, _ = compute_spectrum(filtered, (0.1, 10.0), "hann", False, 3.0)
            
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(x=freqs_orig, y=spec_orig, mode='lines',
                                         name='原始', line=dict(color='blue')))
            fig_spec.add_trace(go.Scatter(x=freqs_filt, y=spec_filt, mode='lines',
                                         name='滤波后', line=dict(color='red')))
            fig_spec.update_layout(
                title="频谱对比",
                xaxis_title="频率 (Hz)",
                yaxis_title="幅值",
                yaxis_type="log",
                height=400
            )
            st.plotly_chart(fig_spec, use_container_width=True)

    elif page == "台站地图":
        st.header("🗺️ 台站地图")
        st.caption("点击地图上的标记即可直接选台站，切到其他页面保持选中状态")

        station_names = list(display_streams.keys())
        quality_metrics = {}
        for name, tr in display_streams.items():
            snr = float(abs(tr.data).max() / (tr.data.std() + 1e-10))
            quality_metrics[name] = snr

        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("显示设置")
            map_style = st.selectbox(
                "地图样式",
                ["OpenStreetMap", "Stamen Terrain", "Stamen Toner"],
                key="map_style_key"
            )
            show_quality = st.checkbox("显示信号质量", value=True, key="map_show_q")
            marker_size = st.slider("标记大小", 5, 30, 12, key="map_marker_size")

            st.markdown("---")
            st.subheader("信号质量评估")

            st.info(f"台站总数: {len(display_streams)}")
            good_count = sum(1 for v in quality_metrics.values() if v > 5)
            medium_count = sum(1 for v in quality_metrics.values() if 2 <= v <= 5)
            poor_count = sum(1 for v in quality_metrics.values() if v < 2)
            st.success(f"高质量台站: {good_count}")
            st.warning(f"中等质量台站: {medium_count}")
            st.error(f"低质量台站: {poor_count}")

            st.markdown("---")
            st.subheader("当前选中")

            current_sel = st.session_state.get("selected_station")
            if current_sel and current_sel in station_names:
                st.success(f"📌 **{current_sel}**")
                st.caption("切到波形/拾取/频谱/质量检查页面保持选中")

                if current_sel in st.session_state.excluded_stations:
                    st.warning("⚠️ 此台站已被机制解排除")
                    if st.button("✅ 恢复该台站", key="map_restore_sel_btn"):
                        st.session_state.excluded_stations.remove(current_sel)
                        clear_report_cache()
                        st.success(f"已恢复: {current_sel}")
                        st.rerun()
                else:
                    if st.button("🚫 排除该台站", key="map_exclude_sel_btn"):
                        st.session_state.excluded_stations.append(current_sel)
                        clear_report_cache()
                        st.warning(f"已排除: {current_sel}")
                        st.rerun()
            else:
                st.info("👈 请点击左侧地图标记选择台站")

            if st.session_state.excluded_stations:
                st.markdown("---")
                with st.expander(f"已排除台站 ({len(st.session_state.excluded_stations)})", expanded=True):
                    for ex in list(st.session_state.excluded_stations):
                        col_ex1, col_ex2 = st.columns([3, 1])
                        with col_ex1:
                            st.text(ex)
                        with col_ex2:
                            if st.button("恢复", key=f"map_restore_{ex}"):
                                st.session_state.excluded_stations.remove(ex)
                                if "_report_preview_fig" in st.session_state:
                                    del st.session_state["_report_preview_fig"]
                                st.success(f"已恢复: {ex}")
                                st.rerun()

        with col1:
            try:
                m = create_station_map(
                    stations_info, quality_metrics,
                    map_style, show_quality, marker_size
                )
                from streamlit_folium import st_folium
                map_ret = st_folium(m, width=800, height=520, key="station_map_interactive")

                clicked_station = parse_clicked_station(map_ret, station_names)

                if clicked_station and clicked_station in station_names:
                    if clicked_station != st.session_state.get("selected_station"):
                        st.session_state.selected_station = clicked_station
                        clear_report_cache()
                        st.rerun()
            except Exception as e:
                st.error(f"地图加载失败: {str(e)}")
                st.info("请确保已安装 folium 和 streamlit-folium 库")

        st.markdown("---")

        sel_st = st.session_state.get("selected_station")
        if sel_st and sel_st in display_streams:
            st.subheader(f"📊 台站 {sel_st} 联动详情")

            tr_sel = display_streams[sel_st]

            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                st.metric("信噪比 (SNR)", f"{quality_metrics.get(sel_st, 0):.2f}")
            with col_d2:
                st.metric("采样率", f"{tr_sel.stats.sampling_rate:.1f} Hz")
            with col_d3:
                p_time = st.session_state.p_picks.get(sel_st, None)
                st.metric("P波到时", f"{p_time:.2f}s" if p_time else "未拾取")
            with col_d4:
                s_time = st.session_state.s_picks.get(sel_st, None)
                st.metric("S波到时", f"{s_time:.2f}s" if s_time else "未拾取")

            if sel_st in st.session_state.excluded_stations:
                st.error("⚠️ 该台站已被排除（不会参与机制解计算和报告）")
            else:
                if quality_metrics.get(sel_st, 0) > 5:
                    st.success("🟢 信号质量优，推荐用于机制解")
                elif quality_metrics.get(sel_st, 0) > 2:
                    st.warning("🟡 信号质量中，可用于机制解")
                else:
                    st.error("🔴 信号质量差，建议排除")

            import plotly.graph_objects as go
            t = np.array(tr_sel.times())
            data_sel = np.array(tr_sel.data)

            fig_sel = go.Figure()
            fig_sel.add_trace(go.Scatter(x=t, y=data_sel, mode='lines',
                                         name=sel_st, line=dict(color='#1f77b4', width=1)))

            if p_time:
                fig_sel.add_vline(x=p_time, line_dash="dash", line_color="green", line_width=2,
                                  annotation_text=f"P={p_time:.2f}s", annotation_font_color="green")
            if s_time:
                fig_sel.add_vline(x=s_time, line_dash="dash", line_color="red", line_width=2,
                                  annotation_text=f"S={s_time:.2f}s", annotation_font_color="red")

            fig_sel.update_layout(
                title=f"{sel_st} 波形（带拾取标记）",
                xaxis_title="时间 (秒)",
                yaxis_title="振幅",
                height=280,
                margin=dict(l=10, r=10, t=40, b=40),
                template="plotly_white"
            )
            st.plotly_chart(fig_sel, use_container_width=True)

            pol_info = None
            for p in st.session_state.polarities:
                if p[0] == sel_st:
                    pol_info = p
                    break

            if pol_info:
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                with col_p1:
                    pol_cn = "压缩 (●)" if pol_info[3] in ['compressional', 'C', 'c', '+', 1, 'up'] else "拉张 (○)"
                    st.metric("初动极性", pol_cn)
                with col_p2:
                    st.metric("方位角", f"{float(pol_info[1]):.1f}°")
                with col_p3:
                    st.metric("倾角", f"{float(pol_info[2]):.1f}°")
                with col_p4:
                    qual_map = {'good': '优 🟢', 'medium': '中 🟡', 'poor': '差 🔴', 'manual': '人工 ✏️'}
                    q = pol_info[4] if len(pol_info) > 4 else '-'
                    st.metric("质量", qual_map.get(q, str(q)))
                st.info("💡 可到「震源机制解 → 初动方向」页面校正此台站的极性")
            else:
                st.info("💡 该台站尚未提取初动极性，请到「震源机制解 → 初动方向」页面提取或手动添加")

        st.markdown("---")
        st.subheader("📊 全部台站详情")

        details = []
        for name, info in stations_info.iterrows():
            snr = quality_metrics.get(name, 0)
            if snr > 5:
                quality = "优"
                quality_color = "🟢"
            elif snr >= 2:
                quality = "中"
                quality_color = "🟡"
            else:
                quality = "差"
                quality_color = "🔴"

            excluded = "是" if name in st.session_state.excluded_stations else "否"

            pol_marker = "—"
            for p in st.session_state.polarities:
                if p[0] == name:
                    pol_marker = "● 压缩" if p[3] in ['compressional', 'C', 'c', '+', 1, 'up'] else "○ 拉张"
                    break

            is_sel = "✅" if name == sel_st else ""

            details.append({
                "选中": is_sel,
                "台站": name,
                "纬度": info["latitude"],
                "经度": info["longitude"],
                "信噪比": round(snr, 2),
                "信号质量": f"{quality_color} {quality}",
                "初动极性": pol_marker,
                "已排除": excluded
            })

        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
