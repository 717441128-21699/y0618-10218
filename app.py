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
    create_event_report_json
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
                    st.success("矩张量已更新")
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
                
                if not st.session_state.p_picks:
                    st.warning("⚠️ 请先在 'P/S波拾取' 页面进行P波到时拾取")
                else:
                    if st.button("🔄 从波形自动提取初动极性", type="primary"):
                        streams_for_pick = {
                            k: v for k, v in display_streams.items() 
                            if k not in st.session_state.excluded_stations
                        }
                        if not streams_for_pick:
                            st.error("所有台站已被排除，请取消排除")
                        else:
                            polarities = extract_first_motions(
                                streams_for_pick,
                                st.session_state.p_picks,
                                stations_info
                            )
                            st.session_state.polarities = polarities
                            st.success(f"✅ 成功提取 {len(polarities)} 个台站的初动极性，沙滩球图已更新")
                            st.rerun()
                    
                    if st.session_state.polarities:
                        active_pols = [
                            p for p in st.session_state.polarities 
                            if p[0] not in st.session_state.excluded_stations
                        ]
                        st.info(f"当前初动数: {len(active_pols)}（已排除 {len(st.session_state.polarities) - len(active_pols)} 个台站）")
                        
                        if st.session_state.excluded_stations:
                            st.warning(f"排除台站: {', '.join(st.session_state.excluded_stations)}")
                        
                        with st.expander("人工校正初动极性", expanded=True):
                            st.markdown("##### 选择台站校正")
                            
                            pol_names = [p[0] for p in st.session_state.polarities]
                            if pol_names:
                                edit_station = st.selectbox(
                                    "选择台站",
                                    pol_names,
                                    key="edit_station_pol"
                                )
                                
                                pol_idx = pol_names.index(edit_station)
                                current_pol = st.session_state.polarities[pol_idx]
                                
                                col_p1, col_p2 = st.columns(2)
                                with col_p1:
                                    new_az = st.number_input(
                                        "方位角 (°)",
                                        0.0, 360.0,
                                        float(current_pol[1]),
                                        1.0
                                    )
                                    new_pl = st.number_input(
                                        "倾角 (°)",
                                        0.0, 90.0,
                                        float(current_pol[2]),
                                        1.0
                                    )
                                with col_p2:
                                    new_pol = st.selectbox(
                                        "极性",
                                        ["compressional (压缩)", "dilational (拉张)"],
                                        0 if current_pol[3] in ['compressional', 'C', 'c', '+', 1, 'up'] else 1
                                    )
                                
                                new_pol_val = 'compressional' if 'compressional' in new_pol else 'dilational'
                                
                                col_b1, col_b2 = st.columns(2)
                                with col_b1:
                                    if st.button("保存修改"):
                                        quality = current_pol[4] if len(current_pol) > 4 else 'manual'
                                        st.session_state.polarities[pol_idx] = (
                                            edit_station, new_az, new_pl, new_pol_val, quality
                                        )
                                        st.success("已保存修改")
                                        st.rerun()
                                with col_b2:
                                    if st.button("删除该台站"):
                                        st.session_state.polarities.pop(pol_idx)
                                        st.success("已删除")
                                        st.rerun()
                                
                                st.markdown("---")
                                st.markdown("##### 手动添加台站")
                                new_station_name = st.text_input("台站名", "NEW01")
                                
                                col_a1, col_a2 = st.columns(2)
                                with col_a1:
                                    add_az = st.number_input("方位角", 0.0, 360.0, 45.0, key="add_az")
                                    add_pl = st.number_input("倾角", 0.0, 90.0, 30.0, key="add_pl")
                                with col_a2:
                                    add_pol = st.selectbox(
                                        "极性",
                                        ["compressional (压缩)", "dilational (拉张)"],
                                        key="add_pol"
                                    )
                                
                                add_pol_val = 'compressional' if 'compressional' in add_pol else 'dilational'
                                
                                col_btn1, col_btn2 = st.columns(2)
                                with col_btn1:
                                    if st.button("添加台站"):
                                        st.session_state.polarities.append(
                                            (new_station_name, float(add_az), float(add_pl), add_pol_val, 'manual')
                                        )
                                        st.success("已添加")
                                        st.rerun()
                                with col_btn2:
                                    if st.button("清空所有初动"):
                                        st.session_state.polarities = []
                                        st.warning("已清空所有初动")
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
                        
                        st.markdown("### 📋 初动极性列表")
                        pol_df = first_motion_to_dataframe(active_polarities)
                        st.dataframe(pol_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"绘制沙滩球图失败: {str(e)}")
                else:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    circle = Circle((0, 0), 1, fill=True, facecolor='#f0f0f0',
                                   edgecolor='black', linewidth=2.5)
                    ax.add_patch(circle)
                    ax.plot([-1, 1], [0, 0], 'k-', linewidth=0.5, alpha=0.3)
                    ax.plot([0, 0], [-1, 1], 'k-', linewidth=0.5, alpha=0.3)
                    ax.text(0, 1.08, 'N', ha='center', fontsize=10, fontweight='bold')
                    ax.text(0, -0.2, '暂无初动数据\n请从左侧提取或添加', 
                           ha='center', fontsize=12, color='gray')
                    ax.set_xlim(-1.25, 1.25)
                    ax.set_ylim(-1.25, 1.25)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    ax.set_title('震源机制解 (初动极性)', fontsize=13, fontweight='bold', pad=10)
                    st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        st.header("📄 事件报告导出")
        
        with st.expander("生成并下载事件报告", expanded=False):
            report_mt = current_mt if current_mt is not None else st.session_state.custom_mt
            report_mw = current_mw
            report_pols = active_polarities if input_method == "初动方向" else st.session_state.polarities
            report_file_info = st.session_state.mt_file_info
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                st.metric("矩震级 Mw", f"{report_mw:.2f}" if report_mw else "N/A")
            with col_r2:
                st.metric("初动台站数", len(report_pols))
            with col_r3:
                st.metric("排除台站数", len(st.session_state.excluded_stations))
            with col_r4:
                st.metric("文件来源", st.session_state.mt_file_info.get('format', '手动') if st.session_state.mt_file_info else '手动')
            
            st.markdown("##### 下载选项")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                if report_mt is not None:
                    try:
                        fig_report = plot_beach_ball_moment_tensor(report_mt, show_axes=True, size=6)
                        png_buf = save_beachball_to_png(fig_report)
                        st.download_button(
                            label="📥 下载沙滩球图 (PNG)",
                            data=png_buf,
                            file_name="beachball.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"PNG导出失败: {e}")
                elif report_pols:
                    try:
                        fig_report = plot_beach_ball_first_motion(report_pols, size=6)
                        png_buf = save_beachball_to_png(fig_report)
                        st.download_button(
                            label="📥 下载沙滩球图 (PNG)",
                            data=png_buf,
                            file_name="beachball_first_motion.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"PNG导出失败: {e}")
                else:
                    st.info("无可用沙滩球图")
            
            with col_dl2:
                try:
                    csv_buf = create_first_motion_csv(
                        report_pols, report_mt, report_mw, report_file_info
                    )
                    st.download_button(
                        label="📥 下载报告 (CSV)",
                        data=csv_buf.getvalue().encode('utf-8-sig'),
                        file_name="event_report.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"CSV导出失败: {e}")
            
            with col_dl3:
                try:
                    json_str = create_event_report_json(
                        report_mt, report_pols, report_mw, report_file_info
                    )
                    st.download_button(
                        label="📥 下载报告 (JSON)",
                        data=json_str.encode('utf-8'),
                        file_name="event_report.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"JSON导出失败: {e}")
            
            if report_mt is not None:
                st.markdown("##### 矩张量分量")
                st.dataframe(
                    pd.DataFrame({
                        "分量": ["Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"],
                        "数值 (N·m)": [f"{v:.4e}" for v in np.array(report_mt)],
                        "标量矩 M0": f"{calculate_scalar_moment(report_mt):.4e}",
                        "矩震级 Mw": f"{calculate_moment_magnitude(report_mt):.2f}"
                    }).head(6),
                    use_container_width=True, hide_index=True
                )

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
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("显示设置")
            map_style = st.selectbox(
                "地图样式",
                ["OpenStreetMap", "Stamen Terrain", "Stamen Toner"]
            )
            show_quality = st.checkbox("显示信号质量", value=True)
            marker_size = st.slider("标记大小", 5, 30, 12)
            
            st.markdown("---")
            st.subheader("信号质量评估")
            
            quality_metrics = {}
            for name, tr in display_streams.items():
                snr = float(abs(tr.data).max() / (tr.data.std() + 1e-10))
                quality_metrics[name] = snr
            
            st.info(f"台站总数: {len(display_streams)}")
            good_count = sum(1 for v in quality_metrics.values() if v > 5)
            medium_count = sum(1 for v in quality_metrics.values() if 2 <= v <= 5)
            poor_count = sum(1 for v in quality_metrics.values() if v < 2)
            st.success(f"高质量台站: {good_count}")
            st.warning(f"中等质量台站: {medium_count}")
            st.error(f"低质量台站: {poor_count}")
            
            st.markdown("---")
            st.subheader("台站选择联动")
            station_names = list(display_streams.keys())
            
            map_selected = None
            map_data = st.session_state.get("_map_data", None)
            if map_data and map_data.get("last_object_clicked_tooltip"):
                tooltip = map_data["last_object_clicked_tooltip"]
                for sn in station_names:
                    if sn in tooltip:
                        map_selected = sn
                        break
            
            if map_selected:
                st.session_state.selected_station = map_selected
            
            select_options = ["（从地图点击选择）"] + station_names
            current_idx = 0
            if st.session_state.selected_station and st.session_state.selected_station in station_names:
                current_idx = station_names.index(st.session_state.selected_station) + 1
            
            selected = st.selectbox(
                "选择台站查看详情",
                select_options,
                index=current_idx,
                key="map_station_select"
            )
            
            if selected != "（从地图点击选择）":
                st.session_state.selected_station = selected
            
            if st.session_state.selected_station and st.session_state.selected_station in station_names:
                st.success(f"📌 已选中: **{st.session_state.selected_station}**")
                
                if st.button("排除该台站（从机制解中排除）"):
                    if st.session_state.selected_station not in st.session_state.excluded_stations:
                        st.session_state.excluded_stations.append(st.session_state.selected_station)
                        st.warning(f"已排除: {st.session_state.selected_station}")
                        st.rerun()
                
                if st.session_state.excluded_stations:
                    with st.expander(f"已排除台站 ({len(st.session_state.excluded_stations)})"):
                        for ex in list(st.session_state.excluded_stations):
                            col_ex1, col_ex2 = st.columns([3, 1])
                            with col_ex1:
                                st.text(ex)
                            with col_ex2:
                                if st.button("恢复", key=f"restore_{ex}"):
                                    st.session_state.excluded_stations.remove(ex)
                                    st.rerun()
        
        with col1:
            try:
                m = create_station_map(
                    stations_info, quality_metrics,
                    map_style, show_quality, marker_size
                )
                from streamlit_folium import st_folium
                map_ret = st_folium(m, width=800, height=500)
                if map_ret:
                    st.session_state["_map_data"] = map_ret
            except Exception as e:
                st.error(f"地图加载失败: {str(e)}")
                st.info("请确保已安装 folium 和 streamlit-folium 库")
        
        st.markdown("---")
        
        if st.session_state.selected_station and st.session_state.selected_station in display_streams:
            sel_st = st.session_state.selected_station
            st.subheader(f"📊 台站 {sel_st} 联动详情")
            
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                st.metric("信噪比 (SNR)", f"{quality_metrics.get(sel_st, 0):.2f}")
            with col_d2:
                tr_sel = display_streams[sel_st]
                st.metric("采样率", f"{tr_sel.stats.sampling_rate:.1f} Hz")
            with col_d3:
                p_time = st.session_state.p_picks.get(sel_st, None)
                st.metric("P波到时", f"{p_time:.2f}s" if p_time else "未拾取")
            with col_d4:
                s_time = st.session_state.s_picks.get(sel_st, None)
                st.metric("S波到时", f"{s_time:.2f}s" if s_time else "未拾取")
            
            quality_rec = "推荐用于机制解" if quality_metrics.get(sel_st, 0) > 2 else "建议排除"
            if sel_st in st.session_state.excluded_stations:
                st.error(f"⚠️ 该台站已被排除（不会参与机制解计算）")
            else:
                if quality_metrics.get(sel_st, 0) > 5:
                    st.success(f"🟢 信号质量优，{quality_rec}")
                elif quality_metrics.get(sel_st, 0) > 2:
                    st.warning(f"🟡 信号质量中，{quality_rec}")
                else:
                    st.error(f"🔴 信号质量差，建议排除")
            
            import plotly.graph_objects as go
            tr_sel = display_streams[sel_st]
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
                height=300,
                template="plotly_white"
            )
            st.plotly_chart(fig_sel, use_container_width=True)
            
            st.session_state.polarities
            pol_info = None
            for p in st.session_state.polarities:
                if p[0] == sel_st:
                    pol_info = p
                    break
            
            if pol_info:
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    pol_cn = "压缩 (C)" if pol_info[3] in ['compressional', 'C', 'c', '+', 1, 'up'] else "拉张 (D)"
                    st.metric("初动极性", pol_cn)
                with col_p2:
                    st.metric("方位角", f"{pol_info[1]:.1f}°")
                with col_p3:
                    st.metric("倾角", f"{pol_info[2]:.1f}°")
                st.info("💡 可到「震源机制解 → 初动方向」页面校正此台站的极性")
            else:
                st.info("💡 该台站尚未提取初动极性，请到「震源机制解 → 初动方向」页面提取")
        
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
            
            details.append({
                "台站": name,
                "纬度": info["latitude"],
                "经度": info["longitude"],
                "信噪比": round(snr, 2),
                "信号质量": f"{quality_color} {quality}",
                "已排除": excluded
            })
        
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
