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
    calculate_moment_magnitude
)
from utils.filter import apply_filter
from utils.station_map import create_station_map

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

st.title("🌍 地震波形数据分析工具")
st.markdown("---")

if "streams" not in st.session_state:
    st.session_state.streams = None
    st.session_state.stations_info = None
    st.session_state.p_picks = {}
    st.session_state.s_picks = {}
    st.session_state.filtered_streams = None

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
            show_stations = st.multiselect(
                "选择台站",
                list(display_streams.keys()),
                default=list(display_streams.keys())
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
        
        with col1:
            selected_streams = {k: v for k, v in display_streams.items() if k in show_stations}
            fig = plot_waveforms(
                selected_streams,
                time_range=time_range,
                normalize=normalize,
                p_picks=st.session_state.p_picks,
                s_picks=st.session_state.s_picks,
                height=fig_height
            )
            st.plotly_chart(fig, use_container_width=True)
        
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
            
            if st.button("自动拾取P/S波", type="primary"):
                p_picks, s_picks = auto_pick_ps(display_streams, sta_window, lta_window, threshold)
                st.session_state.p_picks = p_picks
                st.session_state.s_picks = s_picks
                st.success("自动拾取完成")
            
            st.markdown("---")
            st.subheader("人工校正")
            station_select = st.selectbox("选择台站", list(display_streams.keys()))
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
            with col_b:
                if st.button("重置"):
                    st.session_state.p_picks = {}
                    st.session_state.s_picks = {}
                    st.success("已重置")
            
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
                        "震源距 (km)": round(dist, 2)
                    })
            
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            if len(results) >= 3:
                st.info(f"✅ 可用于后续定位的台站数: {len(results)}")
            else:
                st.warning(f"⚠️ 至少需要3个台站才能进行震中定位，当前有 {len(results)} 个")
        else:
            st.info("请先进行P/S波拾取")

    elif page == "频谱分析":
        st.header("📈 频谱分析")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("分析设置")
            station_select = st.selectbox("选择台站", list(display_streams.keys()))
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

    elif page == "震源机制解":
        st.header("🎯 震源机制解（沙滩球图）")
        
        if "polarities" not in st.session_state:
            st.session_state.polarities = []
        if "custom_mt" not in st.session_state:
            st.session_state.custom_mt = None
        if "mt_file_info" not in st.session_state:
            st.session_state.mt_file_info = None
        
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
            
            if input_method == "示例数据":
                current_mt = generate_sample_moment_tensor()
                st.info("✅ 已加载示例矩张量数据")
                
                m0 = calculate_scalar_moment(current_mt)
                mw = calculate_moment_magnitude(current_mt)
                st.metric("矩震级 Mw", f"{mw:.2f}")
            
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
                
                if st.button("应用矩张量", type="primary"):
                    st.session_state.custom_mt = current_mt
                    st.success("矩张量已更新")
            
            elif input_method == "上传矩张量文件":
                st.markdown("##### 上传矩张量文件")
                st.caption("支持格式: .txt, .csv, .json (6个分量: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)")
                
                mt_file = st.file_uploader(
                    "选择矩张量文件",
                    type=["txt", "csv", "json", "dat"]
                )
                
                if mt_file is not None:
                    try:
                        mt_loaded, file_info = load_moment_tensor_file(mt_file)
                        current_mt = mt_loaded
                        st.session_state.custom_mt = mt_loaded
                        st.session_state.mt_file_info = file_info
                        
                        st.success(f"✅ 文件加载成功 ({file_info.get('format', '未知格式')})")
                        
                        mw = calculate_moment_magnitude(mt_loaded)
                        st.metric("矩震级 Mw", f"{mw:.2f}")
                        
                        with st.expander("查看分量详情"):
                            st.dataframe(
                                pd.DataFrame({
                                    "分量": ["Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"],
                                    "数值": [f"{v:.3e}" for v in mt_loaded]
                                }),
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"❌ 加载失败: {str(e)}")
                        st.info("文件应包含6个矩张量分量，空格或逗号分隔")
            
            elif input_method == "初动方向":
                st.markdown("##### P波初动极性")
                
                if not st.session_state.p_picks:
                    st.warning("⚠️ 请先在 'P/S波拾取' 页面进行P波到时拾取")
                else:
                    if st.button("从波形自动提取初动极性", type="primary"):
                        polarities = extract_first_motions(
                            display_streams, 
                            st.session_state.p_picks,
                            stations_info
                        )
                        st.session_state.polarities = polarities
                        st.success(f"✅ 成功提取 {len(polarities)} 个台站的初动极性")
                    
                    if st.session_state.polarities:
                        st.info(f"当前初动数: {len(st.session_state.polarities)}")
                        
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
                                with col_b2:
                                    if st.button("删除该台站"):
                                        st.session_state.polarities.pop(pol_idx)
                                        st.success("已删除")
                                
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
                                
                                if st.button("添加台站"):
                                    st.session_state.polarities.append(
                                        (new_station_name, add_az, add_pl, add_pol_val, 'manual')
                                    )
                                    st.success("已添加")
                                
                                if st.button("清空所有初动"):
                                    st.session_state.polarities = []
                                    st.warning("已清空所有初动")
            
            st.markdown("---")
            st.subheader("显示设置")
            show_axes = st.checkbox("显示主应力轴 (T/P轴)", value=True)
            show_nodal = st.checkbox("显示节线", value=True)
        
        with col1:
            if input_method in ["示例数据", "矩张量分量", "上传矩张量文件"]:
                mt_to_plot = None
                if input_method == "示例数据":
                    mt_to_plot = generate_sample_moment_tensor()
                elif input_method == "矩张量分量":
                    if 'current_mt' in locals() and current_mt is not None:
                        mt_to_plot = current_mt
                    elif st.session_state.custom_mt is not None:
                        mt_to_plot = st.session_state.custom_mt
                    else:
                        mt_to_plot = generate_sample_moment_tensor()
                elif input_method == "上传矩张量文件":
                    if st.session_state.custom_mt is not None:
                        mt_to_plot = st.session_state.custom_mt
                    else:
                        mt_to_plot = generate_sample_moment_tensor()
                
                if mt_to_plot is not None:
                    try:
                        fig_beach = plot_beach_ball_moment_tensor(
                            mt_to_plot,
                            show_axes=show_axes,
                            size=6
                        )
                        st.pyplot(fig_beach, use_container_width=True)
                    except Exception as e:
                        st.error(f"绘制沙滩球图失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.info("请输入或上传矩张量数据")
            
            elif input_method == "初动方向":
                if st.session_state.polarities and len(st.session_state.polarities) > 0:
                    try:
                        fig_beach = plot_beach_ball_first_motion(
                            st.session_state.polarities,
                            size=6
                        )
                        st.pyplot(fig_beach, use_container_width=True)
                        
                        st.markdown("### 📋 初动极性列表")
                        pol_df = first_motion_to_dataframe(st.session_state.polarities)
                        st.dataframe(pol_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"绘制沙滩球图失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.info("👈 请在左侧点击「从波形自动提取初动极性」，或手动添加台站")
                    
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
            st.success(f"高质量台站: {good_count}")
            medium_count = sum(1 for v in quality_metrics.values() if 2 <= v <= 5)
            st.warning(f"中等质量台站: {medium_count}")
            poor_count = sum(1 for v in quality_metrics.values() if v < 2)
            st.error(f"低质量台站: {poor_count}")
        
        with col1:
            try:
                m = create_station_map(
                    stations_info, quality_metrics,
                    map_style, show_quality, marker_size
                )
                from streamlit_folium import st_folium
                st_folium(m, width=800, height=600)
            except Exception as e:
                st.error(f"地图加载失败: {str(e)}")
                st.info("请确保已安装 folium 和 streamlit-folium 库")
        
        st.markdown("---")
        st.subheader("📊 台站详情")
        
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
            
            details.append({
                "台站": name,
                "纬度": info["latitude"],
                "经度": info["longitude"],
                "信噪比 (dB)": round(snr, 2),
                "信号质量": f"{quality_color} {quality}"
            })
        
        import pandas as pd
        st.dataframe(pd.DataFrame(details), use_container_width=True)
