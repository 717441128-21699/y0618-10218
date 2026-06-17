import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection


def generate_sample_moment_tensor():
    """生成示例矩张量"""
    m0 = 1e18
    mt = np.array([
        m0 * 0.5,
        -m0 * 0.3,
        -m0 * 0.2,
        m0 * 0.4,
        -m0 * 0.15,
        m0 * 0.25
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


def plot_beach_ball(input_method="moment_tensor", mt=None, size=300,
                  show_axes=True, show_fault_planes=True):
    """绘制沙滩球图（震源机制解）"""
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, fill=False, linewidth=2, color='black')
    ax.add_patch(circle)
    
    if mt is None:
        mt = generate_sample_moment_tensor()
    
    mt = np.array(mt)
    t_axis, p_axis, b_axis, eigenvalues = moment_tensor_to_t_axes(mt)
    
    if input_method == "moment_tensor":
        fig, ax = plot_moment_tensor_beachball(mt, ax, size)
    
    if show_axes:
        t_az, t_pl = vector_to_azimuth_plunge(t_axis)
        p_az, p_pl = vector_to_azimuth_plunge(p_axis)
        
        t_x, t_y = equal_area_projection(t_az, t_pl)
        p_x, p_y = equal_area_projection(p_az, p_pl)
        
        ax.plot(t_x, t_y, 'o', color='white', markersize=10,
               markeredgecolor='black', label='T轴')
        ax.plot(p_x, p_y, 's', color='black', markersize=10,
               markeredgecolor='black', label='P轴')
        
        ax.legend(loc='lower right')
    
    if show_fault_planes:
        pass
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('震源机制解 (沙滩球图)', fontsize=14, fontweight='bold')
    
    return fig


def plot_moment_tensor_beachball(mt, ax, size=300):
    """根据矩张量绘制沙滩球图"""
    mt = np.array(mt)
    mt_matrix = np.array([
        [mt[0], mt[3], mt[4]],
        [mt[3], mt[1], mt[5]],
        [mt[4], mt[5], mt[2]]
    ])
    
    npts = 200
    theta = np.linspace(0, np.pi, npts)
    phi = np.linspace(0, 2 * np.pi, npts)
    theta, phi = np.meshgrid(theta, phi)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    radiation = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            n = np.array([x[i,j], y[i,j], z[i,j]])
            radiation[i,j] = np.dot(n, np.dot(mt_matrix, n))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    circle = Circle((0, 0), 1, fill=False, linewidth=2, color='black')
    ax.add_patch(circle)
    
    n_angles = 360
    angles = np.linspace(0, 2 * np.pi, n_angles)
    
    x_proj = np.zeros(n_angles)
    y_proj = np.zeros(n_angles)
    radiation_line = np.zeros(n_angles)
    
    for i, angle in enumerate(angles):
        az = angle
        pl = 45
        x_proj[i], y_proj[i] = equal_area_projection(np.degrees(az), pl)
        
        n = np.array([
            np.sin(np.radians(pl)) * np.cos(np.radians(az)),
            np.sin(np.radians(pl)) * np.sin(np.radians(az)),
            np.cos(np.radians(pl))
        ])
        radiation_line[i] = np.dot(n, np.dot(mt_matrix, n))
    
    npts_radial = 50
    npts_angular = 360
    
    angles = np.linspace(0, 2 * np.pi, npts_angular)
    radii = np.linspace(0, 1, npts_radial)
    
    radiation_grid = np.zeros((npts_angular, npts_radial))
    
    for i, angle in enumerate(angles):
        for j, r in enumerate(radii):
            pl = 2 * np.degrees(np.arcsin(r / np.sqrt(2)))
            
            n = np.array([
                np.sin(np.radians(pl)) * np.cos(angle),
                np.sin(np.radians(pl)) * np.sin(angle),
                np.cos(np.radians(pl))
            ])
            
            radiation_grid[i, j] = np.dot(n, np.dot(mt_matrix, n))
    
    X = np.outer(np.cos(angles), radii)
    Y = np.outer(np.sin(angles), radii)
    
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=-np.max(np.abs(radiation_grid)), vmax=np.max(np.abs(radiation_grid)))
    
    cmap = plt.cm.RdBu_r
    
    ax.pcolormesh(X, Y, radiation_grid, cmap=cmap, norm=norm, alpha=0.8)
    
    circle_border = Circle((0, 0), 1, fill=False, linewidth=2, color='black', zorder=5)
    ax.add_patch(circle_border)
    
    ax.plot([-1, 1], [0, 0], 'k-', linewidth=0.5, alpha=0.5)
    ax.plot([0, 0], [-1, 1], 'k-', linewidth=0.5, alpha=0.5)
    
    mt_total = np.sqrt(np.sum(mt**2) / 2)
    mw = (2/3) * (np.log10(mt_total) - 9.1)
    
    ax.text(0, -1.15, f'矩震级 Mw ≈ {mw:.1f}', 
            ha='center', fontsize=10)
    
    return fig, ax


def equal_area_projection(azimuth, plunge):
    """等面积投影（下半球）"""
    az_rad = np.radians(azimuth)
    pl_rad = np.radians(plunge)
    
    r = np.sqrt(2) * np.sin(pl_rad / 2)
    
    x = r * np.sin(az_rad)
    y = -r * np.cos(az_rad)
    
    return x, y


def vector_to_azimuth_plunge(vector):
    """将三维向量转换为方位角和倾角"""
    x, y, z = vector
    
    plunge = np.degrees(np.arccos(np.abs(z)))
    azimuth = np.degrees(np.arctan2(y, x))
    
    if azimuth < 0:
        azimuth += 360
    
    return azimuth, plunge


def plot_first_motion_beachball(polarities, ax):
    """根据初动方向绘制沙滩球图"""
    circle = Circle((0, 0), 1, fill=False, linewidth=2, color='black')
    ax.add_patch(circle)
    
    for az, pl, pol in polarities:
        x, y = equal_area_projection(az, pl)
        if pol == 'compressional':
            ax.plot(x, y, 'o', color='black', markersize=6)
        else:
            ax.plot(x, y, 'o', color='white', markersize=6,
                   markeredgecolor='black')
    
    return ax


def simple_beach_ball(strike, dip, rake):
    """根据震源参数绘制简单沙滩球图"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    circle = Circle((0, 0), 1, fill=False, linewidth=2, color='black')
    ax.add_patch(circle)
    
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    rake_rad = np.radians(rake)
    
    fault_x = np.array([-1, 1])
    fault_y = np.array([0, 0])
    
    ax.plot(fault_x, fault_y, 'k-', linewidth=2)
    
    aux_plane_dip = 90 - dip
    aux_radius = np.sqrt(2) * np.sin(np.radians(aux_plane_dip) / 2)
    
    circle_dip = Circle((0, 0), aux_radius, fill=False,
                       linewidth=1.5, color='black', linestyle='--')
    ax.add_patch(circle_dip)
    
    wedge = Wedge((0, 0), 1, strike - 90, strike - 90 + 180,
                 facecolor='black', alpha=0.3)
    ax.add_patch(wedge)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title(f'震源机制解\n走向: {strike}° 倾角: {dip}° 滑动角: {rake}°',
                 fontsize=12)
    
    return fig
