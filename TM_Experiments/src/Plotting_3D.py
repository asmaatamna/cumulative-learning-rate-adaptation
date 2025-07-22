#---------------------------------------------------------*\
# Title: 3D Plotting
# Author: 
#---------------------------------------------------------*/

import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plotly.colors import qualitative
import plotly.graph_objects as go
import numpy as np
import torch
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from matplotlib.colors import Normalize

pio.kaleido.scope.default_format = "pdf"
plt.rcParams['figure.dpi'] = 100  # Set DPI to 150 (default is usually 72)

#---------------------------------------------------------*/
# Mathematical Functions
#---------------------------------------------------------*/

def sphere(x):
    return np.sum(x**2)

def sphere_torch(x):
    return torch.sum(x**2)

sigma = 0.1  # Noise std
def sphere_noisy(x, sigma=sigma):
    x_noisy = x + np.random.normal(0, sigma, size=x.shape)
    return np.sum(x_noisy**2)

def sphere_noisy_torch(x, sigma=sigma):
    x_noisy = x + sigma * torch.randn_like(x)
    return torch.sum(x_noisy**2)

cond = 1e3
def ellipsoid(x):
    n = len(x)
    weights = np.power(cond, np.arange(n) / (n - 1))
    return np.sum(weights * x**2)

def ellipsoid_torch(x):
    n = len(x)
    weights = torch.pow(torch.tensor(cond), torch.arange(n, dtype=x.dtype, device=x.device) / (n - 1))
    return torch.sum(weights * x**2)

def ellipsoid_noisy(x, sigma=sigma):
    x_noisy = x + np.random.normal(0, sigma, size=x.shape)
    n = len(x_noisy)
    weights = np.power(cond, np.arange(n) / (n - 1))
    return np.sum(weights * x_noisy**2)

def ellipsoid_noisy_torch(x, sigma=sigma):
    n = len(x)
    weights = torch.pow(torch.tensor(cond, dtype=x.dtype, device=x.device),
                        torch.arange(n, dtype=x.dtype, device=x.device) / (n - 1))
    x_noisy = x + sigma * torch.randn_like(x)
    return torch.sum(weights * x_noisy**2)

def rosenbrock(x):
    a, b = 1, 100
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

def rosenbrock_torch(x):
    a, b = 1, 100
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_torch(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def beale(x):
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def beale_torch(x):
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)
    
    
def eggholder(x):
    return (-(x[1]+47)*np.sin(np.sqrt(abs(x[0]/2 + (x[1]+47)))) - 
            x[0]*np.sin(np.sqrt(abs(x[0] - (x[1]+47)))))

def eggholder_torch(x):
    term1 = -(x[1]+47)*torch.sin(torch.sqrt(torch.abs(x[0]/2 + (x[1]+47))))
    term2 = -x[0]*torch.sin(torch.sqrt(torch.abs(x[0] - (x[1]+47))))
    return term1 + term2


def three_hump_camel(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2

def three_hump_camel_torch(x):
    return (2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2)


def ackley(x):
    x0, x1 = x[0], x[1]
    part1 = -20*np.exp(-0.2*np.sqrt(0.5*(x0**2 + x1**2)))
    part2 = -np.exp(0.5*(np.cos(2*np.pi*x0) + np.cos(2*np.pi*x1)))
    return part1 + part2 + np.e + 20

def ackley_torch(x):
    part1 = -20*torch.exp(-0.2*torch.sqrt(0.5*(x[0]**2 + x[1]**2)))
    part2 = -torch.exp(0.5*(torch.cos(2*torch.pi*x[0]) + torch.cos(2*torch.pi*x[1])))
    return part1 + part2 + torch.exp(torch.tensor(1.0)) + 20


#---------------------------------------------------------*/
# Helper Functions
#---------------------------------------------------------*/

def show_plot_and_save(fig, permanent_path="./results/", filename=None):
    """Create an HTML file with a custom name, save it, and open it."""
    os.makedirs(permanent_path, exist_ok=True)
    file_path = os.path.join(permanent_path, filename)
    fig.write_html(file_path, auto_open=False)
    print(f"✅ Plot saved as {file_path}")


def to_torch_tensor(x, device='cpu'):
    return torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)


def numpy_func_wrapper(func):
    """Return a function that converts torch.Tensor to numpy and applies func"""
    def f(x_torch):
        x_np = x_torch.detach().cpu().numpy()
        return func(x_np)
    return f

def get_func_and_torch(name):
    funcs_np = {
        "sphere": sphere,
        "sphere_noisy": sphere_noisy,
        "ellipsoid": ellipsoid,
        "ellipsoid_noisy": ellipsoid_noisy,
        "rosenbrock": rosenbrock,
        "himmelblau": himmelblau,
        "beale": beale,
        "eggholder": eggholder,
        "three_hump_camel": three_hump_camel,
        "ackley": ackley,
    }
    funcs_torch = {
        "sphere": sphere_torch,
        "sphere_noisy": sphere_noisy_torch,
        "ellipsoid": ellipsoid_torch,
        "ellipsoid_noisy": ellipsoid_noisy_torch,
        "rosenbrock": rosenbrock_torch,
        "himmelblau": himmelblau_torch,
        "beale": beale_torch,
        "eggholder": eggholder_torch,
        "three_hump_camel": three_hump_camel_torch,
        "ackley": ackley_torch,
    }
    return funcs_np[name], funcs_torch[name]

#---------------------------------------------------------*/
# 3D Plotting
#---------------------------------------------------------*/
def optimize_and_plot(
    func_np, func_torch, global_optimum,
    optimizers,
    start_points,
    bounds=[-2, 2],
    steps_dict=None,
    lr_dict=None,
    gradient_clip=None,
    device='cpu',
    func_name=None,
    show = False
):
    # Surface grid
    x_vals = y_vals = np.linspace(bounds[0], bounds[1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # Compute surface
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                val = func_np(np.array([X[i, j], Y[i, j]]))
                Z[i, j] = val if np.isfinite(val) else np.nan
            except Exception:
                Z[i, j] = np.nan

    # print(f"[{func_name}] Z-statistics before clipping: min={np.nanmin(Z):.2e}, max={np.nanmax(Z):.2e}")

    # ------------------------------------------------
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    # print(f"[{func_name}] Clipping disabled – Using full Z-range: {zmin:.2f} to {zmax:.2f}")
    # ------------------------------------------------

    # Optimization paths
    paths = {}
    color_cycle = qualitative.Plotly
    color_list = list(color_cycle)
    color_map = {}
    arrow_traces = []
    
    for idx, (name, OptimizerClass) in enumerate(optimizers.items()):
        lr = lr_dict.get(name, 0.01) if lr_dict else 0.01
        steps = steps_dict.get(name, 50) if steps_dict else 50
        start = start_points[name]

        x_torch = torch.tensor(start, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = OptimizerClass([x_torch], lr=lr)
        path = [x_torch.detach().cpu().clone().numpy()]

        for step_idx in range(steps):
            optimizer.zero_grad()
            val = func_torch(x_torch)
            val.backward()

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_([x_torch], max_norm=gradient_clip)

            optimizer.step()

            with torch.no_grad():
                x_torch[:] = torch.clamp(x_torch, bounds[0], bounds[1])

            if not torch.isfinite(x_torch).all() or (x_torch.abs() > 1e3).any():
                print(f"⚠️ Divergence in optimizer '{name}' at step {step_idx}: {x_torch.detach().cpu().numpy()}")
                break

            path.append(x_torch.detach().cpu().clone().numpy())

        paths[name] = np.array(path)
        
        # Store direction vector (last step)
        if len(path) >= 2:
            direction = path[-1] - path[-2]
            direction = direction / np.linalg.norm(direction) * 0.1  # normalize and scale
            arrow_start = path[-1]
            arrow_end = arrow_start + direction

            # Compute z-values at both ends
            z_arrow = [
                np.clip(func_np(arrow_start), zmin, zmax),
                np.clip(func_np(arrow_end), zmin, zmax)
            ]

            # Erzeuge Trace, aber füge ihn noch nicht zur Figur hinzu
            arrow_trace = go.Scatter3d(
                x=[arrow_start[0], arrow_end[0]],
                y=[arrow_start[1], arrow_end[1]],
                z=z_arrow,
                mode='lines+markers',
                line=dict(color='black', width=6),
                marker=dict(size=[0, 6], color='black'),
                showlegend=False
            )

            arrow_traces.append(arrow_trace)
        
        color_map[name] = color_list[idx % len(color_list)]

    # Plotly figure
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.9,
        name='Loss Surface',
        showscale=True,
        colorbar=dict(title='Loss')
    ))
    
    for arrow in arrow_traces:
        fig.add_trace(arrow)

    for name, path in paths.items():
        try:
            Z_path = [np.clip(func_np(p), zmin, zmax) for p in path]
        except Exception as e:
            warnings.warn(f"Pfadberechnung für {name} fehlgeschlagen: {e}")
            Z_path = [zmax for _ in path]

    for name, path in paths.items():
        try:
            Z_path = [np.clip(func_np(p), zmin, zmax) for p in path]
        except Exception as e:
            warnings.warn(f"Pfadberechnung für {name} fehlgeschlagen: {e}")
            Z_path = [zmax for _ in path]

        min_width, max_width = 1, 3
        min_opacity, max_opacity = 0.5, 1.0

        # Generiere Liniensegmente mit steigender Transparenz
        for i in range(1, len(path)):
            width = min_width + (max_width - min_width) * (i / len(path))
            opacity = min_opacity + (max_opacity - min_opacity) * (i / len(path))
            fig.add_trace(go.Scatter3d(
                x=[path[i - 1][0], path[i][0]],
                y=[path[i - 1][1], path[i][1]],
                z=[Z_path[i - 1], Z_path[i]],
                mode='lines',
                line=dict(color=color_map[name], width=width),
                opacity=opacity,
                showlegend=False
            ))

        # Punkte mit voller Deckkraft
        fig.add_trace(go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=Z_path,
            mode='markers',
            marker=dict(size=4, color=color_map[name], opacity=1.0),
            name=f'{name} Path'
        ))


    fig.update_layout(
        title=f"Optimization paths on objective function: {func_name}",
        autosize=False,  # Disable automatic resizing
        width=800,       # Set the desired width
        height=600,      # Set the desired height
        xaxis_title="x",
        yaxis_title="y",
        margin=dict(l=0, r=0, t=50, b=0),
        font=dict(family="Arial", size=14),
        legend=dict(x=0.8, y=0.95)
    )

    if show == True:
        fig.show()
    show_plot_and_save(fig, permanent_path="./results/", filename=f"{func_name}_optimization_paths.html")


#---------------------------------------------------------*/
# 2D Plotting
#---------------------------------------------------------*/

def optimize_and_plot_2d(
    func_np, func_torch, global_optimum,
    optimizers,
    start_points,
    bounds=[-2, 2],
    steps_dict=None,
    lr_dict=None,
    gradient_clip=None,
    device='cpu',
    func_name=None,
    show = False
):
    # Grid for objective function
    x_vals = y_vals = np.linspace(bounds[0], bounds[1], 300)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                val = func_np(np.array([X[i, j], Y[i, j]]))
                Z[i, j] = val if np.isfinite(val) else np.nan
            except Exception:
                Z[i, j] = np.nan

    # Color Normalization
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    norm = Normalize(vmin=zmin, vmax=zmax)

    # Colors
    color_palette = sns.color_palette("Set1", n_colors=len(optimizers))
    paths = {}

    # Optimizer Paths
    for idx, (name, OptimizerClass) in enumerate(optimizers.items()):
        lr = lr_dict.get(name, 0.01) if lr_dict else 0.01
        steps = steps_dict.get(name, 50) if steps_dict else 50
        start = start_points[name]

        x_torch = torch.tensor(start, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = OptimizerClass([x_torch], lr=lr)
        path = [x_torch.detach().cpu().clone().numpy()]

        for step_idx in range(steps):
            optimizer.zero_grad()
            val = func_torch(x_torch)
            val.backward()

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_([x_torch], max_norm=gradient_clip)

            optimizer.step()
            with torch.no_grad():
                x_torch[:] = torch.clamp(x_torch, bounds[0], bounds[1])

            if not torch.isfinite(x_torch).all():
                print(f"⚠️ Divergenz bei {name}, Schritt {step_idx}")
                break

            path.append(x_torch.detach().cpu().clone().numpy())

        paths[name] = np.array(path)

    # ---- Plotting ----
    fig, ax = plt.subplots(figsize=(8, 6))

    # Target Function
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', norm=norm)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r"$f(x, y)$", fontsize=14)

    # Optimizer Paths
    for idx, (name, path) in enumerate(paths.items()):
        color = color_palette[idx]
        ax.plot(path[:, 0], path[:, 1], '-', label=name, color=color, linewidth=2)
        ax.scatter(path[::5, 0], path[::5, 1], s=20, color=color, edgecolors='k', zorder=3)

    # Global Optimum
    ax.scatter(global_optimum[0], global_optimum[1], s=80, c='black', marker='*', label='Global Optimum', zorder=4)

    # Labels and Title
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(f"Optimization Trajectories on {func_name} Objective", fontsize=16)

    # Legend
    ax.legend(loc='upper right', fontsize=12, frameon=True)

    # Axis Limits
    ax.grid(True, linestyle='--', alpha=0.3)

    # Save
    output_path = f"./results/{func_name}_optimization_paths_2d_matplotlib.pdf"
    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    print(f"✅ Plot saved as PDF: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

#---------------------------------------------------------*/
# Animated Plot
#---------------------------------------------------------*/


def optimize_and_animate(
    func_np, func_torch, global_optimum,
    optimizers,
    start_points,
    bounds=[-2, 2],
    steps_dict=None,
    lr_dict=None,
    gradient_clip=None,
    device='cpu',
    func_name=None,
    show = False
):
    # Loss surface
    x_vals = y_vals = np.linspace(bounds[0], bounds[1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                val = func_np(np.array([X[i, j], Y[i, j]]))
                Z[i, j] = val if np.isfinite(val) else np.nan
            except Exception:
                Z[i, j] = np.nan

    zmin, zmax = np.nanmin(Z), np.nanmax(Z)

    # === Optimization paths ===
    paths = {}
    max_steps = 0
    for name, OptimizerClass in optimizers.items():
        steps = steps_dict.get(name, 50) if steps_dict else 50
        start = start_points[name]
        lr = lr_dict.get(name, 0.01) if lr_dict else 0.01

        x_torch = torch.tensor(start, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = OptimizerClass([x_torch], lr=lr)

        point = x_torch.detach().cpu().clone().numpy()
        z_val = func_np(point)
        path = [(point, z_val)]  # tuple (x, z)

        for _ in range(steps):
            optimizer.zero_grad()
            val = func_torch(x_torch)
            val.backward()

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_([x_torch], max_norm=gradient_clip)
            optimizer.step()
            with torch.no_grad():
                x_torch[:] = torch.clamp(x_torch, bounds[0], bounds[1])

            point = x_torch.detach().cpu().clone().numpy()
            z_val = func_np(point)  # Rauschen nur einmal!
            path.append((point, z_val))


        paths[name] = path 
        max_steps = max(max_steps, len(path))

    # === Base figure (initial frame) ===
    fig = go.Figure()

    # Update zmin, zmax basierend auf Pfaden
    for path in paths.values():
        z_vals = [z for (_, z) in path]
        zmin = min(zmin, np.nanmin(z_vals))
        zmax = max(zmax, np.nanmax(z_vals))

    # Optional: padding hinzufügen
    padding = 0.05 * (zmax - zmin)
    zmin -= padding
    zmax += padding

    # Surface (static, but must be re-added in each frame)
    surface_trace = go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.8,
        showscale=False,
        name="Loss Surface"
    )
    fig.add_trace(surface_trace)

    # Static initialization of all optimizer traces (empty)
    colors = px.colors.qualitative.Plotly
    path_traces = []
    marker_traces = []
    for idx, (name, path) in enumerate(paths.items()):
        path_traces.append(go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(color=colors[idx], width=4),
            name=f"{name} Path",
            showlegend=True
        ))
        marker_traces.append(go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(size=8, color='red'),
            name=f"{name} Current",
            showlegend=False
        ))

    for trace in path_traces + marker_traces:
        fig.add_trace(trace)

    # === Animation frames ===
    frames = []
    for step in range(1, max_steps):
        frame_data = [surface_trace]  # surface must be in every frame

        for idx, (name, path) in enumerate(paths.items()):
            if step < len(path):
                sub_path = path[:step + 1]
                x_vals = [p[0][0] for p in sub_path]
                y_vals = [p[0][1] for p in sub_path]
                z_vals = [p[1]     for p in sub_path]

                current_point = sub_path[-1][0]
                z_current = sub_path[-1][1]
                
                # full path so far
                frame_data.append(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='lines',
                    line=dict(color=colors[idx], width=4),
                    name=f"{name} Path",
                    showlegend=False
                ))
                # current point
                frame_data.append(go.Scatter3d(
                    x=[current_point[0]],
                    y=[current_point[1]],
                    z=[z_current],
                    mode='markers',
                    marker=dict(size=8, color=colors[idx]),
                    name=f"{name} Current",
                    showlegend=False
                ))


        frames.append(go.Frame(data=frame_data, name=f"frame{step}"))

    fig.frames = frames

    # Animation settings
    fig.update_layout(
        title=f"Optimization (Animation): {func_name}",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)",
            zaxis=dict(range=[zmin, zmax]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 1000, "redraw": True},
                                                                  "fromcurrent": True, "mode": "immediate"}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                     "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
                        label=str(i)) for i, f in enumerate(frames)],
            active=0
        )]
    )

    # Save + show
    if show == True:
        fig.show()
    show_plot_and_save(fig, filename=f"{func_name}_animation.html")


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\