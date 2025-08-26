#---------------------------------------------------------*\
# Title: 
# Author: 
#---------------------------------------------------------*/

import numpy as np
import matplotlib.pyplot as plt

def visualize_paths(num_steps=15):
    def sphere(x, y):
        return x**2 + y**2

    # Grid for the sphere function
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = sphere(X, Y)

    # Starting points
    start_long = np.array([2.0, 2.0])
    start_short = np.array([0.1, 0.1])  # Short path closer to optimum

    np.random.seed(43)

    # Reference Path for long path
    steps_long_ref = np.cumsum(np.random.normal(scale=0.2, size=(num_steps-1, 2)), axis=0)
    ref_steps_long = np.vstack([start_long, start_long + steps_long_ref])

    # Longer Path
    long_steps = np.cumsum(np.tile([-0.2, -0.2], (num_steps-1, 1)) +
                           np.random.normal(scale=0.05, size=(num_steps-1, 2)), axis=0)
    long_path = np.vstack([start_long, start_long + long_steps])

    # Shorter Path closer to optimum
    short_steps = np.cumsum(np.random.choice([-1, 1], size=(num_steps-1, 2)) *
                            np.random.normal(scale=0.15, size=(num_steps-1, 2)), axis=0)
    short_path = np.vstack([start_short, start_short + short_steps])

    # Second Reference Path for short path
    steps_short_ref = np.cumsum(np.random.normal(scale=0.15, size=(num_steps-1, 2)), axis=0)
    ref_steps_short = np.vstack([start_short, start_short + steps_short_ref])

    fig, ax = plt.subplots(figsize=(6, 3))  # Make the figure more flat (wider than tall)
    contours = ax.contour(X, Y, Z, levels=30, cmap='viridis', linewidths=0.8)

    # Plot paths for long path
    ax.plot(ref_steps_long[:, 0], ref_steps_long[:, 1], 'o-', color='grey', label='Reference Path (2x)', markersize=4)
    ax.plot([ref_steps_long[0, 0], ref_steps_long[-1, 0]],
            [ref_steps_long[0, 1], ref_steps_long[-1, 1]],
            color='black', linewidth=1, linestyle='--')

    ax.plot(long_path[:, 0], long_path[:, 1], 'o-', color='blue', label='Longer Path', markersize=4)
    ax.plot([long_path[0, 0], long_path[-1, 0]],
            [long_path[0, 1], long_path[-1, 1]],
            color='black', linewidth=1, )

    # Plot paths for short path
    ax.plot(ref_steps_short[:, 0], ref_steps_short[:, 1], 'o-', color='grey', label='', markersize=4)
    ax.plot([ref_steps_short[0, 0], ref_steps_short[-1, 0]],
            [ref_steps_short[0, 1], ref_steps_short[-1, 1]],
            color='black', linewidth=1, linestyle='--')

    ax.plot(short_path[:, 0], short_path[:, 1], 'o-', color='red', label='Shorter Path', markersize=3)
    ax.plot([short_path[0, 0], short_path[-1, 0]],
            [short_path[0, 1], short_path[-1, 1]],
            color='black', linewidth=1, )

    # Styling
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(-0.5, 2.5)

    plt.tight_layout()
    fig.savefig('../results/visualization_plot.png', dpi=300, format='png')
    plt.show()

visualize_paths(num_steps=6)


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\