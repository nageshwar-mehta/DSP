# === Script: Perform trilateration on real scan data using RTT+RSS and visualize results ===
"""
This script loads real Wi-Fi access point data and scan results,
performs trilateration using both RTT and RSS-based distance estimates,
computes error metrics, and plots estimates versus ground truth and receiver (sensor) positions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from real_data import trilat_real
import os

if __name__ == '__main__':
    # Use this script's directory for data
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    # Load CSVs: AP positions and scan measurements
    wifis = pd.read_csv(os.path.join(data_dir, 'wifis.csv'))
    scans = pd.read_csv(os.path.join(data_dir, 'scans.csv'))
    errors, sensors, estimates, gt = trilat_real(wifis, scans, include_rtt=True, include_rss=True)
    # Print basic error statistics
    print(f"Receivers: {len(errors)}")
    print(f"Median error: {np.median(errors):.3f} m")
    print(f"Mean error: {np.mean(errors):.3f} m")
    # Plot
    plt.figure(figsize=(6,5))
    for s, x, xgt in zip(sensors, estimates, gt):
        plt.plot([xgt[0], x[0]], [xgt[1], x[1]], color='gray', lw=0.5)
        plt.scatter(s[:,0], s[:,1], c='lime', s=30, marker='^', alpha=0.5, label='sensors' if 'sensors' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.scatter([g[0] for g in gt], [g[1] for g in gt], c='black', label='ground truth')
    ests = np.array(estimates)
    plt.scatter(ests[:,0], ests[:,1], c='red', label='estimates')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.savefig('real_data_combined_plot.png', dpi=300)
    print("Saved plot to real_data_combined_plot.png")
    plt.show()