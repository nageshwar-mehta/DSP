# === Script: Perform trilateration on real scan data using RSS only and visualize results ===
"""
This script loads real Wi-Fi access point data and scan results,
performs trilateration using only RSS-based distance estimates,
computes error metrics, and plots estimates versus ground truth.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trilat import trilat
import os

def trilat_real_rss(wifis, scans, rss_std=5.0, use_weights=True):
    """
    Perform trilateration for each scan group in `scans`, using only RSS measurements.
    Returns arrays of errors, sensor positions, estimates, and ground truth.
    """
    wifis = wifis.set_index('bssid')
    errors = []
    sensors = []
    estimates = []
    ground_truth = []
    for scan_id, group in scans.groupby('scanId'):
        s_list, d2_list, w_list = [], [], []
        for _, row in group.iterrows():
            ap = wifis.loc[row.bssid]
            si = np.array([ap.x, ap.y])
            # RSS only
            if not pd.isna(row.rssi):
                C = row.rssi
                C0 = ap.txPower
                eta = ap.pathLossExponent
                d2 = 10**((C0 - C) / (5 * eta))
                w = (5 * eta / (d2 * np.log(10)))**2 / rss_std**2
                s_list.append(si)
                d2_list.append(d2)
                w_list.append(w)
        if len(d2_list) < 2:
            continue
        s = np.column_stack(s_list)
        d2 = np.array(d2_list)
        W = np.diag(w_list) if use_weights else None
        x_sol = trilat(s, d2, W)
        if x_sol.shape[1] > 1:
            x = x_sol[:,0]
        else:
            x = x_sol.ravel()
        xgt = np.array([group.iloc[0].x, group.iloc[0].y])
        err = np.linalg.norm(x - xgt)
        errors.append(err)
        sensors.append(s.T)
        estimates.append(x)
        ground_truth.append(xgt)
    return np.array(errors), sensors, estimates, ground_truth

if __name__ == '__main__':
    # Use this script's directory for data
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    # Load CSVs: AP positions and scan measurements
    wifis = pd.read_csv(os.path.join(data_dir, 'wifis.csv'))
    scans = pd.read_csv(os.path.join(data_dir, 'scans.csv'))
    errors, sensors, estimates, gt = trilat_real_rss(wifis, scans)
    # Print basic error statistics
    print(f"Receivers: {len(errors)}")
    print(f"Median error: {np.median(errors):.3f} m")
    print(f"Mean error: {np.mean(errors):.3f} m")
    # Plot
    plt.figure(figsize=(6,5))
    for s, x, xgt in zip(sensors, estimates, gt):
        plt.plot([xgt[0], x[0]], [xgt[1], x[1]], color='gray', lw=0.5)
    plt.scatter([ap[0] for ap in sensors[0]], [ap[1] for ap in sensors[0]], c='lime', label='sensors')
    plt.scatter([g[0] for g in gt], [g[1] for g in gt], c='black', label='ground truth')
    ests = np.array(estimates)
    plt.scatter(ests[:,0], ests[:,1], c='red', label='estimates')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.savefig('real_data_rss_plot.png', dpi=300)
    print("Saved plot to real_data_rss_plot.png")
    plt.show()