import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trilat import trilat
import os


def trilat_real(wifis, scans, include_rtt=True, include_rss=False, use_weights=True, rtt_std=1.0, rss_std=5.0):
    # Prepare data
    wifis = wifis.set_index('bssid')
    errors = []
    sensors = []
    estimates = []
    ground_truth = []

    for scan_id, group in scans.groupby('scanId'):
        s_list, d2_list, w_list = [], [], []
        # optional RSS parameters
        for _, row in group.iterrows():
            ap = wifis.loc[row.bssid]
            si = np.array([ap.x, ap.y])
            # RTT
            if include_rtt and not pd.isna(row.rttDist):
                d = row.rttDist/1000  # m
                d2 = d**2
                w = 1/(4*d2) / rtt_std**2
                s_list.append(si)
                d2_list.append(d2)
                w_list.append(w)
            # RSS
            if include_rss and not pd.isna(row.rssi):
                C = row.rssi
                C0 = ap.txPower
                eta = ap.pathLossExponent
                d2 = 10**((C0 - C)/(5*eta))
                w = (5*eta/(d2*np.log(10)))**2 / rss_std**2
                s_list.append(si)
                d2_list.append(d2)
                w_list.append(w)
        if len(d2_list) < 2:
            continue
        # sensors s: dims×count matrix
        s = np.column_stack(s_list)
        d2 = np.array(d2_list)
        W = np.diag(w_list) if use_weights else None
        x_sol = trilat(s, d2, W)
        if x_sol.shape[1] > 1:
            x = x_sol[:,0]
        else:
            x = x_sol.ravel()
        xgt = np.array([group.iloc[0].x, group.iloc[0].y])
        errors.append(np.linalg.norm(x - xgt))
        sensors.append(s.T)
        estimates.append(x)
        ground_truth.append(xgt)
    return np.array(errors), sensors, estimates, ground_truth


if __name__ == '__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    wifis = pd.read_csv(os.path.join(base, 'data', 'wifis.csv'))
    scans = pd.read_csv(os.path.join(base, 'data', 'scans.csv'))
    errors, sensors, estimates, gt = trilat_real(wifis, scans, use_weights=False)
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
    plt.savefig('real_data_plot.png', dpi=300)
    print("Saved plot to real_data_plot.png")
    # Display plot interactively
    plt.show()

    # Compare metrics with Julia implementation
    import subprocess, re
    print("\n=== Comparing with Julia ===")
    julia_script = os.path.join(base, 'test_real_data.jl')
    jl = subprocess.run(
        ['julia', '--project=' + base, julia_script],
        cwd=base, capture_output=True, text=True
    )
    print(jl.stdout)
    # Extract metrics
    jl_median = None; jl_mean = None
    for line in jl.stdout.splitlines():
        m = re.search(r"Median error:\s*([0-9.]+)", line)
        if m: jl_median = float(m.group(1))
        m = re.search(r"Mean error:\s*([0-9.]+)", line)
        if m: jl_mean = float(m.group(1))
    py_median = float(np.median(errors))
    py_mean = float(np.mean(errors))
    print(f"Python median: {py_median:.6f}, Julia median: {jl_median:.6f}")
    print(f"Python mean:   {py_mean:.6f}, Julia mean:   {jl_mean:.6f}")
