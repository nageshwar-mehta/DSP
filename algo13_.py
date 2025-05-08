import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class TrilaterationSolver:
    def __init__(self):
        pass
    
    def solve_algorithm1(self, sender_positions, distance_measurements, weights=None):
        """
        Implementation of Algorithm 1 from the paper
        
        Args:
            sender_positions: array of shape (m, n) where m is number of senders
                             and n is dimension (typically 2 or 3)
            distance_measurements: array of shape (m,) with distance measurements
            weights: optional weight matrix of shape (m, m)
                     if None, uses identity matrix
                     
        Returns:
            Estimated receiver position as array of shape (n,)
        """
        m, n = sender_positions.shape
        
        # Step 1: Normalize weights
        if weights is None:
            weights = np.eye(m)
        weights = weights / np.sum(weights)
        
        # Step 2: Translate senders
        t = np.zeros(n)
        for i in range(m):
            for j in range(m):
                t += weights[i, j] * sender_positions[i]
        
        translated_senders = sender_positions - t
        
        # Step 3: Calculate A and g
        A = np.zeros((n, n))
        g = np.zeros(n)
        
        for i in range(m):
            for j in range(m):
                w_ij = weights[i, j]
                s_i = translated_senders[i]
                s_j = translated_senders[j]
                d_i = distance_measurements[i]
                
                term = w_ij * (2 * np.outer(s_j, s_i) + (np.dot(s_i, s_i) - d_i**2) * np.eye(n))
                A += term
                
                g_term = w_ij * (np.dot(s_i, s_i) - d_i**2) * s_j
                g -= g_term
        
        # Step 4: Construct M_A matrix and find largest real eigenvalue
        M_A = np.zeros((2*n + 1, 2*n + 1))
        
        # Top-left block: A
        M_A[:n, :n] = A
        
        # Top-middle block: Identity
        M_A[:n, n:2*n] = np.eye(n)
        
        # Middle-left block: Zero
        M_A[n:2*n, :n] = np.zeros((n, n))
        
        # Middle-middle block: A
        M_A[n:2*n, n:2*n] = A
        
        # Middle-right block: -g
        M_A[n:2*n, 2*n] = -g
        
        # Bottom-left block: -g^T
        M_A[2*n, :n] = -g
        
        # Find eigenvalues
        eigenvalues = eigh(M_A, eigvals_only=True)
        
        # Find largest real eigenvalue
        real_eigenvalues = eigenvalues[np.isreal(eigenvalues)].real
        lambda_max = np.max(real_eigenvalues)
        
        # Step 5: Calculate receiver position
        x = -np.linalg.solve(lambda_max * np.eye(n) - A, g)
        
        # Step 6: Undo translation
        x += t
        
        return x
    
    def solve_rtt_measurements(self, scans_df, wifis_df):
        """
        Solve using RTT measurements with appropriate weighting
        
        Args:
            scans_df: DataFrame from scans.csv
            wifis_df: DataFrame from wifis.csv
            
        Returns:
            Dictionary mapping scanId to estimated position
        """
        # Merge data to get sender positions
        merged = pd.merge(scans_df, wifis_df, on='bssid', suffixes=('_scan', '_wifi'))
        
        results = {}
        sigma_rtt = 1.0  # Standard deviation of RTT noise (1m as in paper)
        
        for scan_id, group in merged.groupby('scanId'):
            # Get sender positions and measurements
            sender_positions = group[['x_wifi', 'y_wifi']].values
            rtt_distances = group['rttDist'].values
            
            # Calculate weights (diagonal matrix)
            weights = np.diag(1 / (4 * sigma_rtt**2 * rtt_distances**2))
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Solve
            try:
                estimated_pos = self.solve_algorithm1(sender_positions, rtt_distances, weights)
                results[scan_id] = estimated_pos
            except np.linalg.LinAlgError:
                print(f"Failed to solve for scanId {scan_id}")
                results[scan_id] = None
                
        return results
    
    def solve_rss_measurements(self, scans_df, wifis_df):
        """
        Solve using RSS measurements with appropriate weighting
        
        Args:
            scans_df: DataFrame from scans.csv
            wifis_df: DataFrame from wifis.csv
            
        Returns:
            Dictionary mapping scanId to estimated position
        """
        # Merge data to get sender positions
        merged = pd.merge(scans_df, wifis_df, on='bssid', suffixes=('_scan', '_wifi'))
        
        results = {}
        sigma_rss = 5.0  # Standard deviation of RSS noise (5 dBm as in paper)
        
        for scan_id, group in merged.groupby('scanId'):
            # Get sender positions and measurements
            sender_positions = group[['x_wifi', 'y_wifi']].values
            rss_values = group['rssi'].values
            path_loss_exponents = group['pathLossExponent'].values
            tx_powers = group['txPower'].values
            
            # Convert RSS to distance using log-distance path loss model
            d_squared = 10**((tx_powers - rss_values) / (5 * path_loss_exponents))
            distances = np.sqrt(d_squared)
            
            # Calculate weights (diagonal matrix)
            weights = np.diag((5 * path_loss_exponents / (sigma_rss * d_squared * np.log(10)))**2)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Solve
            try:
                estimated_pos = self.solve_algorithm1(sender_positions, distances, weights)
                results[scan_id] = estimated_pos
            except np.linalg.LinAlgError:
                print(f"Failed to solve for scanId {scan_id}")
                results[scan_id] = None
                
        return results
    
    def solve_combined_measurements(self, scans_df, wifis_df):
        """
        Solve using both RTT and RSS measurements with balanced weighting
        
        Args:
            scans_df: DataFrame from scans.csv
            wifis_df: DataFrame from wifis.csv
            
        Returns:
            Dictionary mapping scanId to estimated position
        """
        # Merge data to get sender positions
        merged = pd.merge(scans_df, wifis_df, on='bssid', suffixes=('_scan', '_wifi'))
        
        results = {}
        sigma_rtt = 1.0  # Standard deviation of RTT noise (1m as in paper)
        sigma_rss = 5.0  # Standard deviation of RSS noise (5 dBm as in paper)
        
        for scan_id, group in merged.groupby('scanId'):
            # Get sender positions
            sender_positions = group[['x_wifi', 'y_wifi']].values
            
            # Initialize lists for all measurements
            all_distances = []
            all_weights = []
            
            # Process RTT measurements if available
            if 'rttDist' in group.columns:
                rtt_mask = ~group['rttDist'].isna()
                if rtt_mask.any():
                    rtt_distances = group.loc[rtt_mask, 'rttDist'].values
                    rtt_weights = 1 / (4 * sigma_rtt**2 * rtt_distances**2)
                    all_distances.extend(rtt_distances)
                    all_weights.extend(rtt_weights)
            
            # Process RSS measurements if available
            if 'rssi' in group.columns:
                rss_mask = ~group['rssi'].isna()
                if rss_mask.any():
                    rss_values = group.loc[rss_mask, 'rssi'].values
                    path_loss_exponents = group.loc[rss_mask, 'pathLossExponent'].values
                    tx_powers = group.loc[rss_mask, 'txPower'].values
                    
                    # Convert RSS to distance
                    d_squared = 10**((tx_powers - rss_values) / (5 * path_loss_exponents))
                    rss_distances = np.sqrt(d_squared)
                    rss_weights = (5 * path_loss_exponents / (sigma_rss * d_squared * np.log(10)))**2
                    
                    all_distances.extend(rss_distances)
                    all_weights.extend(rss_weights)
            
            # Skip if no measurements
            if not all_distances:
                results[scan_id] = None
                continue
                
            # Balance weights between RTT and RSS
            if len(all_weights) > 0:
                max_weight = max(all_weights)
                if max_weight > 0:
                    all_weights = [w/max_weight for w in all_weights]
            
            # Create weight matrix (diagonal)
            weights = np.diag(all_weights)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Solve using all available measurements
            try:
                estimated_pos = self.solve_algorithm1(sender_positions, np.array(all_distances), weights)
                results[scan_id] = estimated_pos
            except np.linalg.LinAlgError:
                print(f"Failed to solve for scanId {scan_id}")
                results[scan_id] = None
                
        return results
    
    def visualize_results(self, results, ground_truth, wifis_df, title="Positioning Results"):
        """
        Visualize the positioning results compared to ground truth
        
        Args:
            results: Dictionary of estimated positions (from solve_* methods)
            ground_truth: DataFrame with ground truth positions
            wifis_df: DataFrame with WiFi access point positions
            title: Title for the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Plot WiFi access points
        plt.scatter(wifis_df['x'], wifis_df['y'], 
                   c='green', marker='^', s=100, label='WiFi APs')
        
        # Plot ground truth positions
        plt.scatter(ground_truth['x'], ground_truth['y'], 
                   c='black', marker='o', s=50, label='Ground Truth')
        
        # Plot estimated positions
        estimated_x = []
        estimated_y = []
        for scan_id, pos in results.items():
            if pos is not None and scan_id in ground_truth.index:
                estimated_x.append(pos[0])
                estimated_y.append(pos[1])
        
        plt.scatter(estimated_x, estimated_y, 
                   c='red', marker='x', s=50, label='Estimated Positions')
        
        # Draw lines between ground truth and estimates
        for scan_id, pos in results.items():
            if pos is not None and scan_id in ground_truth.index:
                gt_pos = ground_truth.loc[scan_id, ['x', 'y']].values
                plt.plot([gt_pos[0], pos[0]], [gt_pos[1], pos[1]], 
                        'b--', alpha=0.3, linewidth=0.7)
        
        # Add labels and legend
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Calculate and display mean error
        errors = []
        for scan_id, pos in results.items():
            if pos is not None and scan_id in ground_truth.index:
                gt_pos = ground_truth.loc[scan_id, ['x', 'y']].values
                error = np.linalg.norm(pos - gt_pos)
                errors.append(error)
        
        if errors:
            mean_error = np.mean(errors)
            plt.text(0.02, 0.98, f"Mean Error: {mean_error:.2f} m", 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()

    def plot_measurement_quality(self, scans_df, wifis_df):
        """
        Visualize the quality of RTT and RSS measurements
        
        Args:
            scans_df: DataFrame from scans.csv
            wifis_df: DataFrame from wifis.csv
        """
        merged = pd.merge(scans_df, wifis_df, on='bssid', suffixes=('_scan', '_wifi'))
        
        # Calculate actual distances
        merged['actual_dist'] = np.sqrt(
            (merged['x_scan'] - merged['x_wifi'])**2 + 
            (merged['y_scan'] - merged['y_wifi'])**2
        )
        
        plt.figure(figsize=(12, 5))
        
        # Plot RTT vs actual
        plt.subplot(121)
        plt.scatter(merged['actual_dist'], merged['rttDist'], alpha=0.5)
        plt.plot([0, merged['actual_dist'].max()], [0, merged['actual_dist'].max()], 'r--')
        plt.xlabel('Actual Distance (m)')
        plt.ylabel('RTT Distance (m)')
        plt.title('RTT Measurement Quality')
        
        # Plot RSS-derived vs actual
        plt.subplot(122)
        d_squared = 10**((merged['txPower'] - merged['rssi']) / (5 * merged['pathLossExponent']))
        rss_dist = np.sqrt(d_squared)
        plt.scatter(merged['actual_dist'], rss_dist, alpha=0.5)
        plt.plot([0, merged['actual_dist'].max()], [0, merged['actual_dist'].max()], 'r--')
        plt.xlabel('Actual Distance (m)')
        plt.ylabel('RSS Distance (m)')
        plt.title('RSS Measurement Quality')
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    scans_df = pd.read_csv('scans.csv')
    wifis_df = pd.read_csv('wifis.csv')
    
    solver = TrilaterationSolver()
    
    # First visualize measurement quality
    print("Visualizing measurement quality...")
    solver.plot_measurement_quality(scans_df, wifis_df)
    
    # Get ground truth positions
    ground_truth = scans_df.groupby('scanId')[['x', 'y']].first()
    
    # Solve and visualize using different methods
    print("\nSolving with RTT measurements...")
    rtt_results = solver.solve_rtt_measurements(scans_df, wifis_df)
    solver.visualize_results(rtt_results, ground_truth, wifis_df, "RTT Positioning Results")
    
    print("\nSolving with RSS measurements...")
    rss_results = solver.solve_rss_measurements(scans_df, wifis_df)
    solver.visualize_results(rss_results, ground_truth, wifis_df, "RSS Positioning Results")
    
    print("\nSolving with combined RTT+RSS measurements...")
    combined_results = solver.solve_combined_measurements(scans_df, wifis_df)
    solver.visualize_results(combined_results, ground_truth, wifis_df, "Combined RTT+RSS Positioning Results")
    
    # Calculate and print mean errors
    def calculate_errors(results, ground_truth):
        errors = {}
        for scan_id, pos in results.items():
            if pos is not None and scan_id in ground_truth.index:
                gt_pos = ground_truth.loc[scan_id, ['x', 'y']].values
                error = np.linalg.norm(pos - gt_pos)
                errors[scan_id] = error
        return errors
    
    rtt_errors = calculate_errors(rtt_results, ground_truth)
    rss_errors = calculate_errors(rss_results, ground_truth)
    combined_errors = calculate_errors(combined_results, ground_truth)
    
    print("\nMean Positioning Errors:")
    print(f"RTT only: {np.mean(list(rtt_errors.values())):.2f} meters")
    print(f"RSS only: {np.mean(list(rss_errors.values())):.2f} meters")
    print(f"Combined RTT+RSS: {np.mean(list(combined_errors.values())):.2f} meters")