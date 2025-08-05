import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Constants ---
# Use a raw string (r"...") or forward slashes for paths to avoid issues with backslashes.
# Please update FILE2_PATH to the path of the second CSV file you want to compare.
FILE1_PATH = 'correlator_testing_output/Processed_Pwr_vs_f_PSplitterNominalCWs-30dBm_Att20dB_25-07-16-16-22-50.csv'
FILE2_PATH = 'correlator_testing_output/Calculated_Gain_from_ND136_25-07-16-09-12-45.csv'

# Headers to be read from the CSV files
FILE1_HEADERS = ['Frequency', 'ch1_gain', 'ch2_gain', 'phase_diff']
FILE2_HEADERS = ['Freq (GHz)', 'Ch1 Gain (dB)', 'Ch2 Gain (dB)', 'Phase Difference (rad)']


# Colors for plotting
COLORS = ["#223A64","#fa753c", "#ffda49", "#a9d574", "#3ca590", "#626c9d"]

def plot_comparison(file1_path, file2_path, file1_headers, file2_headers):
    """
    Reads gain data from two CSV files, resamples the second to the first's
    frequency points using power averaging, and plots them for comparison.

    Args:
        file1_path (str): Path to the first CSV file (target sampling).
        file2_path (str): Path to the second CSV file (data to be resampled).
        file1_headers (list): List of headers for file 1 [frequency, ch1_gain, ch2_gain].
        file2_headers (list): List of headers for file 2 [frequency, ch1_gain, ch2_gain].
    """
    if not os.path.exists(file1_path):
        print(f"Error: File not found at {file1_path}")
        return

    if not os.path.exists(file2_path):
        print(f"Error: File not found at {file2_path}")
        print("Please update the FILE2_PATH constant in the script.")
        return


    # Get the column names from the header constants
    freq1_col, ch1_gain1_col, ch2_gain1_col, phase_diff1_col = file1_headers
    freq2_col, ch1_gain2_col, ch2_gain2_col, phase_diff2_col = file2_headers

    # Read the data from the CSV files
    df1 = pd.read_csv(file1_path)
    # Use converters for complex columns in df2
    converters = {
        ch1_gain2_col: complex,
        ch2_gain2_col: complex
    }
    df2 = pd.read_csv(file2_path, converters=converters)


    # --- Resample df2 to df1's frequency grid using power averaging ---

    # define the bin edges for resampling
    # the edges are the midpoints between df1's frequency points
    bin_edges = (df1[freq1_col].values[:-1] + df1[freq1_col].values[1:]) / 2.0
    
    # add -inf and +inf to capture all points in df2
    bin_edges = np.insert(bin_edges, 0, -np.inf)
    bin_edges = np.append(bin_edges, np.inf)

    # assign each frequency in df2 to a bin corresponding to df1's frequencies
    df2['freq_bin'] = pd.cut(df2[freq2_col], bins=bin_edges, labels=df1[freq1_col].values)

    def power_avg_agg(x_db):
        # correctly average dB values by converting to linear (power),
        # averaging, and converting back to dB
        linear_power = 10**(abs(x_db) / 10.0)
        mean_power = np.mean(linear_power)
        # handle cases where mean_power is zero or negative to avoid log errors
        if mean_power <= 0:
            return np.nan
        return 10 * np.log10(mean_power)

    # group by the new frequency bins and apply the power averaging
    resampled_ch1 = df2.groupby('freq_bin', observed=False)[ch1_gain2_col].apply(power_avg_agg)
    resampled_ch2 = df2.groupby('freq_bin', observed=False)[ch2_gain2_col].apply(power_avg_agg)

    # create a new DataFrame from the resampled data
    df2_resampled = pd.DataFrame({
        freq1_col: resampled_ch1.index.astype(float), # ensure index is float for merging
        'resampled_ch1_gain': resampled_ch1.values,
        'resampled_ch2_gain': resampled_ch2.values
    })

    # merge the resampled data with the original df1 to align data points
    comparison_df = pd.merge(df1, df2_resampled, on=freq1_col, how='left')

    # --- Resample phase difference from df2 to df1's frequency grid ---

    def circular_mean(angles):
        # angles in radians
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        return np.arctan2(sin_sum, cos_sum)

    resampled_phase = df2.groupby('freq_bin', observed=False)[phase_diff2_col].apply(circular_mean)

    # Add resampled phase to the resampled DataFrame
    df2_resampled['nd_phase_diff'] = resampled_phase.values

    # Merge again to include the resampled phase in the comparison DataFrame
    comparison_df = pd.merge(comparison_df, df2_resampled[[freq1_col, 'nd_phase_diff']], on=freq1_col, how='left', suffixes=('', '_resampled'))

        # Plot phase difference
    plt.figure(figsize=(12, 8))
    plt.plot(comparison_df[freq1_col], np.degrees(np.unwrap(comparison_df[phase_diff1_col])), label='Expected - Phase Diff', color=COLORS[0])
    plt.plot(df2[freq2_col], np.degrees(np.unwrap(df2[phase_diff2_col])), label='Actual - Phase Diff', color=COLORS[4], alpha=0.5)
    plt.title('Phase Difference Comparison: Expected vs. Actual')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Phase Difference (rad)')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/apf1/Library/CloudStorage/OneDrive-NIST/SURF/Images/Graphs/Phase Diff CW vs Noise.png', bbox_inches='tight')
    plt.show()

    # plot resampled data
    plt.figure(figsize=(12, 8))
    plt.plot(comparison_df[freq1_col], np.degrees(np.unwrap(comparison_df[phase_diff1_col])), label='Expected - Phase Diff', color=COLORS[0])
    plt.plot(comparison_df[freq1_col], np.degrees(np.unwrap(comparison_df['phase_diff'])), label='Actual (Resampled) - Phase Diff', color=COLORS[4], alpha=0.5)
    plt.title('Phase Difference Comparison: Expected vs. Resampled Actual')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Phase Difference (rad)')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/apf1/Library/CloudStorage/OneDrive-NIST/SURF/Images/Graphs/Phase Diff CW vs Noise (resampled).png', bbox_inches='tight')
    plt.show()


    # --- Plotting ---
    plt.figure(figsize=(12, 8))

    # plot data from file 1 (original reference)
    plt.plot(comparison_df[freq1_col], comparison_df[ch1_gain1_col], label='Expected - CH1 Gain', color=COLORS[0])
    plt.plot(comparison_df[freq1_col], comparison_df[ch2_gain1_col], label='Expected - CH2 Gain', color=COLORS[1])
    

    plt.plot(df2[freq2_col], df2[ch1_gain2_col], label='Actual - CH1 Gain', color=COLORS[4], alpha=0.5)
    plt.plot(df2[freq2_col], df2[ch2_gain2_col], label='Actual - CH2 Gain', linestyle='-', color=COLORS[3], alpha=0.5)

    # add plot details
    plt.title('Gain Comparison: Expected vs. Actual')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/apf1/Library/CloudStorage/OneDrive-NIST/SURF/Images/Graphs/CW vs Noise.png', bbox_inches='tight')
    plt.show()

     # --- Plotting ---
    plt.figure(figsize=(12, 8))

    # plot data from file 1 (original reference)
    plt.plot(comparison_df[freq1_col], comparison_df[ch1_gain1_col], label='Expected - CH1 Gain', color=COLORS[0])
    plt.plot(comparison_df[freq1_col], comparison_df[ch2_gain1_col], label='Expected - CH2 Gain', color=COLORS[1])
    
    # plot resampled (smoothed) data from file 2
    plt.plot(comparison_df[freq1_col], comparison_df['resampled_ch1_gain'], label='Actual (Resampled) - CH1 Gain', color=COLORS[4], linestyle='-')
    plt.plot(comparison_df[freq1_col], comparison_df['resampled_ch2_gain'], label='Actual (Resampled) - CH2 Gain', color=COLORS[3], linestyle='-')

    # add plot details
    plt.title('Gain Comparison: Expected vs. Resampled Actual')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/apf1/Library/CloudStorage/OneDrive-NIST/SURF/Images/Graphs/CW vs Noise (resampled).png', bbox_inches='tight')
    plt.show()




    # Save the comparison data to CSV with specified columns and names
    output_df = pd.DataFrame({
        'Freq': comparison_df[freq1_col],
        'CW_Ch1': comparison_df[ch1_gain1_col],
        'CW_Ch2': comparison_df[ch2_gain1_col],
        'CW_Phase_Diff': comparison_df[phase_diff1_col],
        'ND_Ch1': comparison_df['resampled_ch1_gain'],
        'ND_Ch2': comparison_df['resampled_ch2_gain'],
        'ND_Phase_Diff': comparison_df['nd_phase_diff']
    })
    output_df.to_csv('./correlator_testing_output/Processed gain.csv', index=False)




if __name__ == '__main__':
    plot_comparison(FILE1_PATH, FILE2_PATH, FILE1_HEADERS, FILE2_HEADERS)
