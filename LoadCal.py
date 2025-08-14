
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Constants import *
from utils import *


def loadcal_main(date:str = LOAD_DATE, **kwargs) -> list[str]:
    """
    Processes load calibration noise data for a given date, calculates average power spectral densities (PSDs) for both channels,
    computes the phase difference, and saves results to CSV.

    Parameters:
        date (str): The date string used for file naming and data selection.
        **kwargs: Optional keyword arguments:
            - num_traces (int): Number of samples to process for testing (default: NUM_TRACES).
            - filename (str): Base filename pattern for load cal data files (default: LOAD_FILENAME).

    Returns:
        list[str]: List of file paths to the generated CSV file.

    Notes:
        - Requires external files: load cal noise data.
        - The function saves results in OUT_DIR and expects certain constants (e.g., LOAD_FILENAME, SAMPLE_CUTOFF, FS, OUT_DIR) to be defined elsewhere.
    """
    # ensure output directory exists
    if not os.path.exists(LOAD_DIR):
        os.makedirs(LOAD_DIR)
    filepaths = []
    # -------------------- Initialization --------------------
    num_traces = kwargs.get("num_traces", NUM_TRACES)
    if num_traces <= 0:
        raise ValueError("num_traces must be a positive integer.")
    filename = kwargs.get("filename", LOAD_FILENAME).format(date=date)
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # -------------------- Process Load Cal Files and Calculate Average PSD --------------------
    ch1_avg_psd, ch2_avg_psd, csd_avg = process_noise_data(
        data_file_base=filename,
        num_traces=num_traces,
        fft_freq=fft_freq
    )

    fft_freq = fft_freq[(fft_freq >= 1e9) & (fft_freq <= 2e9)]


    # Convert to dBm/Hz
    ch1_avg_psd_dbm = dBm(ch1_avg_psd / R_0)
    ch2_avg_psd_dbm = dBm(ch2_avg_psd / R_0)

    phase_diff = np.angle(csd_avg)

    # -------------------- Account for gain --------------------
    # Load CW gain data from CSV file
    cw_gain_file = kwargs.get("gain_file", CW_GAIN_FILE)
    if not os.path.exists(cw_gain_file):
        raise FileNotFoundError(f"CW gain file '{cw_gain_file}' does not exist.")
    
    cw_gain_headers = kwargs.get("gain_headers", CW_GAIN_HEADERS) # should be like [Freq, S31, S46]
    cw_gain_df = pd.read_csv(cw_gain_file, converters={cw_gain_headers[1]: parse_complex, cw_gain_headers[2]: parse_complex})

    # Interpolate the CW complex gain values to match the PSD frequency bins
    cw_ch1_gain = np.interp(fft_freq, cw_gain_df['Frequency'] * 1e9, 2 * dB(np.abs(cw_gain_df[cw_gain_headers[1]]))) # square to get into dBW
    cw_ch2_gain = np.interp(fft_freq, cw_gain_df['Frequency'] * 1e9, 2 * dB(np.abs(cw_gain_df[cw_gain_headers[2]])))
    cw_gain_phase_diff = np.interp(fft_freq, cw_gain_df['Frequency'] * 1e9, np.unwrap(np.angle(cw_gain_df[cw_gain_headers[1]] * np.conj(cw_gain_df[cw_gain_headers[2]]))))

    # Load ND gain data from CSV file
    nd_gain_file = kwargs.get("nd_gain_file", ND_GAIN_FILE)
    if not os.path.exists(nd_gain_file):
        raise FileNotFoundError(f"ND gain file '{nd_gain_file}' does not exist.")
    
    nd_gain_headers = kwargs.get("nd_gain_headers", ND_GAIN_HEADERS)
    nd_gain_df = pd.read_csv(nd_gain_file, converters={nd_gain_headers[1]: parse_complex, nd_gain_headers[2]: parse_complex})

    # Interpolate the ND gain values to match the PSD frequency bins
    nd_ch1_gain = np.interp(fft_freq, nd_gain_df[nd_gain_headers[0]] * 1e9, 2 * dB(np.abs(nd_gain_df[nd_gain_headers[1]])))  # square to get into dBW
    nd_ch2_gain = np.interp(fft_freq, nd_gain_df[nd_gain_headers[0]] * 1e9, 2 * dB(np.abs(nd_gain_df[nd_gain_headers[2]])))
    nd_gain_phase_diff = np.interp(fft_freq, nd_gain_df[nd_gain_headers[0]] * 1e9, np.unwrap(np.angle(nd_gain_df[nd_gain_headers[1]] * np.conj(nd_gain_df[nd_gain_headers[2]]))))

    # -------------------- Graphing --------------------

    if(kwargs.get("graph_pregain", True)):
        plt.figure(figsize=(8, 5))
        plt.plot(fft_freq / 1e9, ch1_avg_psd_dbm, label='Ch1 PSD', color=COLORS[0])
        plt.plot(fft_freq / 1e9, ch2_avg_psd_dbm, label='Ch2 PSD', color=COLORS[1])
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('PSD (dBm/Hz)')
        plt.title('Load Calibration: PSDs Before Gain Subtraction')
        plt.legend()
        plt.tight_layout()
        pregain_plot_path = os.path.join(LOAD_DIR, f"Load_PSDs_Before_Gain_{date}.png")
        plt.savefig(pregain_plot_path, bbox_inches='tight')
        filepaths.append(pregain_plot_path)
        plt.close()

    # subtract CW gain from PSDs
    cw_ch1_psd_gain_adj = ch1_avg_psd_dbm - cw_ch1_gain
    cw_ch2_psd_gain_adj = ch2_avg_psd_dbm - cw_ch2_gain
    cw_phase_diff = phase_diff - cw_gain_phase_diff

    # subtract ND gain from PSDs
    nd_ch1_psd_gain_adj = ch1_avg_psd_dbm - nd_ch1_gain
    nd_ch2_psd_gain_adj = ch2_avg_psd_dbm - nd_ch2_gain
    nd_phase_diff = phase_diff - nd_gain_phase_diff

    if kwargs.get("graph_gain", True):
        plt.figure(figsize=(10, 6))
        plt.plot(fft_freq / 1e9, cw_ch1_gain, label='CW Ch1 Gain (dB)', color=COLORS[0], linestyle='-')
        plt.plot(fft_freq / 1e9, cw_ch2_gain, label='CW Ch2 Gain (dB)', color=COLORS[0], linestyle='--')
        plt.plot(fft_freq / 1e9, nd_ch1_gain, label='ND Ch1 Gain (dB)', color=COLORS[1], linestyle='-')
        plt.plot(fft_freq / 1e9, nd_ch2_gain, label='ND Ch2 Gain (dB)', color=COLORS[1], linestyle='--')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Gain (dB)')
        plt.title('CW and ND Channel Gains vs Frequency')
        plt.legend()
        plt.tight_layout()
        gain_plot_path = os.path.join(LOAD_DIR, f"Load_CW_ND_Gains_{date}.png")
        plt.savefig(gain_plot_path, bbox_inches='tight')
        filepaths.append(gain_plot_path)
        plt.close()

    graph_postgain = kwargs.get("graph_postgain", True)
    if(graph_postgain):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot CW-adjusted PSDs
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('PSD (dBm/Hz)')
        ax1.plot(fft_freq / 1e9, cw_ch1_psd_gain_adj, label='CW Ch1 PSD', color=COLORS[0])
        ax1.plot(fft_freq / 1e9, cw_ch2_psd_gain_adj, label='CW Ch2 PSD', color=COLORS[1])
        ax1.set_title('CW Gain-Adjusted PSDs')
        ax1.legend()

        # Plot CW phase difference
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Phase Diff (degrees)')
        ax2.plot(fft_freq / 1e9, np.degrees(np.unwrap(cw_phase_diff)), label='CW Phase Diff', color=COLORS[2])
        ax2.set_title('CW Phase Difference')
        ax2.legend()

        # Plot ND-adjusted PSDs
        ax3.set_xlabel('Frequency (GHz)')
        ax3.set_ylabel('PSD (dBm/Hz)')
        ax3.plot(fft_freq / 1e9, nd_ch1_psd_gain_adj, label='ND Ch1 PSD', color=COLORS[0])
        ax3.plot(fft_freq / 1e9, nd_ch2_psd_gain_adj, label='ND Ch2 PSD', color=COLORS[1])
        ax3.set_title('ND Gain-Adjusted PSDs')
        ax3.legend()

        # Plot ND phase difference
        ax4.set_xlabel('Frequency (GHz)')
        ax4.set_ylabel('Phase Diff (degrees)')
        ax4.plot(fft_freq / 1e9, np.degrees(np.unwrap(nd_phase_diff)), label='ND Phase Diff', color=COLORS[2])
        ax4.set_title('ND Phase Difference')
        ax4.legend()

        plt.suptitle('Load Calibration: CW and ND Gain-Adjusted Results')
        plt.tight_layout()
        postgain_plot_path = os.path.join(LOAD_DIR, f"Load_CW_ND_Adjusted_{date}.png")
        plt.savefig(postgain_plot_path, bbox_inches='tight')
        filepaths.append(postgain_plot_path)
        plt.close()

    # -------------------- Save data to csv --------------------
    # Convert gain-adjusted PSDs back to linear units (W/Hz)
    cw_ch1_psd_gain_adj_linear = 10**(cw_ch1_psd_gain_adj / 10) * 1e-3  # Convert dBm/Hz to W/Hz
    cw_ch2_psd_gain_adj_linear = 10**(cw_ch2_psd_gain_adj / 10) * 1e-3  # Convert dBm/Hz to W/Hz
    nd_ch1_psd_gain_adj_linear = 10**(nd_ch1_psd_gain_adj / 10) * 1e-3  # Convert dBm/Hz to W/Hz
    nd_ch2_psd_gain_adj_linear = 10**(nd_ch2_psd_gain_adj / 10) * 1e-3  # Convert dBm/Hz to W/Hz
    
    results_df = pd.DataFrame({
        'Freq': fft_freq / 1e9,
        'Ch1 PSD': ch1_avg_psd,
        'Ch2 PSD': ch2_avg_psd,
        'CW Ch1 PSD (gain adjusted)': cw_ch1_psd_gain_adj_linear,
        'CW Ch2 PSD (gain adjusted)': cw_ch2_psd_gain_adj_linear,
        'CW Phase Diff': cw_phase_diff,
        'ND Ch1 PSD (gain adjusted)': nd_ch1_psd_gain_adj_linear,
        'ND Ch2 PSD (gain adjusted)': nd_ch2_psd_gain_adj_linear,
        'ND Phase Diff': nd_phase_diff,
    })
    filepaths.append(os.path.join(LOAD_DIR, f"Processed_Load_{date}.csv"))
    results_df.to_csv(filepaths[-1], index=False)

    return filepaths
    

if __name__ == "__main__":
    loadcal_main()