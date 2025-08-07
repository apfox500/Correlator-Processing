
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
            - num_samples_test (int): Number of samples to process for testing (default: NUM_SAMPLES_TEST).
            - data_file_base (str): Base filename pattern for load cal data files (default: LOAD_FILENAME).

    Returns:
        list[str]: List of file paths to the generated CSV file.

    Notes:
        - Requires external files: load cal noise data.
        - The function saves results in OUT_DIR and expects certain constants (e.g., LOAD_FILENAME, SAMPLE_CUTOFF, FS, OUT_DIR) to be defined elsewhere.
    """
    filepaths = []
    # -------------------- Initialization --------------------
    num_samples_test = kwargs.get("num_samples_test", NUM_SAMPLES_TEST)
    if num_samples_test <= 0:
        raise ValueError("num_samples_test must be a positive integer.")
    data_file_base = kwargs.get("data_file_base", LOAD_FILENAME).format(date=date)
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # -------------------- Process Load Cal Files and Calculate Average PSD --------------------
    ch1_avg_psd, ch2_avg_psd, csd_avg = process_noise_data(
        data_file_base=data_file_base,
        num_samples_test=num_samples_test,
        fft_freq=fft_freq
    )

    fft_freq = fft_freq[(fft_freq >= 1e9) & (fft_freq <= 2e9)]


    # Convert to dBm/Hz
    ch1_avg_psd_dbm = dBm(ch1_avg_psd / IMPEDANCE)
    ch2_avg_psd_dbm = dBm(ch2_avg_psd / IMPEDANCE)

    phase_diff = np.angle(csd_avg)

    # -------------------- Account for gain --------------------
    # Load gain data from CSV file

    gain_file = kwargs.get("gain_file", GAIN_FILE)
    if not os.path.exists(gain_file):
        raise FileNotFoundError(f"Gain file '{gain_file}' does not exist.")
    
    gain_headers = kwargs.get("gain_headers", GAIN_HEADERS) # should be like [Freq, S31, S46]

    gain_df = pd.read_csv(gain_file, converters={gain_headers[1]: parse_complex, gain_headers[2]: parse_complex})

    # Interpolate the complex gain values to match the PSD frequency bins
    ch1_gain = np.interp(fft_freq, gain_df['Frequency'] * 1e9, 2 * dB(np.abs(gain_df[gain_headers[1]])))
    ch2_gain = np.interp(fft_freq, gain_df['Frequency'] * 1e9, 2 * dB(np.abs(gain_df[gain_headers[2]])))
    gain_phase_diff = np.angle(ch1_gain * np.conj(ch2_gain))

    if(kwargs.get("graph_pregain", True)):
        plt.figure(figsize=(8, 5))
        plt.plot(fft_freq / 1e9, ch1_avg_psd_dbm, label='Ch1 PSD', color=COLORS[0])
        plt.plot(fft_freq / 1e9, ch2_avg_psd_dbm, label='Ch2 PSD', color=COLORS[1])
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('PSD (dBm/Hz)')
        plt.title('Load Calibration: PSDs Before Gain Subtraction')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # subtract gain from PSDs
    ch1_psd_gain_adj = ch1_avg_psd_dbm - ch1_gain
    ch2_psd_gain_adj = ch2_avg_psd_dbm - ch2_gain
    phase_diff -= gain_phase_diff

    
    if kwargs.get("graph_gain", True):
        plt.figure(figsize=(8, 5))
        plt.plot(fft_freq / 1e9, ch1_gain, label='Ch1 Gain (dB)', color=COLORS[0])
        plt.plot(fft_freq / 1e9, ch2_gain, label='Ch2 Gain (dB)', color=COLORS[1])
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Gain (dB)')
        plt.title('Channel Gains vs Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    

    graph_postgain = kwargs.get("graph_postgain", True)
    if(graph_postgain):

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot PSDs on left y-axis
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('PSD (dBm/Hz)')
        ax1.plot(fft_freq / 1e9, ch1_psd_gain_adj, label='Ch1 PSD', color=COLORS[0])
        ax1.plot(fft_freq / 1e9, ch2_psd_gain_adj, label='Ch2 PSD', color=COLORS[1])
        ax1.legend(loc='upper left')

        # Plot phase diff on right y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Phase Diff (degrees)')
        ax2.plot(fft_freq / 1e9, np.degrees(np.unwrap(phase_diff)), label='Phase Diff', color=COLORS[2], linestyle='-')
        ax2.legend(loc='upper right')

        plt.title('Load Calibration: PSDs and Phase Difference (with gain subtracted)')
        plt.tight_layout()
        plt.show()

    # -------------------- Save data to csv --------------------
    results_df = pd.DataFrame({
        'Freq': fft_freq / 1e9,
        'Ch1 PSD': ch1_avg_psd,
        'Ch2 PSD': ch2_avg_psd,
        "Ch1 PSD (gain adjusted)": ch1_psd_gain_adj,
        "Ch2 PSD (gain adjusted)": ch2_psd_gain_adj,
        'Phase Diff': phase_diff,
    })
    filepaths.append(LOAD_FILE)
    results_df.to_csv(filepaths[-1], index=False)

    return filepaths
    