import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Constants import *
from utils import dB, dBm, parse_complex, process_noise_data

def load_gain_data(gain_file, gain_headers, fft_freq):
    """
    Load and interpolate gain data from CSV file to match frequency bins.
    
    Parameters:
        gain_file (str): Path to the gain CSV file.
        gain_headers (list): List of column headers [freq, ch1_gain, ch2_gain].
        fft_freq (np.ndarray): Frequency array in Hz for interpolation.
        
    Returns:
        tuple: A tuple containing:
            - ch1_gain_db (np.ndarray): Channel 1 gain in dB.
            - ch2_gain_db (np.ndarray): Channel 2 gain in dB.
            - phase_diff (np.ndarray): Phase difference in radians.
    """
    if not os.path.exists(gain_file):
        raise FileNotFoundError(f"Gain file '{gain_file}' does not exist.")
    
    # load complex gain data
    gain_df = pd.read_csv(gain_file, converters={
        gain_headers[1]: parse_complex, 
        gain_headers[2]: parse_complex
    })
    
    # interpolate gain values to match PSD frequency bins
    # convert power gain to dB: 20*log10(|S|) for voltage, but we want power so 2*20*log10 = 40*log10 = 2*dB
    freq_col = gain_headers[0] if gain_headers[0] in gain_df.columns else 'Frequency'
    source_freq = gain_df[freq_col] * 1e9  # convert GHz to Hz
    
    ch1_gain_db = np.interp(fft_freq, source_freq, 2 * dB(np.abs(gain_df[gain_headers[1]])))
    ch2_gain_db = np.interp(fft_freq, source_freq, 2 * dB(np.abs(gain_df[gain_headers[2]])))
    
    # calculate phase difference
    phase_diff = np.interp(fft_freq, source_freq, 
                          np.unwrap(np.angle(gain_df[gain_headers[1]] * np.conj(gain_df[gain_headers[2]]))))
    
    return ch1_gain_db, ch2_gain_db, phase_diff

def plot_psd_before_gain(fft_freq, ch1_avg_psd_dbm, ch2_avg_psd_dbm, date, output_dir):
    """
    Plot PSDs before gain adjustment.
    
    Parameters:
        fft_freq (np.ndarray): Frequency array in Hz.
        ch1_avg_psd_dbm (np.ndarray): Channel 1 PSD in dBm/Hz.
        ch2_avg_psd_dbm (np.ndarray): Channel 2 PSD in dBm/Hz.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(fft_freq / 1e9, ch1_avg_psd_dbm, label='Ch1 PSD', color=COLORS[0])
    plt.plot(fft_freq / 1e9, ch2_avg_psd_dbm, label='Ch2 PSD', color=COLORS[1])
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('PSD (dBm/Hz)')
    plt.title('Load Calibration: PSDs Before Gain Subtraction')
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"Load_PSDs_Before_Gain_{date}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_gain_comparison(fft_freq, cw_ch1_gain, cw_ch2_gain, nd_ch1_gain, nd_ch2_gain, date, output_dir):
    """
    Plot comparison of CW and ND gains.
    
    Parameters:
        fft_freq (np.ndarray): Frequency array in Hz.
        cw_ch1_gain (np.ndarray): CW channel 1 gain in dB.
        cw_ch2_gain (np.ndarray): CW channel 2 gain in dB.
        nd_ch1_gain (np.ndarray): ND channel 1 gain in dB.
        nd_ch2_gain (np.ndarray): ND channel 2 gain in dB.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
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
    
    plot_path = os.path.join(output_dir, f"Load_CW_ND_Gains_{date}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_gain_adjusted_results(fft_freq, cw_ch1_psd_adj, cw_ch2_psd_adj, cw_phase_diff,
                              nd_ch1_psd_adj, nd_ch2_psd_adj, nd_phase_diff, date, output_dir):
    """
    Plot gain-adjusted PSDs and phase differences for both CW and ND.
    
    Parameters:
        fft_freq (np.ndarray): Frequency array in Hz.
        cw_ch1_psd_adj (np.ndarray): CW channel 1 gain-adjusted PSD in dBm/Hz.
        cw_ch2_psd_adj (np.ndarray): CW channel 2 gain-adjusted PSD in dBm/Hz.
        cw_phase_diff (np.ndarray): CW phase difference in radians.
        nd_ch1_psd_adj (np.ndarray): ND channel 1 gain-adjusted PSD in dBm/Hz.
        nd_ch2_psd_adj (np.ndarray): ND channel 2 gain-adjusted PSD in dBm/Hz.
        nd_phase_diff (np.ndarray): ND phase difference in radians.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # plot CW-adjusted PSDs
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('PSD (dBm/Hz)')
    ax1.plot(fft_freq / 1e9, cw_ch1_psd_adj, label='CW Ch1 PSD', color=COLORS[0])
    ax1.plot(fft_freq / 1e9, cw_ch2_psd_adj, label='CW Ch2 PSD', color=COLORS[1])
    ax1.set_title('CW Gain-Adjusted PSDs')
    ax1.legend()

    # plot CW phase difference
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase Diff (degrees)')
    ax2.plot(fft_freq / 1e9, np.degrees(np.unwrap(cw_phase_diff)), label='CW Phase Diff', color=COLORS[2])
    ax2.set_title('CW Phase Difference')
    ax2.legend()

    # plot ND-adjusted PSDs
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('PSD (dBm/Hz)')
    ax3.plot(fft_freq / 1e9, nd_ch1_psd_adj, label='ND Ch1 PSD', color=COLORS[0])
    ax3.plot(fft_freq / 1e9, nd_ch2_psd_adj, label='ND Ch2 PSD', color=COLORS[1])
    ax3.set_title('ND Gain-Adjusted PSDs')
    ax3.legend()

    # plot ND phase difference
    ax4.set_xlabel('Frequency (GHz)')
    ax4.set_ylabel('Phase Diff (degrees)')
    ax4.plot(fft_freq / 1e9, np.degrees(np.unwrap(nd_phase_diff)), label='ND Phase Diff', color=COLORS[2])
    ax4.set_title('ND Phase Difference')
    ax4.legend()

    plt.suptitle('Load Calibration: CW and ND Gain-Adjusted Results')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"Load_CW_ND_Adjusted_{date}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path


def loadcal_main(date: str = LOAD_DATE, **kwargs) -> list[str]:
    """
    Process load calibration measurement data and calculate gain-adjusted PSDs.
    
    This function processes load calibration noise data, calculates average power 
    spectral densities, applies gain corrections from CW and ND measurements,
    and generates analysis plots and CSV output.
    
    Parameters:
        date (str): The date string used for file naming and data selection.
                   Defaults to LOAD_DATE from Constants.
        **kwargs: Additional keyword arguments.
            num_traces (int): Number of samples to process (default: NUM_TRACES).
            filename (str): Base filename pattern for load cal data files (default: LOAD_FILENAME).
            gain_file (str): Path to CW gain file (default: CW_GAIN_FILE).
            gain_headers (list): Column headers for CW gain file (default: CW_GAIN_HEADERS).
            nd_gain_file (str): Path to ND gain file (default: ND_GAIN_FILE).
            nd_gain_headers (list): Column headers for ND gain file (default: ND_GAIN_HEADERS).
            graph_pregain (bool): Plot PSDs before gain adjustment (default: True).
            graph_gain (bool): Plot gain comparison (default: True).
            graph_postgain (bool): Plot gain-adjusted results (default: True).

    Returns:
        list[str]: List of file paths to the generated CSV and plot files.

    Raises:
        ValueError: If num_traces is not a positive integer.
        FileNotFoundError: If required gain files do not exist.
    """
    # ensure output directory exists
    if not os.path.exists(LOAD_DIR):
        os.makedirs(LOAD_DIR)
    filepaths = []
    
    # initialization
    num_traces = kwargs.get("num_traces", NUM_TRACES)
    if num_traces <= 0:
        raise ValueError("num_traces must be a positive integer.")
    filename = kwargs.get("filename", LOAD_FILENAME).format(date=date)
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # process load calibration files and calculate average PSD
    ch1_avg_psd, ch2_avg_psd, csd_avg = process_noise_data(
        data_file_base=filename,
        num_traces=num_traces,
        fft_freq=fft_freq
    )

    # filter frequency range to 1-2 GHz
    fft_freq = fft_freq[(fft_freq >= 1e9) & (fft_freq <= 2e9)]

    # convert to dBm/Hz for analysis
    ch1_avg_psd_dbm = dBm(ch1_avg_psd / R_0)
    ch2_avg_psd_dbm = dBm(ch2_avg_psd / R_0)
    phase_diff = np.angle(csd_avg)

    # load CW and ND gain data
    cw_ch1_gain, cw_ch2_gain, cw_gain_phase_diff = load_gain_data(
        kwargs.get("gain_file", CW_GAIN_FILE),
        kwargs.get("gain_headers", CW_GAIN_HEADERS),
        fft_freq
    )
    
    nd_ch1_gain, nd_ch2_gain, nd_gain_phase_diff = load_gain_data(
        kwargs.get("nd_gain_file", ND_GAIN_FILE),
        kwargs.get("nd_gain_headers", ND_GAIN_HEADERS),
        fft_freq
    )

    # generate plots
    if kwargs.get("graph_pregain", True):
        pregain_plot_path = plot_psd_before_gain(fft_freq, ch1_avg_psd_dbm, ch2_avg_psd_dbm, date, LOAD_DIR)
        filepaths.append(pregain_plot_path)

    # subtract gains from PSDs to get gain-adjusted values
    cw_ch1_psd_gain_adj = ch1_avg_psd_dbm - cw_ch1_gain
    cw_ch2_psd_gain_adj = ch2_avg_psd_dbm - cw_ch2_gain
    cw_phase_diff = np.real(phase_diff - cw_gain_phase_diff)  # ensure real-valued

    nd_ch1_psd_gain_adj = ch1_avg_psd_dbm - nd_ch1_gain
    nd_ch2_psd_gain_adj = ch2_avg_psd_dbm - nd_ch2_gain
    nd_phase_diff = np.real(phase_diff - nd_gain_phase_diff)  # ensure real-valued

    if kwargs.get("graph_gain", True):
        gain_plot_path = plot_gain_comparison(fft_freq, cw_ch1_gain, cw_ch2_gain, nd_ch1_gain, nd_ch2_gain, date, LOAD_DIR)
        filepaths.append(gain_plot_path)

    if kwargs.get("graph_postgain", True):
        postgain_plot_path = plot_gain_adjusted_results(
            fft_freq, cw_ch1_psd_gain_adj, cw_ch2_psd_gain_adj, cw_phase_diff,
            nd_ch1_psd_gain_adj, nd_ch2_psd_gain_adj, nd_phase_diff, date, LOAD_DIR
        )
        filepaths.append(postgain_plot_path)

    # save results to CSV
    # convert gain-adjusted PSDs back to linear units (W/Hz) using safe conversion
    cw_ch1_psd_gain_adj_linear = linear(cw_ch1_psd_gain_adj) * 1e-3  # convert dBm/Hz to W/Hz
    cw_ch2_psd_gain_adj_linear = linear(cw_ch2_psd_gain_adj) * 1e-3  # convert dBm/Hz to W/Hz
    nd_ch1_psd_gain_adj_linear = linear(nd_ch1_psd_gain_adj) * 1e-3  # convert dBm/Hz to W/Hz
    nd_ch2_psd_gain_adj_linear = linear(nd_ch2_psd_gain_adj) * 1e-3  # convert dBm/Hz to W/Hz
    
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
    
    csv_output_path = os.path.join(LOAD_DIR, f"Processed_Load_{date}.csv")
    results_df.to_csv(csv_output_path, index=False)
    filepaths.append(csv_output_path)

    return filepaths

if __name__ == "__main__":
    loadcal_main()