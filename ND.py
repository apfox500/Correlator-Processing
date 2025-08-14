import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf

from Constants import *
from utils import dB, process_noise_data

def calculate_noise_gains(ch1_avg_psd, ch2_avg_psd, csd_avg, nd_psd_ref):
    """
    Calculate complex gains from power spectral densities using noise diode reference.
    
    Parameters:
        ch1_avg_psd (np.ndarray): Average PSD for channel 1 in V²/Hz.
        ch2_avg_psd (np.ndarray): Average PSD for channel 2 in V²/Hz.
        csd_avg (np.ndarray): Average cross-spectral density.
        nd_psd_ref (np.ndarray): Reference noise diode PSD in W/Hz.
        
    Returns:
        tuple: A tuple containing:
            - ch1_gain_complex (np.ndarray): Complex gain for channel 1.
            - ch2_gain_complex (np.ndarray): Complex gain for channel 2.
            - phase_diff (np.ndarray): Phase difference between channels in radians.
    """
    # protect against division by zero or very small values
    nd_psd_ref = np.where(nd_psd_ref > 1e-20, nd_psd_ref, 1e-20)
    
    # calculate power gains (PSD in V²/Hz converted to W/Hz by dividing by R_0)
    ch1_power_gain = (ch1_avg_psd / R_0) / nd_psd_ref
    ch2_power_gain = (ch2_avg_psd / R_0) / nd_psd_ref
    
    # calculate complex cross-gain for phase difference extraction
    csd_gain = (csd_avg / R_0) / nd_psd_ref
    
    # extract phase difference: φ1 - φ2 = angle(csd_gain)
    # cross-spectrum: CSD = G1 * G2* = |G1||G2| * e^(i*(φ1-φ2))
    phase_diff = np.angle(csd_gain)
    
    # convert power gains to voltage-equivalent gains
    # power gain = |S|², so voltage gain magnitude = sqrt(power_gain)
    ch1_power_gain = np.maximum(ch1_power_gain, 1e-20)  # ensure positive values
    ch2_power_gain = np.maximum(ch2_power_gain, 1e-20)  # ensure positive values
    
    ch1_gain_mag = np.sqrt(ch1_power_gain)
    ch2_gain_mag = np.sqrt(ch2_power_gain)
    
    # distribute phase difference between channels
    # convention: assign half the phase to each channel with opposite signs
    # preserves relative phase difference: angle(G1) - angle(G2) = phase_diff
    ch1_phase = phase_diff / 2
    ch2_phase = -phase_diff / 2
    
    # create complex gains: G = |G| * e^(i*φ)
    ch1_gain_complex = ch1_gain_mag * np.exp(1j * ch1_phase)
    ch2_gain_complex = ch2_gain_mag * np.exp(1j * ch2_phase)
    
    return ch1_gain_complex, ch2_gain_complex, phase_diff

def load_and_interpolate_sparameters(ch1_file, ch2_file, freq_axis_ghz):
    """
    Load S-parameter files and interpolate S21 values to the frequency axis.
    
    Parameters:
        ch1_file (str): Path to S-parameter file for channel 1.
        ch2_file (str): Path to S-parameter file for channel 2.
        freq_axis_ghz (np.ndarray): Frequency axis in GHz for interpolation.
        
    Returns:
        tuple: A tuple containing:
            - ch1_s21 (np.ndarray): Complex S21 for channel 1, or None if file not found.
            - ch2_s21 (np.ndarray): Complex S21 for channel 2, or None if file not found.
    """
    try:
        ch1_cal_net = rf.Network(ch1_file)
        ch2_cal_net = rf.Network(ch2_file)
        
        # extract magnitude and phase of S21
        ch1_s21_mag = np.abs(ch1_cal_net.s[:, 1, 0])
        ch1_s21_ang = np.angle(ch1_cal_net.s[:, 1, 0])
        ch2_s21_mag = np.abs(ch2_cal_net.s[:, 1, 0])
        ch2_s21_ang = np.angle(ch2_cal_net.s[:, 1, 0])

        # interpolate magnitude and phase separately
        ch1_s21_mag_interp = np.interp(freq_axis_ghz, ch1_cal_net.f/1e9, ch1_s21_mag)
        ch1_s21_ang_interp = np.interp(freq_axis_ghz, ch1_cal_net.f/1e9, ch1_s21_ang)
        ch2_s21_mag_interp = np.interp(freq_axis_ghz, ch2_cal_net.f/1e9, ch2_s21_mag)
        ch2_s21_ang_interp = np.interp(freq_axis_ghz, ch2_cal_net.f/1e9, ch2_s21_ang)

        # reconstruct complex S21
        ch1_s21 = ch1_s21_mag_interp * np.exp(1j * ch1_s21_ang_interp)
        ch2_s21 = ch2_s21_mag_interp * np.exp(1j * ch2_s21_ang_interp)
        
        return ch1_s21, ch2_s21
        
    except FileNotFoundError as e:
        print(f"S-parameter files not found: {e}. Skipping S-parameter correction.")
        return None, None

def apply_sparameter_corrections(ch1_gain_complex, ch2_gain_complex, csd_gain, ch1_s21, ch2_s21):
    """
    Apply S-parameter corrections to complex gains and recalculate phase difference.
    
    Parameters:
        ch1_gain_complex (np.ndarray): Complex gain for channel 1.
        ch2_gain_complex (np.ndarray): Complex gain for channel 2.
        csd_gain (np.ndarray): Complex cross-spectral density gain.
        ch1_s21 (np.ndarray): Complex S21 for channel 1.
        ch2_s21 (np.ndarray): Complex S21 for channel 2.
        
    Returns:
        tuple: A tuple containing:
            - ch1_gain_corrected (np.ndarray): Corrected complex gain for channel 1.
            - ch2_gain_corrected (np.ndarray): Corrected complex gain for channel 2.
            - phase_diff_corrected (np.ndarray): Corrected phase difference in radians.
    """
    # apply S-parameter corrections using voltage transfer function S21
    ch1_gain_corrected = ch1_gain_complex / ch1_s21
    ch2_gain_corrected = ch2_gain_complex / ch2_s21
    
    # correct cross-spectrum with complex S-parameters
    csd_gain_corrected = csd_gain / (ch1_s21 * np.conj(ch2_s21))
    
    # recalculate phase difference after corrections
    phase_diff_corrected = np.angle(ch1_gain_corrected) - np.angle(ch2_gain_corrected)
    
    return ch1_gain_corrected, ch2_gain_corrected, phase_diff_corrected

def plot_noise_results(freq_ghz, ch1_gain_complex, ch2_gain_complex, phase_diff, date, output_dir):
    """
    Plot gain and phase difference results for noise analysis.
    
    Parameters:
        freq_ghz (np.ndarray): Frequency array in GHz.
        ch1_gain_complex (np.ndarray): Complex gain for channel 1.
        ch2_gain_complex (np.ndarray): Complex gain for channel 2.
        phase_diff (np.ndarray): Phase difference in radians.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
    # convert to dB and degrees for plotting
    ch1_gain_db = 2 * dB(np.abs(ch1_gain_complex))
    ch2_gain_db = 2 * dB(np.abs(ch2_gain_complex))
    
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # plot gain in dB
    ax1.plot(freq_ghz, ch1_gain_db, label='Ch1 Average Gain', color=COLORS[4])
    ax1.plot(freq_ghz, ch2_gain_db, label='Ch2 Average Gain', color=COLORS[3])
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Gain (dB)")
    ax1.legend(loc="center left")

    # plot phase difference in degrees
    phase_diff_deg = np.degrees(np.unwrap(phase_diff))
    ax2.plot(freq_ghz, phase_diff_deg, label='Phase Difference', color=COLORS[2])
    ax2.set_ylabel("Phase Difference (degrees)")
    ax2.legend(loc="center right")

    plt.title("Average Gain and Phase Difference")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"Noise_Gain_Phase_{date}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

def noise_main(date: str = ND_DATE, **kwargs) -> list[str]:
    """
    Process noise diode measurement data and calculate calibrated channel gains.
    
    This function processes noise data files, calculates average power spectral densities,
    applies noise diode calibration and S-parameter corrections, then generates analysis
    plots and saves results to CSV.
    
    Parameters:
        date (str): The date string used for file naming and data selection.
                   Defaults to ND_DATE from Constants.
        **kwargs: Additional keyword arguments.
            num_traces (int): Number of samples to process (default: NUM_TRACES).
            filename (str): Base filename pattern for noise data files (default: ND_FILENAME).
            cal_file (str): Path to noise diode calibration file (default: CAL_FILE).
            ch1_file (str): Path to S-parameter file for channel 1 (default: CH1_FILE).
            ch2_file (str): Path to S-parameter file for channel 2 (default: CH2_FILE).

    Returns:
        list[str]: List of file paths to the generated CSV and plot files.

    Raises:
        ValueError: If num_traces is not a positive integer.
    """
    # ensure output directory exists
    if not os.path.exists(ND_DIR):
        os.makedirs(ND_DIR)
    filepaths = []
    
    # initialization
    num_traces = kwargs.get("num_traces", NUM_TRACES)
    if num_traces <= 0:
        raise ValueError("num_traces must be a positive integer.")
    filename = kwargs.get("filename", ND_FILENAME).format(date=date)
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # process noise files and calculate average PSD
    ch1_avg_psd, ch2_avg_psd, csd_avg = process_noise_data(
        data_file_base=filename,
        num_traces=num_traces,
        fft_freq=fft_freq
    )

    # load noise diode calibration data
    ND136_df = pd.read_csv(kwargs.get("cal_file", CAL_FILE), comment='#', header=None, names=['Freq (GHz)', 'NT', 'U_NT', 'ENR'])
    ND136_df = ND136_df.groupby('Freq (GHz)').mean().reset_index()
    ND136_df['PSD'] = ND136_df['NT'] * KB
    ND136_df_interpolated = pd.DataFrame({
        'Freq (GHz)': np.linspace(1, 2, DATASET_LENGTH),
        'PSD': np.interp(np.linspace(1, 2, DATASET_LENGTH), ND136_df['Freq (GHz)'], ND136_df['PSD'])
    })

    # calculate complex gains
    ch1_gain_complex, ch2_gain_complex, phase_diff = calculate_noise_gains(
        ch1_avg_psd, ch2_avg_psd, csd_avg, ND136_df_interpolated['PSD'].values
    )

    # apply S-parameter corrections
    freq_axis_ghz = np.linspace(1, 2, DATASET_LENGTH, dtype=float)
    ch1_s21, ch2_s21 = load_and_interpolate_sparameters(
        kwargs.get("ch1_file", CH1_FILE),
        kwargs.get("ch2_file", CH2_FILE),
        freq_axis_ghz
    )

    if ch1_s21 is not None and ch2_s21 is not None:
        # recalculate for S-parameter corrections
        csd_gain = (csd_avg / R_0) / ND136_df_interpolated['PSD'].values
        ch1_gain_complex, ch2_gain_complex, phase_diff = apply_sparameter_corrections(
            ch1_gain_complex, ch2_gain_complex, csd_gain, ch1_s21, ch2_s21
        )

    # save results to CSV
    results_df = pd.DataFrame({
        'Freq (GHz)': ND136_df_interpolated['Freq (GHz)'],
        'Ch1 Gain': ch1_gain_complex,  # store as complex
        'Ch2 Gain': ch2_gain_complex,  # store as complex  
        'Phase Difference (rad)': phase_diff,
    })
    csv_output_path = os.path.join(ND_DIR, f"Processed_Noise_{date}.csv")
    results_df.to_csv(csv_output_path, index=False)
    filepaths.append(csv_output_path)

    # generate plots
    plot_path = plot_noise_results(
        ND136_df_interpolated['Freq (GHz)'], 
        ch1_gain_complex, ch2_gain_complex, phase_diff, 
        date, ND_DIR
    )
    filepaths.append(plot_path)

    return filepaths

if __name__ == "__main__":
    noise_main()