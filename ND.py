import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf

from Constants import *
from utils import process_noise_data, dB, linear


def noise_main(date:str, **kwargs) -> list[str]:
    """
    Processes noise data for a given date, calculates average power spectral densities (PSDs), calibrates using noise diode data,
    applies S-parameter corrections, computes channel gains and phase differences, saves results to CSV, and generates plots.

    Parameters:
        date (str): The date string used for file naming and data selection.
        **kwargs: Optional keyword arguments:
            - num_samples_test (int): Number of samples to process for testing (default: 500).
            - data_file_base (str): Base filename pattern for noise data files (default: ND_FILENAME).
            - ch1_file (str): Path to S-parameter file for channel 1 (default: CH1_FILE).
            - ch2_file (str): Path to S-parameter file for channel 2 (default: CH2_FILE).

    Returns:
        list[str]: List of file paths to the generated CSV and plot files.

    Raises:
        ValueError: If num_samples_test is not a positive integer.

    Notes:
        - Requires external files: noise data, noise diode calibration data, and S-parameter files.
        - The function saves results in OUT_DIR and expects certain constants (e.g., ND_FILENAME, SAMPLE_CUTOFF, FS, BOLTZ, OUT_DIR, COLORS) to be defined elsewhere.
    """

    filepaths = []
    # -------------------- Initialization --------------------
    num_samples_test = kwargs.get("num_samples_test", NUM_SAMPLES_TEST)
    if num_samples_test <= 0:
        raise ValueError("num_samples_test must be a positive integer.")
    data_file_base = kwargs.get("data_file_base", ND_FILENAME).format(date=date)
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # -------------------- Process Noise Files and Calculate Average PSD --------------------
    ch1_avg_psd, ch2_avg_psd, csd_avg = process_noise_data(
        data_file_base=data_file_base,
        num_samples_test=num_samples_test,
        fft_freq=fft_freq
    )

    # -------------------- Noise Diode Calibration Data --------------------
    ND136_df = pd.read_csv(kwargs.get("cal_file", CAL_FILE), comment='#', header=None, names=['Freq (GHz)', 'NT', 'U_NT', 'ENR'])
    ND136_df = ND136_df.groupby('Freq (GHz)').mean().reset_index()
    ND136_df['PSD'] = ND136_df['NT'] * BOLTZ
    ND136_df_interpolated = pd.DataFrame({
        'Freq (GHz)': np.linspace(1, 2, 3001), # How can  make the bins be 5 MHz like the CW
        'PSD': np.interp(np.linspace(1, 2, 3001), ND136_df['Freq (GHz)'], ND136_df['PSD'])
    })

    # -------------------- Gain Calculation --------------------
    ch1_gain = ch1_avg_psd / (IMPEDANCE * ND136_df_interpolated['PSD'].values)
    ch2_gain = ch2_avg_psd / (IMPEDANCE * ND136_df_interpolated['PSD'].values)

    phase_diff = np.angle(csd_avg)

    # --------------------- S-parameter ---------------------
    try:
        ch1_cal_net = rf.Network(kwargs.get("ch1_file", CH1_FILE))
        ch2_cal_net = rf.Network(kwargs.get("ch2_file", CH2_FILE))
        # Interpolate to 1-2 GHz with 2001 steps
        n_points = 3001
        freq_axis_ghz = np.linspace(1, 2, n_points, dtype=float)

        # Interpolate magnitude and angle separately, then reconstruct complex S21
        ch1_s21_mag = np.abs(ch1_cal_net.s[:, 1, 0])
        ch1_s21_ang = np.angle(ch1_cal_net.s[:, 1, 0])
        ch2_s21_mag = np.abs(ch2_cal_net.s[:, 1, 0])
        ch2_s21_ang = np.angle(ch2_cal_net.s[:, 1, 0])

        # Interpolate magnitude and angle
        ch1_s21_mag_interp = np.interp(freq_axis_ghz, ch1_cal_net.f/1e9, ch1_s21_mag)
        ch1_s21_ang_interp = np.interp(freq_axis_ghz, ch1_cal_net.f/1e9, ch1_s21_ang)
        ch2_s21_mag_interp = np.interp(freq_axis_ghz, ch2_cal_net.f/1e9, ch2_s21_mag)
        ch2_s21_ang_interp = np.interp(freq_axis_ghz, ch2_cal_net.f/1e9, ch2_s21_ang)

        # Reconstruct complex S21
        ch1_cal = ch1_s21_mag_interp * np.exp(1j * ch1_s21_ang_interp)
        ch2_cal = ch2_s21_mag_interp * np.exp(1j * ch2_s21_ang_interp)
        ch1_s21 = ch1_cal
        ch2_s21 = ch2_cal
    except FileNotFoundError as e:
        print(f"S-parameter files not found: {e}. Skipping S-parameter correction.")
        ch1_s21 = None
        ch2_s21 = None

    if ch1_s21 is not None and ch2_s21 is not None:
        ch1_gain /= np.abs(ch1_s21) ** 2
        ch2_gain /= np.abs(ch2_s21) ** 2

    ch1_gain = dB(ch1_gain)
    ch2_gain = dB(ch2_gain)

    # -------------------- Save data to csv --------------------
    results_df = pd.DataFrame({
        'Freq (GHz)': ND136_df_interpolated['Freq (GHz)'],
        'Ch1 Gain (dB)': ch1_gain,
        'Ch2 Gain (dB)': ch2_gain,
        'Phase Difference (rad)': phase_diff,
    })
    filepaths.append(os.path.join(OUT_DIR, f"ND_Gain_{date}.csv"))
    results_df.to_csv(filepaths[-1], index=False)

    # -------------------- Plotting Results --------------------
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot gain
    ax1.plot(ND136_df_interpolated['Freq (GHz)'], ch1_gain, label='Ch1 Average Gain', color=COLORS[4])
    ax1.plot(ND136_df_interpolated['Freq (GHz)'], ch2_gain, label='Ch2 Average Gain', color=COLORS[3])
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Gain (dB)")
    ax1.legend(loc="middle left")

    # Plot phase difference (convert to degrees)
    phase_diff_deg = np.degrees(np.unwrap(phase_diff))
    ax2.plot(ND136_df_interpolated['Freq (GHz)'], phase_diff_deg, label='Phase Difference', color=COLORS[2])
    ax2.set_ylabel("Phase Difference (degrees)")
    ax2.legend(loc="middle right")

    plt.title("Average Gain and Phase Difference")
    plt.tight_layout()
    filepaths.append(os.path.join(OUT_DIR, f"ND_Gain_Phase_{date}.png"))
    plt.savefig(filepaths[-1], bbox_inches='tight')
    plt.show()

    return filepaths
