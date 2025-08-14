import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf

from Constants import *
from utils import process_noise_data, dB, linear


def noise_main(date:str = ND_DATE, **kwargs) -> list[str]:
    """
    Processes noise data for a given date, calculates average power spectral densities (PSDs), calibrates using noise diode data,
    applies S-parameter corrections, computes channel gains and phase differences, saves results to CSV, and generates plots.

    Parameters:
        date (str): The date string used for file naming and data selection.
        **kwargs: Optional keyword arguments:
            - num_traces (int): Number of samples to process for testing (default: 500).
            - filename (str): Base filename pattern for noise data files (default: ND_FILENAME).
            - ch1_file (str): Path to S-parameter file for channel 1 (default: CH1_FILE).
            - ch2_file (str): Path to S-parameter file for channel 2 (default: CH2_FILE).

    Returns:
        list[str]: List of file paths to the generated CSV and plot files.

    Raises:
        ValueError: If num_traces is not a positive integer.

    Notes:
        - Requires external files: noise data, noise diode calibration data, and S-parameter files.
        - The function saves results in OUT_DIR and expects certain constants (e.g., ND_FILENAME, SAMPLE_CUTOFF, FS, BOLTZ, OUT_DIR, COLORS) to be defined elsewhere.
    """
    # ensure output directory exists
    if not os.path.exists(ND_DIR):
        os.makedirs(ND_DIR)
    filepaths = []
    # -------------------- Initialization --------------------
    num_traces = kwargs.get("num_traces", NUM_TRACES)
    if num_traces <= 0:
        raise ValueError("num_traces must be a positive integer.")
    filename = kwargs.get("filename", ND_FILENAME).format(date=date)
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # -------------------- Process Noise Files and Calculate Average PSD --------------------
    ch1_avg_psd, ch2_avg_psd, csd_avg = process_noise_data(
        data_file_base=filename,
        num_traces=num_traces,
        fft_freq=fft_freq
    )

    # -------------------- Noise Diode Calibration Data --------------------
    ND136_df = pd.read_csv(kwargs.get("cal_file", CAL_FILE), comment='#', header=None, names=['Freq (GHz)', 'NT', 'U_NT', 'ENR'])
    ND136_df = ND136_df.groupby('Freq (GHz)').mean().reset_index()
    ND136_df['PSD'] = ND136_df['NT'] * KB
    ND136_df_interpolated = pd.DataFrame({
        'Freq (GHz)': np.linspace(1, 2, DATASET_LENGTH),
        'PSD': np.interp(np.linspace(1, 2, DATASET_LENGTH), ND136_df['Freq (GHz)'], ND136_df['PSD'])
    })

    # -------------------- Gain Calculation --------------------
    # Calculate power gains first (this is what ND fundamentally measures)
    # Convert PSD (V²/Hz) to power (W/Hz) by dividing by R_0
    
    # Protect against division by zero or very small values
    nd_psd_ref = ND136_df_interpolated['PSD'].values
    nd_psd_ref = np.where(nd_psd_ref > 1e-20, nd_psd_ref, 1e-20)  # Avoid division by zero
    
    ch1_power_gain = (ch1_avg_psd / R_0) / nd_psd_ref
    ch2_power_gain = (ch2_avg_psd / R_0) / nd_psd_ref
    
    # For cross-spectrum, calculate the complex cross-gain
    csd_gain = (csd_avg / R_0) / nd_psd_ref
    
    # Extract phase difference from cross-spectrum: φ1 - φ2 = angle(csd_gain)
    # The cross-spectrum contains: CSD = G1 * G2* = |G1||G2| * e^(i*(φ1-φ2))
    # So angle(CSD/reference) gives us the phase difference between channels
    phase_diff = np.angle(csd_gain)
    
    # Convert power gains to voltage-equivalent gains with proper phase handling
    # Power gain = |S|², so voltage gain magnitude = sqrt(power_gain)
    # Protect against negative values due to numerical errors
    ch1_power_gain = np.maximum(ch1_power_gain, 1e-20)  # Ensure positive values
    ch2_power_gain = np.maximum(ch2_power_gain, 1e-20)  # Ensure positive values
    
    ch1_gain_mag = np.sqrt(ch1_power_gain)
    ch2_gain_mag = np.sqrt(ch2_power_gain)
    
    # Distribute phase difference between channels
    # Convention: assign half the phase to each channel with opposite signs
    # This preserves the relative phase difference: angle(G1) - angle(G2) = φ1 - φ2 = phase_diff
    # while making both gains complex (needed for S-parameter corrections)
    ch1_phase = phase_diff / 2
    ch2_phase = -phase_diff / 2
    
    # Create complex gains: G = |G| * e^(i*φ)
    ch1_gain_complex = ch1_gain_mag * np.exp(1j * ch1_phase)
    ch2_gain_complex = ch2_gain_mag * np.exp(1j * ch2_phase)

    # --------------------- S-parameter ---------------------
    try:
        ch1_cal_net = rf.Network(kwargs.get("ch1_file", CH1_FILE))
        ch2_cal_net = rf.Network(kwargs.get("ch2_file", CH2_FILE))
        # Interpolate to 1-2 GHz with 2001 steps
        n_points = DATASET_LENGTH
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
        # Apply S-parameter corrections 
        # Since we converted to voltage-equivalent gains, use voltage transfer function S21
        ch1_gain_complex /= ch1_s21  # Use complex S21 (voltage transfer)
        ch2_gain_complex /= ch2_s21  # Use complex S21 (voltage transfer)
        # For cross-spectrum (complex), use complex correction
        csd_gain /= (ch1_s21 * np.conj(ch2_s21))  # Correct cross-spectrum with complex S-parameters
        
        # Recalculate phase difference after S-parameter corrections
        # This ensures consistency: phase_diff = angle(G1) - angle(G2)
        phase_diff = np.angle(ch1_gain_complex) - np.angle(ch2_gain_complex)

    # -------------------- Save data to csv --------------------
    # Store complex values directly in CSV (following CW module pattern)
    results_df = pd.DataFrame({
        'Freq (GHz)': ND136_df_interpolated['Freq (GHz)'],
        'Ch1 Gain': ch1_gain_complex,  # Store as complex
        'Ch2 Gain': ch2_gain_complex,  # Store as complex  
        'Phase Difference (rad)': phase_diff,  # Use calculated phase difference
    })
    filepaths.append(os.path.join(ND_DIR, f"Processed_Noise_{date}.csv"))
    results_df.to_csv(filepaths[-1], index=False)

    # -------------------- Plotting Results --------------------
    # Convert to dB and degrees only for plotting/display
    ch1_gain_db = 2 * dB(np.abs(ch1_gain_complex))  # **CONVERSION TO REAL**: For plotting only
    ch2_gain_db = 2 * dB(np.abs(ch2_gain_complex))  # **CONVERSION TO REAL**: For plotting only
    
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot gain in dB (for display only)
    ax1.plot(ND136_df_interpolated['Freq (GHz)'], ch1_gain_db, label='Ch1 Average Gain', color=COLORS[4])
    ax1.plot(ND136_df_interpolated['Freq (GHz)'], ch2_gain_db, label='Ch2 Average Gain', color=COLORS[3])
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Gain (dB)")
    ax1.legend(loc="center left")

    # Plot phase difference (convert to degrees for display)
    phase_diff_deg = np.degrees(np.unwrap(phase_diff))  # **CONVERSION TO REAL**: For plotting only
    ax2.plot(ND136_df_interpolated['Freq (GHz)'], phase_diff_deg, label='Phase Difference', color=COLORS[2])
    ax2.set_ylabel("Phase Difference (degrees)")
    ax2.legend(loc="center right")

    plt.title("Average Gain and Phase Difference")
    plt.tight_layout()
    filepaths.append(os.path.join(ND_DIR, f"Noise_Gain_Phase_{date}.png"))
    plt.savefig(filepaths[-1], bbox_inches='tight')
    plt.close()

    return filepaths


if __name__ == "__main__":
    noise_main()