import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf
from tqdm import tqdm

from Constants import *
from utils import dBm

def process_CWdata(ch1_data, ch2_data, input_power_dbm, freq, **kwargs):
    """Process time-domain data to calculate raw complex S-parameters.

    This function takes time-domain voltage data for two channels and calculates
    the raw, uncorrected complex S-parameters (e.g., S31, S46) based on the
    known input power.

    Parameters:
        ch1_data: Time-domain data for channel 1.
        ch2_data: Time-domain data for channel 2.
        input_power_dbm: The input power in dBm.
        freq: The target frequency in GHz.
        **kwargs: Additional keyword arguments.
            fft_freq: Frequency bins for the FFT.
            ch1s, ch2s: S-parameter compensation networks (optional).
            impedance: System impedance (default R_0).
            n_samples: Number of FFT samples (default SAMPLE_CUTOFF).

    Returns:
        A tuple containing:
        - s31_raw (complex): The raw complex S-parameter for channel 1.
        - s46_raw (complex): The raw complex S-parameter for channel 2.
        - ch1_real_freq (float): The actual frequency bin for channel 1.
        - ch2_real_freq (float): The actual frequency bin for channel 2.
        - ch1_fft (np.ndarray): The full complex FFT spectrum for channel 1.
        - ch2_fft (np.ndarray): The full complex FFT spectrum for channel 2.
    """
    # get kwargs
    fft_freq = kwargs.get("fft_freq", np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9)))
    n_samples = kwargs.get("n_samples", SAMPLE_CUTOFF)
    impedance = kwargs.get("impedance", R_0)
    ch1s = kwargs.get("ch1s", None)
    ch2s = kwargs.get("ch2s", None)

    # convert ADC ticks to voltage
    ch1_voltage = ch1_data * VOLT_PER_TICK
    ch2_voltage = ch2_data * VOLT_PER_TICK

    # perform FFT
    ch1_fft = np.fft.rfft(ch1_voltage)
    ch2_fft = np.fft.rfft(ch2_voltage)

    # compensate with S-parameters if provided
    if ch1s is not None and ch2s is not None:
        ch1_fft /= ch1s[:, 1, 0]  # S21 for channel 1
        ch2_fft /= ch2s[:, 1, 0]  # S21 for channel 2

    # find the index closest to the target frequency
    target_freq_hz = freq * 1e9
    freq_idx_ch1 = np.argmin(np.abs(fft_freq - target_freq_hz))
    freq_idx_ch2 = np.argmin(np.abs(fft_freq - target_freq_hz))

    ch1_fft_peak = ch1_fft[freq_idx_ch1]
    ch2_fft_peak = ch2_fft[freq_idx_ch2]

    # --- Complex S-Parameter Calculation ---
    # calculate the incident wave 'a' from input power
    power_watts = 10**((input_power_dbm - 30) / 10)
    a_in = np.sqrt(power_watts) # assume phase is 0

    # calculate the outgoing wave 'b' from the measured FFT voltage
    # b = V_rms / sqrt(Z0), where V_rms = V_peak / sqrt(2)
    # V_peak = (FFT_magnitude / N) * 2 (for single-sided spectrum)
    def calculate_b_out(fft_peak_value):
        v_peak_complex = (fft_peak_value / n_samples) * 2
        v_rms_complex = v_peak_complex / np.sqrt(2)
        return v_rms_complex / np.sqrt(impedance)

    b_out_ch1 = calculate_b_out(ch1_fft_peak)
    b_out_ch2 = calculate_b_out(ch2_fft_peak)

    # S-parameter is the ratio of the outgoing to incoming wave
    s31_raw = b_out_ch1 / a_in
    s46_raw = b_out_ch2 / a_in
    
    ch1_real_freq = fft_freq[freq_idx_ch1]
    ch2_real_freq = fft_freq[freq_idx_ch2]

    return s31_raw, s46_raw, ch1_real_freq, ch2_real_freq, ch1_fft, ch2_fft

def alphabeta_correction(power_df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, bool]:
    """
    Load S-parameter files and calculate alpha and beta corrections for the power dataframe.
    
    Parameters:
        power_df: The power dataframe containing frequency and power measurements.
        **kwargs: Additional keyword arguments for file paths and processing options.
            n_points: Number of points for frequency axis (default is 2001).
            file1, file2, file3, file4, file5, file6: Paths to S-parameter files.

    Returns:
        A tuple containing the updated power dataframe and a boolean indicating success.
    """
    try:
        # define a common frequency axis for all S-parameters
        n_points = kwargs.get("n_points", 2001)
        freq_axis_ghz = np.linspace(1, 2, n_points, dtype=float)
        common_freq = rf.Frequency.from_f(freq_axis_ghz, unit='GHz')

        # 1 - S^SP_1A & S^SP_11
        file1 = kwargs.get("file1", FILE1)
        A1 = rf.Network(file1).interpolate(common_freq)
        SSP1A = pd.Series(A1.s[:, 1, 0], index=freq_axis_ghz)
        SSP11 = pd.Series(A1.s[:, 1, 1], index=freq_axis_ghz)

        # 2 - S^SP_BB & S^SP_BA
        file2 = kwargs.get("file2", FILE2)
        ab_df = pd.read_csv(file2)
        sspbb_real = np.interp(freq_axis_ghz, ab_df['Freq'], ab_df["S22_r"])
        sspbb_imag = np.interp(freq_axis_ghz, ab_df['Freq'], ab_df["S22_i"])
        SSPBB = pd.Series(sspbb_real + 1j * sspbb_imag, index=freq_axis_ghz)

        sspba_real = np.interp(freq_axis_ghz, ab_df['Freq'], ab_df["S21_r"])
        sspba_imag = np.interp(freq_axis_ghz, ab_df['Freq'], ab_df["S21_i"])
        SSPBA = pd.Series(sspba_real + 1j * sspba_imag, index=freq_axis_ghz)

        # 3 - G^PM
        file3 = kwargs.get("file3", FILE3)
        gpm_df = pd.read_csv(file3)
        GPM = pd.Series(
            np.interp(freq_axis_ghz, gpm_df['Freq'] / 1e9, gpm_df["S11_r"] + 1j * gpm_df["S11_i"]), 
            index=freq_axis_ghz
        )

        # 4 - G^NCin
        file4 = kwargs.get("file4", FILE4)
        GNCin = pd.Series(
            rf.Network(file4).interpolate(common_freq).s[:, 0, 0],  # S11
            index=freq_axis_ghz
        )

        # 5 - S^SP_66 & S^SP_6A
        file5 = kwargs.get("file5", FILE5)
        A6 = rf.Network(file5).interpolate(common_freq)
        SSP66 = pd.Series(A6.s[:, 1, 1], index=freq_axis_ghz)  # S22
        SSP6A = pd.Series(A6.s[:, 1, 0], index=freq_axis_ghz)  # S21

        # 6 - G^NCout
        file6 = kwargs.get("file6", FILE6)
        GNCout = pd.Series(
            rf.Network(file6).interpolate(common_freq).s[:, 0, 0], # S11
            index=freq_axis_ghz
        )  

        # Load inot one dataframe
        sparam_df = pd.DataFrame({
            'SSP1A': SSP1A, 'SSP11': SSP11,
            'SSPBB': SSPBB, 'SSPBA': SSPBA,
            'GPM': GPM, 'GNCin': GNCin,
            'SSP66': SSP66, 'SSP6A': SSP6A,
            'GNCout': GNCout
        })

        alpha_num = sparam_df['SSP1A'] * (1 - sparam_df['SSPBB'] * sparam_df['GPM'])
        alpha_den = sparam_df['SSPBA'] * (1 - sparam_df['SSP11'] * sparam_df['GNCin'])
        sparam_df['alpha'] = alpha_num / alpha_den

        beta_num = sparam_df['SSP6A'] * (1 - sparam_df['SSPBB'] * sparam_df['GPM'])
        beta_den = sparam_df['SSPBA'] * (1 - sparam_df['SSP66'] * sparam_df['GNCout'])
        sparam_df['beta'] = beta_num / beta_den
        
        # Sort and merge alpha and beta with power dataframe

        power_df = power_df.sort_values('Frequency').reset_index(drop=True)
        sparam_df = sparam_df.sort_index().reset_index().rename(columns={'index': 'Frequency'})
        # Only merge alpha and beta columns from sparam_df into power_df
        power_df = pd.merge_asof(
            left=power_df,
            right=sparam_df[['Frequency', 'alpha', 'beta', 'GPM']],
            on='Frequency',
            direction='nearest'  # finds the closest s-param frequency
        )

        return power_df, True

    except FileNotFoundError as e:
        print(f"Alpha/Beta S-parameter files not found: {e}. Skipping Alpha/Beta correction.")
        return power_df, False


def CW_main(date:str=CW_DATE, **kwargs) -> list[str]:
    """
    Main function to process continuous wave (CW) measurement data, calculate gain, apply S-parameter corrections,
    and generate summary plots.
    
    Parameters:
        date (str): The date string used to identify the relevant CSV and data files. Defaults to CW_DATE.
        **kwargs (dict): Additional keyword arguments:
            - filename (str): Template for the data file names. Defaults to CW_FILENAME.
            - graph_flag (int): Controls which frequency indices to graph.
                0 = none, 1 = first and last, 2 = all. Defaults to 0.
            - file_kwargs (dict): Additional keyword arguments passed to the alphabeta_correction function.
    
    Returns:
        list[str]: A list of file paths for all generated files.

    Raises:
        FileNotFoundError: If no CSV files matching the date pattern are found in the data directory.
    
    Process Overview:
        1. Locates the first CSV file in the data directory matching the specified date.
        2. Loads the power measurement data from the CSV file.
        3. Applies S-parameter (alpha/beta) corrections if available.
        4. Iterates over each frequency row:
            - Loads the corresponding data file.
            - Processes the data to extract raw complex S-parameters.
            - Optionally graphs FFT spectrum for selected indices.
        5. Applies alpha/beta corrections using complex arithmetic.
        6. Calculates final channel gains (dB) and phase from the corrected complex S-parameters.
        7. Prints summary statistics for channel gains.
        8. Saves the processed DataFrame to a new CSV file.
        9. Generates and saves plots for power, gain, and phase difference versus frequency.
        10. Saves the corrected complex S-parameters to a CSV file.
    """    
    filepaths = []

    # find the first CSV file ending with _{date}.csv
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(f"_{date}.csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR} ending with _{date}.csv")
    
    csv_name = csv_files[0]
    print(f"Using CSV file: {csv_name}")
    
    # read in power csv
    csv_path = os.path.join(DATA_DIR, csv_name)
    power_df = pd.read_csv(csv_path)

    # load S-parameters for alpha/beta correction
    filekwargs = kwargs.get("file_kwargs", {})
    power_df, ab_success = alphabeta_correction(power_df, **filekwargs)

    # initialize lists to store results
    s31_raw_list, s46_raw_list = [], []
    ch1_real_freq_list, ch2_real_freq_list = [], []

    filename = kwargs.get("filename", CW_FILENAME)
    graph_flag = kwargs.get("graph_flag", 0)
    graph_indices = {0, len(power_df) - 1} if graph_flag == 1 else set(range(len(power_df))) if graph_flag == 2 else set()
    
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # process each frequency
    for idx, row in tqdm(power_df.iterrows(), total=len(power_df), unit="frequencies"):
        freq = float(row["Frequency"])
        curr_filename = filename.format(freq=freq, FS=FS, date=date)

        if not os.path.exists(os.path.join(DATA_DIR, curr_filename)):
            print(f"File {curr_filename} does not exist in {DATA_DIR}. Skipping...")
            # append placeholder values to maintain index alignment
            s31_raw_list.append(np.nan)
            s46_raw_list.append(np.nan)
            ch1_real_freq_list.append(np.nan)
            ch2_real_freq_list.append(np.nan)
            continue
        
        data_array = np.load(os.path.join(DATA_DIR, curr_filename))

        s31_raw, s46_raw, f1, f2, ch1_fft, ch2_fft = process_CWdata(
            ch1_data=data_array[0, :SAMPLE_CUTOFF],
            ch2_data=data_array[1, :SAMPLE_CUTOFF],
            input_power_dbm=row["Power(dBm)"],
            freq=freq,
            fft_freq=fft_freq
        )
        s31_raw_list.append(s31_raw)
        s46_raw_list.append(s46_raw)
        ch1_real_freq_list.append(f1)
        ch2_real_freq_list.append(f2)

        # graph frequency response if requested
        if idx in graph_indices:
            # this logic is just for the plot, replicating old fft_to_dbm
            def to_dbm(fft_array, n_samples, impedance):
                v_peak = (np.abs(fft_array) / n_samples) * 2
                v_rms = v_peak / np.sqrt(2)
                power_watts = (v_rms**2) / impedance
                power_watts[power_watts < 1e-20] = 1e-20 # avoid log(0)
                return 10 * np.log10(power_watts) + 30

            ch1_dbm = to_dbm(ch1_fft, SAMPLE_CUTOFF, R_0)
            ch2_dbm = to_dbm(ch2_fft, SAMPLE_CUTOFF, R_0)
            
            plt.figure(figsize=(10, 5))
            plt.plot(fft_freq, ch1_dbm, '-', label='Ch1 FFT', color=COLORS[0], linewidth=1)
            plt.plot(fft_freq, ch2_dbm, '-', label='Ch2 FFT', color=COLORS[1], linewidth=1)
            plt.scatter([f1], [ch1_dbm[np.argmin(np.abs(fft_freq - f1))]], color=COLORS[0], s=100, label='Ch1 Selected Freq', zorder=5)
            plt.scatter([f2], [ch2_dbm[np.argmin(np.abs(fft_freq - f2))]], color=COLORS[1], s=100, label='Ch2 Selected Freq', zorder=5)
            plt.title(f"FFT of Channels at {freq} GHz")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power (dBm)")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # add raw results to dataframe
    power_df["s31_raw"] = s31_raw_list
    power_df["s46_raw"] = s46_raw_list
    power_df["ch1_real_freq"] = ch1_real_freq_list
    power_df["ch2_real_freq"] = ch2_real_freq_list

    # apply corrections if successful
    if ab_success:
        # ensure data types are correct for complex math
        power_df['alpha'] = power_df['alpha'].astype(complex)
        power_df['beta'] = power_df['beta'].astype(complex)
        power_df['GPM'] = power_df['GPM'].astype(complex)

        # mismatch correction factor (scalar)
        mismatch_factor = np.sqrt(1 - np.abs(power_df['GPM'])**2)

        # apply correction using complex division
        power_df["s31_corrected"] = (power_df["s31_raw"] / power_df["alpha"]) * mismatch_factor
        power_df["s46_corrected"] = (power_df["s46_raw"] / power_df["beta"]) * mismatch_factor
    else:
        # if no correction, the corrected value is just the raw value
        power_df["s31_corrected"] = power_df["s31_raw"]
        power_df["s46_corrected"] = power_df["s46_raw"]

    # calculate final gain and phase from the corrected complex S-parameters
    power_df["ch1_gain"] = 20 * np.log10(np.abs(power_df["s31_corrected"]))
    power_df["ch2_gain"] = 20 * np.log10(np.abs(power_df["s46_corrected"]))
    power_df["ch1_phase"] = np.angle(power_df["s31_corrected"])
    power_df["ch2_phase"] = np.angle(power_df["s46_corrected"])
    
    # calculate phase difference from corrected values
    cross_corrected = power_df["s31_corrected"] * np.conjugate(power_df["s46_corrected"])
    power_df["phase_diff"] = np.angle(cross_corrected)

    # print average gain
    print(f"Ch1 Gain: mean = {power_df['ch1_gain'].mean():.2f} dB, median = {power_df['ch1_gain'].median():.2f} dB")
    print(f"Ch2 Gain: mean = {power_df['ch2_gain'].mean():.2f} dB, median = {power_df['ch2_gain'].median():.2f} dB")

    # save the updated dataframe
    filepaths.append(os.path.join(OUT_DIR, f"CW_Processed_{csv_name}"))
    power_df.to_csv(filepaths[-1], index=False)
    
    # ---------------------- Graphing All --------------------

    # calculate final output power for plotting
    power_df['ch1_power_out_dbm'] = power_df['ch1_gain'] + power_df['Power(dBm)']
    power_df['ch2_power_out_dbm'] = power_df['ch2_gain'] + power_df['Power(dBm)']

    plt.figure(figsize=(10, 5))
    plt.plot(power_df["ch1_real_freq"] / 1e9, power_df["ch1_power_out_dbm"], '-o', label='Ch1 Power Out (Corrected)', color=COLORS[0], alpha=.7)
    plt.plot(power_df["ch2_real_freq"] / 1e9, power_df["ch2_power_out_dbm"], '-o', label='Ch2 Power Out (Corrected)', color=COLORS[1], alpha=.7)
    
    plt.plot(power_df["Frequency"], power_df["Power(dBm)"], '-', label='Power In', color='red')
    if "power_norminal" in power_df.columns:
        plt.plot(power_df["Frequency"], power_df["power_norminal"], '--', label='Nominal Power In', color='red', alpha=0.5)

    plt.title("Corrected Output Power from CW Measurement")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power (dBm)")
    plt.legend()
    plt.tight_layout()
    filepaths.append(os.path.join(OUT_DIR, f"CW_Power_{date}.png"))
    plt.savefig(filepaths[-1], bbox_inches='tight')
    plt.show()

    # ---------------------- Graph Gain and Phase Difference --------------------
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # plot gain
    ax1.plot(power_df["ch1_real_freq"] / 1e9, power_df["ch1_gain"], '-', color=COLORS[0], label='Ch1 Gain')
    ax1.plot(power_df["ch2_real_freq"] / 1e9, power_df["ch2_gain"], '-', color=COLORS[1], label='Ch2 Gain')
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Gain (dB)")
    ax1.legend(loc="lower left")

    # plot phase difference (convert to degrees)
    phase_diff_deg = np.degrees(np.unwrap(power_df["phase_diff"].dropna()))
    ax2.plot(power_df["ch1_real_freq"].dropna() / 1e9, phase_diff_deg, '-', color=COLORS[3], label='Phase Difference')
    ax2.set_ylabel("Phase Difference (degrees)")
    ax2.legend(loc="upper right")

    plt.title("Gain and Phase Difference vs Frequency")
    plt.tight_layout()
    filepaths.append(os.path.join(OUT_DIR, f"CW_Gain_Phase_{date}.png"))
    plt.savefig(filepaths[-1], bbox_inches='tight')
    plt.show()

    # ---------------------- Create Complex Gain CSV ----------------------
    complex_gain_df = pd.DataFrame({
        'Frequency': power_df['Frequency'], # Frequency in GHz
        'S31': power_df['s31_corrected'],
        'S46': power_df['s46_corrected'],
    })

    filepaths.append(os.path.join(OUT_DIR, f"CW_Complex_Gain_{date}.csv"))
    complex_gain_df.to_csv(filepaths[-1], index=False)

    return filepaths
