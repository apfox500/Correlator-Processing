import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf
from tqdm import tqdm

from Constants import *
from utils import dBm

def fft_to_dbm(complex_fft_result, n_samples, **kwargs):
  """Converts a complex FFT result array to a power spectrum in dBm.

  Parameters:
    complex_fft_result: The complex array from np.fft.rfft.
    n_samples: The total number of samples (N) used in the FFT.
    **kwargs: Additional keyword arguments for processing options.
      impedance: The system impedance in Ohms (default is IMPEDANCE).

  Returns:
    A numpy array of power values in dBm.
  """
  impedance = kwargs.get("impedance", IMPEDANCE)
  # get the magnitude from the complex fft result
  fft_magnitude = np.abs(complex_fft_result)

  # scale the fft magnitude to get peak voltage
  # multiply by 2 for a single-sided spectrum
  voltage_peak = (fft_magnitude / n_samples) * 2

  # convert peak voltage to rms
  voltage_rms = voltage_peak / np.sqrt(2)

  # calculate power in watts
  power_watts = (voltage_rms**2) / impedance
  
  # avoid log(0) errors for noise floor
  power_watts[power_watts < 1e-20] = 1e-20

  # convert power to dBm
  return dBm(power_watts)

def process_CWdata(ch1_data, ch2_data, idx, freq, power_df, **kwargs):
    """ Process time-domain data from two channels for a specific CW tone at a given frequency and produces the gain, power, and phase information and updates the power dataframe.
    Parameters:
        ch1_data: The data from channel 1.
        ch2_data: The data from channel 2.
        idx: The index of the current frequency in the power dataframe.
        freq: The frequency in GHz for which the data is being processed.
        power_df: The dataframe containing power measurements.
    **kwargs: Additional keyword arguments for processing options.
    fft_freq: The frequency bins for the FFT (default is np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/10e9)).
    freq_graph: If True, plots the frequency response graph for the current frequency.
    ch1s: S-parameter data for channel 1 (optional).
    ch2s: S-parameter data for channel 2 (optional).

    
    Returns:
        The updated power dataframe with gain and phase information.
        """

    fft_freq = kwargs.get("fft_freq", np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9)))
    freq_graph = kwargs.get("freq_graph", False)
    ch1s = kwargs.get("ch1s", None)
    ch2s = kwargs.get("ch2s", None)
    if (ch1s is not None) ^ (ch2s is not None):
        raise ValueError("Both 'ch1s' and 'ch2s' must be provided together, or neither.")

    
    # -------------------- Magnitude Calculation --------------------
    # perform the conversion to voltage
    ch1_voltage = ch1_data * VOLT_PER_TICK
    ch2_voltage = ch2_data * VOLT_PER_TICK

    # Now, perform the FFT on the VOLTAGE signal
    ch1_fft = np.fft.rfft(ch1_voltage)
    ch2_fft = np.fft.rfft(ch2_voltage)

    # Compensate with S-parameters if provided
    if ch1s is not None and ch2s is not None:
        # Divide FFT by S21(complex)
        ch1_fft = ch1_fft / ch1s[:, 1, 0]  # S21 for channel 1
        ch2_fft = ch2_fft / ch2s[:, 1, 0]  # S21 for channel 2

    # convert from data to dbM
    ch1_dbm = fft_to_dbm(ch1_fft, SAMPLE_CUTOFF)
    ch2_dbm = fft_to_dbm(ch2_fft, SAMPLE_CUTOFF)

    # Find the index closest to the desired frequency (in Hz)
    target_freq_hz = freq * 1e9  # Convert GHz to Hz
    freq_idx_ch1 = np.argmin(np.abs(fft_freq - target_freq_hz))
    freq_idx_ch2 = np.argmin(np.abs(fft_freq - target_freq_hz))  # If ch2_freq is different, use ch2_freq here

    ch1_fft_val = ch1_dbm[freq_idx_ch1]
    ch2_fft_val = ch2_dbm[freq_idx_ch2]

    # Add the freq + val pair to dataframe
    power_df.at[idx, "ch1_real_freq"] = fft_freq[freq_idx_ch1]
    power_df.at[idx, "ch1_fft_val"] = ch1_fft_val
    power_df.at[idx, "ch2_real_freq"] = fft_freq[freq_idx_ch2]
    power_df.at[idx, "ch2_fft_val"] = ch2_fft_val
    
    # ---------------------- Phase Calculation --------------------
    cross = ch1_fft[freq_idx_ch1] * np.conjugate(ch2_fft[freq_idx_ch2])
    phase_diff = np.angle(cross)

    # Add the phases + difference to the dataframe
    power_df.at[idx, "ch1_phase"] = np.angle(ch1_fft[freq_idx_ch1])
    power_df.at[idx, "ch2_phase"] = np.angle(ch2_fft[freq_idx_ch2])
    power_df.at[idx, "phase_diff"] = phase_diff

    # ----------------------- Gain Calculation -----------------------
    # find the gain for ch1 and ch2
    power_df.at[idx, "ch1_gain"] = ch1_fft_val - power_df.at[idx, "Power(dBm)"]
    power_df.at[idx, "ch2_gain"] = ch2_fft_val - power_df.at[idx, "Power(dBm)"]

    # Graph frequency response if requested
    if freq_graph:
        plt.figure(figsize=(10, 5))
        plt.plot(fft_freq, ch1_dbm, '-', label='Ch1 FFT', color=COLORS[0], linewidth=1)
        plt.plot(fft_freq, ch2_dbm, '-', label='Ch2 FFT', color=COLORS[1], linewidth=1)
        # Highlight the selected frequency points
        plt.scatter([fft_freq[freq_idx_ch1]], [np.abs(ch1_fft_val)], color=COLORS[0], s=100, label='Ch1 Selected Freq', zorder=5)
        plt.scatter([fft_freq[freq_idx_ch2]], [np.abs(ch2_fft_val)], color=COLORS[1], s=100, label='Ch2 Selected Freq', zorder=5)
        plt.title(f"FFT of Channels at {freq} GHz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return power_df

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
            - Processes the data to extract FFT values and other metrics.
            - Optionally graphs data for selected indices.
        5. Calculates channel gains and applies corrections if S-parameters are available.
        6. Prints summary statistics for channel gains.
        7. Saves the processed DataFrame to a new CSV file.
        8. Generates and saves plots for power, gain, and phase difference versus frequency.
    
    Outputs:
        - Processed CSV file with calculated gains and corrections.
        - Plots of calculated power, gain, and phase difference saved to the output directory.
    """    

    filepaths = []

    # Find the first CSV file ending with _{date}.csv
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(f"_{date}.csv")]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR} ending with _{date}.csv")
    
    csv_name = csv_files[0]  # Use the first file found
    print(f"Using CSV file: {csv_name}")
    
    # read in power csv
    csv_path = os.path.join(DATA_DIR, csv_name)
    power_df = pd.read_csv(csv_path)

    # --------------------- S-parameter ---------------------
    filekwargs = kwargs.get("file_kwargs", {})
    # Load S parameters for alpha/beta
    power_df, ab_success = alphabeta_correction(power_df, **filekwargs)

    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    filename = kwargs.get("filename", CW_FILENAME)
    graph_flag = kwargs.get("graph_flag", 0) # 0 means none, 1 means first and last, 2 means all
    # Determine which indices to graph based on graph_flag
    if graph_flag == 2:
        graph_indices = set(range(len(power_df)))
    elif graph_flag == 1:
        graph_indices = {0, len(power_df) - 1}
    else:
        graph_indices = set()


    #  -------------------- Process Each Frequency --------------------
    for idx, row in tqdm(power_df.iterrows(), total=len(power_df), unit="frequencies"):
        
        freq = float(row["Frequency"])
        # -------------------- Read in the file --------------------
        curr_filename = filename.format(freq=freq, FS=FS, date=date)
        # read in data for specifc frequency
        if not os.path.exists(os.path.join(DATA_DIR, curr_filename)):
            print(f"File {curr_filename} does not exist in {DATA_DIR}. Skipping...")
            continue
        data_array = np.load(os.path.join(DATA_DIR, curr_filename))

        # process the data
        power_df = process_CWdata(
            ch1_data=data_array[0, :SAMPLE_CUTOFF],
            ch2_data=data_array[1, :SAMPLE_CUTOFF],
            idx=idx,
            freq=freq,
            power_df=power_df,
            freq_graph= idx in graph_indices,
            fft_freq=fft_freq
        )

    # -------------------- Calculate Gain --------------------
    # Calculate the gain for ch1 and ch2
    power_df["ch1_gain"] = power_df["ch1_fft_val"] - power_df["Power(dBm)"]
    power_df["ch2_gain"] = power_df["ch2_fft_val"] - power_df["Power(dBm)"]

    if ab_success: # if alpha/beta correction was successful, add the corrections
        alpha_mag = np.abs(power_df["alpha"].values.astype(complex))
        beta_mag = np.abs(power_df["beta"].values.astype(complex))
        gamma_pm_mag = np.abs(power_df["GPM"].values.astype(complex))

        ch1_correction_db = 10 * np.log10(1 - gamma_pm_mag**2) - 20 * np.log10(alpha_mag)
        ch2_correction_db = 10 * np.log10(1 - gamma_pm_mag**2) - 20 * np.log10(beta_mag)

        power_df["ch1_gain"] += ch1_correction_db
        power_df["ch2_gain"] += ch2_correction_db

    # Print average gain
    print(f"Ch1 Gain: mean = {power_df['ch1_gain'].mean():.2f} dB, median = {power_df['ch1_gain'].median():.2f} dB")
    print(f"Ch2 Gain: mean = {power_df['ch2_gain'].mean():.2f} dB, median = {power_df['ch2_gain'].median():.2f} dB")

    # Save the updated dataframe to a new CSV file
    filepaths.append(os.path.join(OUT_DIR, f"CW_Processed_{csv_name}"))
    power_df.to_csv(filepaths[-1], index=False)

    # ---------------------- Graphing All --------------------

    # Graph the freq values of ch1 and ch2
    plt.figure(figsize=(10, 5))
    plt.plot(power_df["ch1_real_freq"] / 1e9, power_df["ch1_fft_val"], '-o', label='Ch1 Power', color=COLORS[0], alpha=.5)
    plt.plot(power_df["ch2_real_freq"] / 1e9, power_df["ch2_fft_val"], '-o', label='Ch2 Power', color=COLORS[1], alpha=.5)
    
    # Plot real power
    plt.plot(power_df["Frequency"], power_df["Power(dBm)"], '-', label='Real Power', color='red')
    # Plot nominal power (dashed line, 50% transparent, same color as power)
    plt.plot(power_df["Frequency"], power_df["power_norminal"], '--', label='Nominal Power', color='red', alpha=0.5)

    # Plot gains
    # plt.plot(power_df["ch1_real_freq"] / 1e9, power_df["ch1_gain"], '-', label='Ch1 Gain', color=COLORS[0])
    # plt.plot(power_df["ch2_real_freq"] / 1e9, power_df["ch2_gain"], '-', label='Ch2 Gain', color=COLORS[1])

    plt.title("Calculated Power from CW measurement")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power (dBm)")
    plt.legend()
    plt.tight_layout()
    filepaths.append(os.path.join(OUT_DIR, f"CW_Power_{date}.png"))
    plt.savefig(filepaths[-1], bbox_inches='tight')
    plt.show()

    # ---------------------- Graph Gain and Phase Difference --------------------
    # should be the final, cleaner graph
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot gain
    ax1.plot(power_df["ch1_real_freq"] / 1e9, power_df["ch1_gain"], '-', color=COLORS[0], label='Ch1 Gain')
    ax1.plot(power_df["ch2_real_freq"] / 1e9, power_df["ch2_gain"], '-', color=COLORS[1], label='Ch2 Gain')
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Gain (dB)")
    ax1.legend(loc="lower left")

    # Plot phase difference (convert to degrees)
    phase_diff_deg = np.degrees(np.unwrap(power_df["phase_diff"].apply(lambda x: x if np.isscalar(x) else x[0])))
    ax2.plot(power_df["ch1_real_freq"] / 1e9, phase_diff_deg, '-', color=COLORS[3], label='Phase Difference')
    ax2.set_ylabel("Phase Difference (degrees)")
    ax2.legend(loc="upper right")

    plt.title("Gain and Phase Difference vs Frequency")
    plt.tight_layout()
    filepaths.append(os.path.join(OUT_DIR, f"CW_Gain_Phase_{date}.png"))
    plt.savefig(filepaths[-1], bbox_inches='tight')
    plt.show()

    # ---------------------- Create Complex Gain CSV ----------------------
     # Convert gain from dB to linear magnitude
    s31_linear_mag = 10**(power_df['ch1_gain'] / 20)
    s46_linear_mag = 10**(power_df['ch2_gain'] / 20)

    # Reconstruct the complex S-parameters using magnitude and phase
    s31_complex = s31_linear_mag * np.exp(1j * power_df['ch1_phase'])
    s46_complex = s46_linear_mag * np.exp(1j * power_df['ch2_phase'])

    # Create a new dataframe for the complex gain data
    complex_gain_df = pd.DataFrame({
        'Frequency': power_df['Frequency'], # Frequency in GHz
        'S31': s31_complex,
        'S46': s46_complex
    })

    filepaths.append(os.path.join(OUT_DIR, f"CW_Complex_Gain_{date}.csv"))
    complex_gain_df.to_csv(filepaths[-1], index=False)

    return filepaths
