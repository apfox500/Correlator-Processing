import os

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Annotated
from typing import Union
from numpy.typing import NDArray

from Constants import *


def process_noise_data(data_file_base:str, num_samples_test:int=NUM_TRACES, **kwargs) -> tuple[
    Annotated[NDArray[np.complex128], (DATASET_LENGTH,)],
    Annotated[NDArray[np.complex128], (DATASET_LENGTH,)],
    Annotated[NDArray[np.complex128], (DATASET_LENGTH,)]
]:
    """
    Process all noise data files and return average PSD for ch1 and ch2.
    This function processes data in chunks of SAMPLE_CUTOFF to handle large files.

    Parameters:
        data_file_base: The base string for the noise data filepaths (without index and .npy).
        num_samples_test: Number of samples/files to process.
        **kwargs: Additional keyword arguments for processing options.

    Returns:
        ch1_avg_psd, ch2_avg_psd: Average PSD arrays for ch1 and ch2 in V^2/Hz.
        csd_avg: Average Cross-Spectral Density (CSD) array.
    """
    # -------------------- Initialization --------------------
    fft_freq = kwargs.get("fft_freq", np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9)))
    power_data = np.zeros((3, DATASET_LENGTH), dtype=np.complex128)
    total_chunks_processed = 0
    

    # -------------------- Process Noise Files and Chunks --------------------
    # loop over the specified number of files
    for idx in tqdm(range(num_samples_test), desc="Processing Noise Files", unit="file", total=num_samples_test):
        filename = f"{data_file_base}_{idx}.npy"
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File {filename} does not exist in {DATA_DIR}. Skipping...")
            continue
        
        data_array = np.load(filepath)
        
        # determine the number of full chunks in the data file
        num_chunks = data_array.shape[1] // SAMPLE_CUTOFF
        if num_chunks == 0:
            print(f"Warning: {filename} is smaller than a single chunk ({SAMPLE_CUTOFF} samples) and will be skipped.")
            continue

        # process the file in chunks
        for i in range(num_chunks):
            start_idx = i * SAMPLE_CUTOFF
            end_idx = start_idx + SAMPLE_CUTOFF
            data_chunk = data_array[:, start_idx:end_idx]
            
            # process a single chunk of data (no change needed to this function)
            ch1_psd, ch2_psd, csd = process_single_noise_data(data_chunk, fft_freq)

            # accumulate power data
            power_data[0, :] += ch1_psd
            power_data[1, :] += ch2_psd
            power_data[2, :] += csd
        
        total_chunks_processed += num_chunks

    # -------------------- Calculate Average PSD --------------------
    if total_chunks_processed == 0:
        print("Warning: No data chunks were processed. Returning zero arrays.")
        # handle case where no data was processed to avoid division by zero
        return (np.zeros(DATASET_LENGTH, dtype=np.complex128),
                np.zeros(DATASET_LENGTH, dtype=np.complex128),
                np.zeros(DATASET_LENGTH, dtype=np.complex128))

    # average by the total number of chunks processed
    ch1_avg_psd = power_data[0, :] / total_chunks_processed
    ch2_avg_psd = power_data[1, :] / total_chunks_processed
    csd_avg = power_data[2, :] / total_chunks_processed

    return ch1_avg_psd, ch2_avg_psd, csd_avg

def process_single_noise_data(data_array: NDArray, fft_freq: NDArray, **kwargs) -> tuple[NDArray, NDArray, NDArray]:
    """
    Processes noise data from two channels, computes their FFT, cross-spectral density (CSD), and power spectral density (PSD).

    Parameters:
        data_array (np.ndarray): 2D array of raw noise data with shape (2, N), where each row corresponds to a channel.
        fft_freq (np.ndarray): 1D array of frequency bins corresponding to the FFT output.
        **kwargs:
            graph_fft (bool, optional): If True, plots the FFT magnitude of both channels. Default is False.
            graph_psd (bool, optional): If True, plots the PSD of both channels. Default is False.

    Returns:
        tuple:
            ch1_psd (np.ndarray): Power spectral density of channel 1 (trimmed to 1-2 GHz).
            ch2_psd (np.ndarray): Power spectral density of channel 2 (trimmed to 1-2 GHz).
            csd (np.ndarray): Cross-spectral density (complex) between channel 1 and channel 2 (trimmed to 1-2 GHz).

    Notes:
        - Only the frequency range 1-2 GHz is considered for PSD and CSD calculations.
        - Plots are displayed if the corresponding keyword arguments are set to True.
    """

    graph_fft = kwargs.get("graph_fft", False)
    graph_psd = kwargs.get("graph_psd", False)

    # -------------------- Extract and Convert Data --------------------
    ch1_noise = data_array[0, :SAMPLE_CUTOFF]
    ch2_noise = data_array[1, :SAMPLE_CUTOFF]
    ch1_voltage = ch1_noise * VOLT_PER_TICK
    ch2_voltage = ch2_noise * VOLT_PER_TICK

    # -------------------- Apply Window Function --------------------
    # Apply Blackman window to reduce spectral leakage
    window = np.blackman(SAMPLE_CUTOFF)
    ch1_windowed = ch1_voltage * window
    ch2_windowed = ch2_voltage * window
    
    # Calculate window power for PSD normalization
    window_power = np.sum(window**2)

    # -------------------- FFT and PSD Calculation --------------------
    ch1_fft = np.fft.rfft(ch1_windowed)
    ch2_fft = np.fft.rfft(ch2_windowed)

    if graph_fft:
        plt.figure(figsize=(10, 5))
        plt.plot(fft_freq, np.abs(ch1_fft), label='Ch1 FFT', color=COLORS[0])
        plt.plot(fft_freq, np.abs(ch2_fft), label='Ch2 FFT', color=COLORS[1])
        plt.title("FFT of Noise Data")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Trim FFT to 1-2 GHz
    mask = (fft_freq >= 1e9) & (fft_freq <= 2e9)
    ch1_fft = ch1_fft[mask]
    ch2_fft = ch2_fft[mask]
    
    # Calculate PSD with window correction
    ch1_psd = 2 * np.abs(ch1_fft) ** 2 / (window_power * FS * 1e9)
    ch2_psd = 2 * np.abs(ch2_fft) ** 2 / (window_power * FS * 1e9)

    # Calculate Cross-Spectral Density (CSD) with window correction
    csd = 2 * (ch1_fft * np.conjugate(ch2_fft)) / (window_power * FS * 1e9)

    if graph_psd:
        plt.figure(figsize=(10, 5))
        plt.plot(fft_freq[mask] / 1e9, dB(ch1_psd), label='Ch1 PSD', color=COLORS[0])
        plt.plot(fft_freq[mask] / 1e9, dB(ch2_psd), label='Ch2 PSD', color=COLORS[1])
        plt.title("Power Spectral Density of Noise Data")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("PSD (dBm/Hz)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ch1_psd, ch2_psd, csd

def dB(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert a linear value or array to decibels (dB).
    Parameters:
        value (float or np.ndarray): Linear value(s) to convert. Must be > 0 for valid dB result.
    Returns:
        float or np.ndarray: Value(s) in dB. Returns -np.inf for elements <= 0.
    """
    return 10 * np.log10(value) if np.all(value > 0) else np.where(value > 0, 10 * np.log10(value), -np.inf)

def db(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Alias for dB function to maintain compatibility with existing code.
    Parameters:
        value (float or np.ndarray): Linear value(s) to convert.
    Returns:
        float or np.ndarray: Value(s) in dB.
    """
    return dB(value)

def dBm(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert a linear value or array (in Watts) to decibels-milliwatts (dBm).
    Parameters:
        value (float or np.ndarray): Linear value(s) in Watts. Must be > 0 for valid dBm result.
    Returns:
        float or np.ndarray: Value(s) in dBm. Returns -np.inf for elements <= 0.
    """
    return 10 * np.log10(value * 1e3) if np.all(value > 0) else np.where(value > 0, 10 * np.log10(value * 1e3), -np.inf)

def dbm(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Alias for dBm function to maintain compatibility with existing code.
    Parameters:
        value (float or np.ndarray): Linear value(s) in Watts.
    Returns:
        float or np.ndarray: Value(s) in dBm.
    """
    return dBm(value)

def linear(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert a value or array in decibels (dB) to linear scale.
    Parameters:
        value (float or np.ndarray): Value(s) in dB.
    Returns:
        float or np.ndarray: Linear value(s). Returns 0 for elements that are -np.inf.
    """
    return 10 ** (value / 10) if np.all(value > -np.inf) else np.where(value > -np.inf, 10 ** (value / 10), 0)

def print_fps(filepaths):
    if filepaths is not None:
        print("Data saved to:")
        for fp in filepaths:
            print(fp)

def parse_complex(s):
    if pd.isna(s) or s == '':
        return np.nan
    return complex(s.strip('()'))

def interp_complex(target:NDArray, source:NDArray, y:NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Interpolates a complex array y from source to target.
    Parameters:
        target (np.ndarray): Target x-coordinates.
        source (np.ndarray): Source x-coordinates.
        y (np.ndarray[np.complex128]): Complex values to interpolate.
    Returns:
        np.ndarray[np.complex128]: Interpolated complex values at target.
    """
    # interpolate phase and magnitude separately
    phase_interp = np.interp(target, source, np.unwrap(np.angle(y)))
    mag_interp = np.interp(target, source, np.abs(y))
    
    # combine magnitude and phase into complex numbers
    return mag_interp * np.exp(1j * phase_interp)



def rms_resample(oldx: NDArray, newx: NDArray, y: NDArray) -> NDArray:
    """
    Resample dB data using RMS averaging to reduce the number of points.
    
    Parameters:
        oldx (NDArray): Original x-coordinates (e.g., frequency points)
        newx (NDArray): Target x-coordinates for resampling (fewer points)
        y (NDArray): Original y-values in dB scale
        
    Returns:
        NDArray: Resampled y-values in dB scale with length matching newx
    """
    if len(newx) >= len(oldx):
        # If we're not reducing points, just interpolate
        return np.interp(newx, oldx, y)
    
    # Convert dB to linear scale for RMS calculation
    y_linear = linear(y)
    
    # Initialize output array
    y_resampled = np.zeros(len(newx))
    
    # Calculate bin edges for the new grid
    # We'll create bins centered on newx points
    if len(newx) == 1:
        # Special case: single point
        y_resampled[0] = dB(np.sqrt(np.mean(y_linear**2)))
        return y_resampled
    
    # Calculate bin edges
    dx = (newx[-1] - newx[0]) / (len(newx) - 1)
    bin_edges = np.zeros(len(newx) + 1)
    bin_edges[0] = newx[0] - dx/2
    bin_edges[1:-1] = (newx[:-1] + newx[1:]) / 2
    bin_edges[-1] = newx[-1] + dx/2
    
    # Ensure bin edges don't go outside the original data range
    bin_edges[0] = max(bin_edges[0], oldx[0])
    bin_edges[-1] = min(bin_edges[-1], oldx[-1])
    
    # For each new point, find corresponding old points and calculate RMS
    for i in range(len(newx)):
        # Find indices of old points that fall within this bin
        if i == 0:
            mask = (oldx >= bin_edges[i]) & (oldx < bin_edges[i+1])
        else:
            mask = (oldx > bin_edges[i]) & (oldx <= bin_edges[i+1])
        
        if np.any(mask):
            # Calculate RMS of linear values in this bin
            rms_value = np.sqrt(np.mean(y_linear[mask]**2))
            y_resampled[i] = dB(rms_value)
        else:
            # No points in this bin, interpolate
            y_resampled[i] = np.interp(newx[i], oldx, y)
    
    return y_resampled