import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Annotated, Union
from numpy.typing import NDArray

from Constants import *


def process_noise_data(data_file_base: str, num_traces: int = NUM_TRACES, **kwargs) -> tuple[
    Annotated[NDArray[np.complex128], (DATASET_LENGTH,)],
    Annotated[NDArray[np.complex128], (DATASET_LENGTH,)],
    Annotated[NDArray[np.complex128], (DATASET_LENGTH,)]
]:
    """
    Process multiple noise data files and return averaged power spectral densities.
    
    Processes data in chunks of SAMPLE_CUTOFF samples to handle large files efficiently.
    Calculates power spectral density (PSD) and cross-spectral density (CSD) from 
    correlator noise measurements.
    
    Parameters:
        data_file_base (str): Base string for noise data filepaths (without index and .npy extension).
        num_traces (int, optional): Number of data files to process. Defaults to NUM_TRACES.
        **kwargs: Additional processing options:
            fft_freq (np.ndarray, optional): Custom frequency array for FFT calculations.
            
    Returns:
        tuple[NDArray, NDArray, NDArray]: Tuple containing:
            - ch1_avg_psd: Average power spectral density for channel 1 in V²/Hz
            - ch2_avg_psd: Average power spectral density for channel 2 in V²/Hz  
            - csd_avg: Average cross-spectral density between channels in V²/Hz
            
    Notes:
        - Files are processed in chunks to manage memory usage for large datasets
        - Only processes files that exist; missing files are skipped with warning
        - Returns zero arrays if no valid data chunks are processed
        - All PSDs are calculated over 1-2 GHz frequency range
    """
    # -------------------- Initialization --------------------
    fft_freq = kwargs.get("fft_freq", np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9)))
    power_data = np.zeros((3, DATASET_LENGTH), dtype=np.complex128)
    total_chunks_processed = 0
    

    # -------------------- Process Noise Files and Chunks --------------------
    # loop over the specified number of files
    for idx in tqdm(range(num_traces), desc="Processing Files", unit="file", total=num_traces, colour='#808080'):
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
    Process noise data from two channels to compute power and cross-spectral densities.
    
    Applies windowing, performs FFT analysis, and calculates power spectral density (PSD)
    and cross-spectral density (CSD) for correlator measurements. Processing is limited
    to the 1-2 GHz frequency band of interest.
    
    Parameters:
        data_array (NDArray): 2D array of raw noise data with shape (2, N) where each row 
            corresponds to a channel.
        fft_freq (NDArray): 1D array of frequency bins corresponding to the FFT output in Hz.
        **kwargs: Optional plotting and processing parameters:
            graph_fft (bool, optional): Plot FFT magnitude of both channels. Defaults to False.
            graph_psd (bool, optional): Plot PSD of both channels. Defaults to False.
            
    Returns:
        tuple[NDArray, NDArray, NDArray]: Tuple containing:
            - ch1_psd: Power spectral density of channel 1 in V²/Hz (1-2 GHz only)
            - ch2_psd: Power spectral density of channel 2 in V²/Hz (1-2 GHz only)
            - csd: Cross-spectral density between channels in V²/Hz (1-2 GHz only)
            
    Notes:
        - Applies Blackman window to reduce spectral leakage
        - Converts raw ADC counts to voltage using VOLT_PER_TICK scaling
        - PSD includes factor of 2 for single-sided spectrum
        - Window power correction applied for accurate PSD calculation
        - Only 1-2 GHz frequency range retained for correlator analysis
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
    Convert linear values to decibels (dB) scale.
    
    Parameters:
        value (Union[float, NDArray]): Linear value(s) to convert. Must be > 0 for valid result.
        
    Returns:
        Union[float, NDArray]: Value(s) in dB. Returns -np.inf for elements ≤ 0.
        
    Notes:
        - Uses 10*log10 conversion formula
        - Handles both scalar and array inputs
        - Returns -inf for non-positive values to avoid math domain errors
    """
    return 10 * np.log10(value) if np.all(value > 0) else np.where(value > 0, 10 * np.log10(value), -np.inf)

def db(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Alias for dB function to maintain backward compatibility.
    
    Parameters:
        value (Union[float, NDArray]): Linear value(s) to convert.
        
    Returns:
        Union[float, NDArray]: Value(s) in dB.
    """
    return dB(value)

def dBm(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert power values in Watts to decibels-milliwatts (dBm).
    
    Parameters:
        value (Union[float, NDArray]): Power value(s) in Watts. Must be > 0 for valid result.
        
    Returns:
        Union[float, NDArray]: Power value(s) in dBm. Returns -np.inf for elements ≤ 0.
        
    Notes:
        - Uses formula: dBm = 10*log10(P_watts * 1000)
        - Reference level is 1 milliwatt (1 mW = 0 dBm)
        - Commonly used for RF power measurements
    """
    return 10 * np.log10(value * 1e3) if np.all(value > 0) else np.where(value > 0, 10 * np.log10(value * 1e3), -np.inf)

def dbm(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Alias for dBm function to maintain backward compatibility.
    
    Parameters:
        value (Union[float, NDArray]): Power value(s) in Watts.
        
    Returns:
        Union[float, NDArray]: Power value(s) in dBm.
    """
    return dBm(value)

def linear(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert decibel (dB) values to linear scale.
    
    Parameters:
        value (Union[float, NDArray]): Value(s) in dB scale.
        
    Returns:
        Union[float, NDArray]: Linear scale value(s). Returns 0 for -np.inf elements.
        
    Notes:
        - Uses formula: linear = 10^(dB/10)
        - Inverse operation of dB function
        - Returns 0 for -inf inputs (representing zero power/amplitude)
    """
    return 10 ** (value / 10) if np.all(value > -np.inf) else np.where(value > -np.inf, 10 ** (value / 10), 0)

def print_fps(filepaths: list[str]) -> None:
    """
    Print a formatted list of file paths with blue color formatting.
    
    Parameters:
        filepaths (list[str]): List of file paths to display.
        
    Returns:
        None
        
    Notes:
        - Prints "Data saved to:" header followed by each file path
        - File paths are displayed in blue color using ANSI escape codes
        - Does nothing if filepaths is None or empty
    """
    if filepaths is not None:
        print("Data saved to:")
        for fp in filepaths:
            print(f"\033[34m{fp}\033[0m")  # Blue color

def parse_complex(s: str) -> complex:
    """
    Parse a string representation of a complex number.
    
    Parameters:
        s (str): String representation of complex number, may include parentheses.
        
    Returns:
        complex: Parsed complex number, or np.nan if input is invalid/empty.
        
    Notes:
        - Handles strings with or without parentheses
        - Returns np.nan for empty strings or pandas NaN values
        - Used for loading complex S-parameter data from CSV files
    """
    if pd.isna(s) or s == '':
        return np.nan
    return complex(s.strip('()'))

def interp_complex(target: NDArray, source: NDArray, y: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Interpolate complex-valued data by separately interpolating magnitude and phase.
    
    Performs complex interpolation by unwrapping the phase to handle discontinuities
    properly, then recombining magnitude and phase components. This approach preserves
    the complex nature of the data better than interpolating real and imaginary parts.
    
    Parameters:
        target (NDArray): Target x-coordinates for interpolated values.
        source (NDArray): Source x-coordinates corresponding to input data.
        y (NDArray[np.complex128]): Complex values to interpolate.
        
    Returns:
        NDArray[np.complex128]: Interpolated complex values at target coordinates.
        
    Notes:
        - Phase is unwrapped before interpolation to handle 2π discontinuities
        - Magnitude and phase are interpolated separately using linear interpolation
        - Result maintains complex data integrity better than real/imaginary interpolation
        - Useful for S-parameter and other RF measurement data interpolation
    """
    # interpolate phase and magnitude separately
    phase_interp = np.interp(target, source, np.unwrap(np.angle(y)))
    mag_interp = np.interp(target, source, np.abs(y))
    
    # combine magnitude and phase into complex numbers
    return mag_interp * np.exp(1j * phase_interp)



def rms_resample(oldx: NDArray, newx: NDArray, y: NDArray) -> NDArray:
    """
    Resample dB-scale data using RMS averaging to reduce data points while preserving power.
    
    Converts dB data to linear scale, performs RMS averaging within bins, then converts
    back to dB. This preserves the power content better than simple linear interpolation
    when downsampling spectral data.
    
    Parameters:
        oldx (NDArray): Original x-coordinates (e.g., frequency points).
        newx (NDArray): Target x-coordinates for resampling (typically fewer points).
        y (NDArray): Original y-values in dB scale to be resampled.
        
    Returns:
        NDArray: Resampled y-values in dB scale with length matching newx.
        
    Notes:
        - If newx has more points than oldx, falls back to linear interpolation
        - Creates bins centered on newx points for RMS averaging
        - Bin edges are constrained to original data range
        - For bins with no data points, uses linear interpolation
        - Preserves power content better than linear resampling for spectral data
        - Handles special case of single target point
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
