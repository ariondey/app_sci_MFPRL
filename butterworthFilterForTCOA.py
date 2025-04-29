import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from datetime import datetime
import matplotlib.pyplot as plt

# Define base paths for input data and output results
BASE_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL\COP_Clinical_curated"
OUTPUT_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL"
SAMPLING_RATE = 100  # Sampling rate in Hz

# Define frequency bands for analysis
FREQ_BANDS = {'LF': (0, 0.3), 'MF': (0.3, 1), 'HF': (1, 3)}

# Function to apply a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Applies a low-pass Butterworth filter to the input data.
    Args:
        data: Input signal
        cutoff: Cutoff frequency in Hz
        fs: Sampling rate in Hz
        order: Filter order
    Returns:
        Filtered signal
    """
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

# Function to compute the Power Spectral Density (PSD) of a signal
def compute_psd(data, fs):
    """
    Computes the Power Spectral Density (PSD) of the input signal.
    Args:
        data: Input signal
        fs: Sampling rate in Hz
    Returns:
        freq: Frequency bins
        power_density: Power spectral density
    """
    n = len(data)
    freq = fftfreq(n, d=1/fs)[:n//2]
    fft_values = np.abs(fft(data))[:n//2]
    power_density = (fft_values ** 2) / n
    return freq, power_density

# Function to integrate power over specified frequency bands
def integrate_power(freq, power_density, bands):
    """
    Integrates power over specified frequency bands.
    Args:
        freq: Frequency bins
        power_density: Power spectral density
        bands: Dictionary of frequency bands
    Returns:
        integrated_power: Dictionary of integrated power for each band
    """
    integrated_power = {}
    for band_name, (low_f, high_f) in bands.items():
        band_mask = (freq >= low_f) & (freq <= high_f)
        integrated_power[band_name] = trapezoid(power_density[band_mask], freq[band_mask]) if np.any(band_mask) else 0
    return integrated_power

# Function to compute rambling and trembling components
def compute_rambling_trembling(cop_signal, force_signal, sampling_rate=100):
    """
    Computes rambling as COP excursion (COP - mean(COP)) at IEPs (force = 0),
    resampled to match the original sampling rate, and trembling as deviations
    from the IEP trajectory.
    Args:
        cop_signal: Center of Pressure (COP) signal
        force_signal: Force signal
        sampling_rate: Sampling rate in Hz
    Returns:
        rambling: Rambling component
        trembling: Trembling component
    """
    # Compute the mean-centered COP (COP excursion)
    cop_excursion = cop_signal - np.mean(cop_signal)

    # Detect zero-crossings in horizontal force signal
    zero_crossings = np.where(np.diff(np.sign(force_signal)))[0]
    iep_times, iep_cop = [], []

    # Find exact IEP times and COP values via linear interpolation
    for i in zero_crossings:
        t1, t2 = i, i + 1
        f1, f2 = force_signal[t1], force_signal[t2]
        if f1 == 0:
            iep_times.append(t1)
            iep_cop.append(cop_excursion[t1])
        else:
            alpha = -f1 / (f2 - f1)
            iep_time = t1 + alpha
            iep_cop_val = cop_excursion[t1] + alpha * (cop_excursion[t2] - cop_excursion[t1])
            iep_times.append(iep_time)
            iep_cop.append(iep_cop_val)

    # Handle insufficient IEPs
    if len(iep_times) < 2:
        print("Insufficient IEPs detected. Extrapolating rambling trajectory.")
        rambling = np.interp(
            np.arange(len(cop_signal)) / sampling_rate,
            np.array(iep_times) / sampling_rate,
            iep_cop,
            left=iep_cop[0] if iep_cop else 0,
            right=iep_cop[-1] if iep_cop else 0
        )
        trembling = cop_signal - rambling
        return rambling, trembling

    # Interpolate the IEP trajectory using cubic spline
    t_full = np.arange(len(cop_signal)) / sampling_rate
    iep_times_sec = np.array(iep_times) / sampling_rate
    rambling = CubicSpline(iep_times_sec, iep_cop, extrapolate=False)(t_full)

    # Replace edge NaNs with nearest valid values
    rambling = pd.Series(rambling).ffill().bfill().values

    # Compute trembling as deviations from the IEP trajectory
    trembling = cop_signal - rambling

    return rambling, trembling

# Function to process a single CSV file
def process_csv_file(file_path, sampling_rate=100):
    """
    Processes a single CSV file to compute rambling and trembling components and COP power data.
    Args:
        file_path: Path to the CSV file
        sampling_rate: Sampling rate in Hz
    Returns:
        result_data: Dictionary of computed results
        cop_power_summary: Dictionary of COP power summary
    """
    try:
        filename = os.path.basename(file_path)
        pattern = r"(\d+)__(\w+)__(\d+)__(\d+)__(\d+)__.*\.csv"
        match = re.match(pattern, filename)
        if not match:
            print(f"Filename '{filename}' does not match pattern.")
            return None, None
        trial_global_num = int(match.group(1))
        study = match.group(2)
        subject_num = int(match.group(3))
        condition_num = int(match.group(4))
        trial_in_condition_num = int(match.group(5))
        df_raw = pd.read_csv(file_path)

        # Extract COP and force data
        cop_x_left = pd.to_numeric(df_raw['L.COFx'], errors='coerce').interpolate().values
        cop_y_left = pd.to_numeric(df_raw['L.COFy'], errors='coerce').interpolate().values
        cop_x_right = pd.to_numeric(df_raw['R.COFx'], errors='coerce').interpolate().values
        cop_y_right = pd.to_numeric(df_raw['R.COFy'], errors='coerce').interpolate().values
        force_x_left = pd.to_numeric(df_raw['L.Fx'], errors='coerce').ffill().bfill().values
        force_y_left = pd.to_numeric(df_raw['L.Fy'], errors='coerce').ffill().bfill().values
        force_x_right = pd.to_numeric(df_raw['R.Fx'], errors='coerce').ffill().bfill().values
        force_y_right = pd.to_numeric(df_raw['R.Fy'], errors='coerce').ffill().bfill().values

        # Combine forces and COP across plates
        force_x_combined = force_x_left + force_x_right
        force_y_combined = force_y_left + force_y_right
        cop_x_combined = np.nanmean([cop_x_left, cop_x_right], axis=0)
        cop_y_combined = np.nanmean([cop_y_left, cop_y_right], axis=0)

        # Preprocess the force signal with a low-pass filter (cutoff > 5 Hz)
        force_x_combined = detrend(force_x_combined)
        force_x_combined = butter_lowpass_filter(force_x_combined, cutoff=6, fs=sampling_rate)
        force_y_combined = detrend(force_y_combined)
        force_y_combined = butter_lowpass_filter(force_y_combined, cutoff=6, fs=sampling_rate)

        # Compute rambling and trembling components
        rambling_x, trembling_x = compute_rambling_trembling(cop_x_combined, force_x_combined, sampling_rate)
        rambling_y, trembling_y = compute_rambling_trembling(cop_y_combined, force_y_combined, sampling_rate)

        # Compute PSD and integrated power for the combined COP signals
        freq_cop_x, psd_cop_x = compute_psd(cop_x_combined, sampling_rate)
        freq_cop_y, psd_cop_y = compute_psd(cop_y_combined, sampling_rate)

        cop_x_power_bands = integrate_power(freq_cop_x, psd_cop_x, FREQ_BANDS)
        cop_y_power_bands = integrate_power(freq_cop_y, psd_cop_y, FREQ_BANDS)

        # Prepare result data
        result_data = {
            "Subject_ID": subject_num,
            "Condition": condition_num,
            "Trial": trial_in_condition_num,
            "Study": study,
            "Rambling_X": rambling_x.mean(),
            "Trembling_X": trembling_x.mean(),
            "Rambling_Y": rambling_y.mean(),
            "Trembling_Y": trembling_y.mean(),
        }

        # Prepare COP power summary
        cop_power_summary = {
            "Subject_ID": subject_num,
            "Condition": condition_num,
            "Trial": trial_in_condition_num,
            "Study": study,
            **{f'COP_X_{band}_Power': val for band, val in cop_x_power_bands.items()},
            **{f'COP_Y_{band}_Power': val for band, val in cop_y_power_bands.items()}
        }

        return result_data, cop_power_summary
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# Function to analyze all subjects in the dataset
def analyze_all_subjects(base_path):
    """
    Analyzes all subjects in the dataset by processing all CSV files.
    Args:
        base_path: Path to the dataset directory
    Returns:
        DataFrame of results and COP power summary
    """
    results_list = []
    cop_power_summary_list = []
    base_path_abs = os.path.abspath(base_path)
    for root_dirpath, _, filenames in os.walk(base_path_abs):
        for file in filenames:
            if file.endswith('.csv'):
                full_file_path = os.path.join(root_dirpath, file)
                processed_trial_data, cop_power_summary = process_csv_file(full_file_path)
                if processed_trial_data is not None:
                    results_list.append(processed_trial_data)
                if cop_power_summary is not None:
                    cop_power_summary_list.append(cop_power_summary)
    return pd.DataFrame(results_list), pd.DataFrame(cop_power_summary_list)

# Function to compute mean and standard deviation of trembling and rambling components
def compute_mean_sd_trembling_rambling(results_df):
    """
    Computes the mean and standard deviation of trembling and rambling components, grouped by cohort.
    Args:
        results_df: DataFrame of results
    Returns:
        DataFrame of mean and standard deviation
    """
    mean_sd_data = {
        "Cohort": [],
        "Component": [],
        "Direction": [],
        "Mean": [],
        "Standard_Deviation": [],
        "Participants": []
    }
    results_df['Cohort'] = results_df['Subject_ID'].apply(lambda x: f"{x // 100 * 100}s")
    for cohort, cohort_df in results_df.groupby('Cohort'):
        participant_count = cohort_df['Subject_ID'].nunique()
        for component in ["Rambling", "Trembling"]:
            for direction in ["X", "Y"]:
                column_name = f"{component}_{direction}"
                if column_name in cohort_df.columns:
                    mean_sd_data["Cohort"].append(cohort)
                    mean_sd_data["Component"].append(component)
                    mean_sd_data["Direction"].append(direction)
                    mean_sd_data["Mean"].append(cohort_df[column_name].mean())
                    mean_sd_data["Standard_Deviation"].append(cohort_df[column_name].std())
                    mean_sd_data["Participants"].append(participant_count)
    return pd.DataFrame(mean_sd_data)

# Function to export mean and standard deviation to an Excel file
def export_mean_sd_to_excel(mean_sd_df, output_dir="./mean_sd_summary"):
    """
    Exports the mean and standard deviation of trembling and rambling components to an Excel file.
    Args:
        mean_sd_df: DataFrame of mean and standard deviation
        output_dir: Directory to save the Excel file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "TCOA_mean_sd_trembling_rambling.xlsx")
    mean_sd_df.to_excel(output_file, index=False)
    print(f"Mean/SD of trembling and rambling components exported to: {output_file}")

def export_cop_power_summary(cop_power_summary_df, output_dir="./cop_power_summary"):
    """
    Exports the COP power summary to an Excel file.
    Args:
        cop_power_summary_df: DataFrame of COP power summary
        output_dir: Directory to save the Excel file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "TCOA_COP_Power_Summary.xlsx")
    cop_power_summary_df.to_excel(output_file, index=False)
    print(f"COP power summary exported to: {output_file}")

# Main script execution
if __name__ == "__main__":
    print("Starting analysis...")
    results_df, cop_power_summary_df = analyze_all_subjects(BASE_PATH)
    if not results_df.empty:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(OUTPUT_PATH, f'tcoa_results_{timestamp_str}.xlsx')
        results_df.to_excel(output_file_path, index=False)
        print(f"Analysis completed successfully. Results saved to {output_file_path}.")
        
        # Compute and export mean/SD of trembling and rambling
        mean_sd_df = compute_mean_sd_trembling_rambling(results_df)
        export_mean_sd_to_excel(mean_sd_df, OUTPUT_PATH)
    else:
        print("Analysis completed but no data was processed.")

    if not cop_power_summary_df.empty:
        export_cop_power_summary(cop_power_summary_df, OUTPUT_PATH)
    else:
        print("No COP power summary data was processed.")