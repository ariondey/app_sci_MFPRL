import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
from datetime import datetime

BASE_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL\PCD_SOT_data"
OUTPUT_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL"
SAMPLING_RATE = 100  # Hz

FREQ_BANDS = {'LF': (0, 0.3), 'MF': (0.3, 1), 'HF': (1, 3)}

def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

def compute_psd(data, fs):
    n = len(data)
    freq = fftfreq(n, d=1/fs)[:n//2]
    fft_values = np.abs(fft(data))[:n//2]
    power_density = (fft_values ** 2) / n
    return freq, power_density

def integrate_power(freq, power_density, bands):
    integrated_power = {}
    for band_name, (low_f, high_f) in bands.items():
        band_mask = (freq >= low_f) & (freq <= high_f)
        integrated_power[band_name] = trapezoid(power_density[band_mask], freq[band_mask]) if np.any(band_mask) else 0
    return integrated_power

def compute_rambling_trembling(cop_signal, force_signal, sampling_rate=100):
    # Use force_signal in the computation if needed
    rambling = butter_lowpass_filter(cop_signal, cutoff=2.0, fs=sampling_rate)
    trembling = cop_signal - rambling
    return rambling, trembling

def process_csv_file(file_path, sampling_rate=100):
    try:
        filename = os.path.basename(file_path)
        pattern = r"(\w+)___(\d+)___(\w+)___(\d+)___(\d{1,2}-\d{1,2}-\d{4})___(\d+)___(\d+)\.csv"
        match = re.match(pattern, filename)
        if not match:
            print(f"Filename '{filename}' does not match pattern.")
            return None, None

        study = match.group(1)
        trial_global_num = int(match.group(2))
        subject_num = int(match.group(4))
        condition_num = int(match.group(6))
        trial_in_condition_num = int(match.group(7))

        df_raw = pd.read_csv(file_path)

        # Extract COP data
        cop_x_left = pd.to_numeric(df_raw['L.COFx'], errors='coerce').interpolate().values
        cop_y_left = pd.to_numeric(df_raw['L.COFy'], errors='coerce').interpolate().values
        cop_x_right = pd.to_numeric(df_raw['R.COFx'], errors='coerce').interpolate().values
        cop_y_right = pd.to_numeric(df_raw['R.COFy'], errors='coerce').interpolate().values

        # Extract force data
        force_x_left = pd.to_numeric(df_raw['L.Fx'], errors='coerce').ffill().bfill().values
        force_y_left = pd.to_numeric(df_raw['L.Fy'], errors='coerce').ffill().bfill().values
        force_x_right = pd.to_numeric(df_raw['R.Fx'], errors='coerce').ffill().bfill().values
        force_y_right = pd.to_numeric(df_raw['R.Fy'], errors='coerce').ffill().bfill().values

        # Compute rambling and trembling components for each direction
        rambling_x_left, trembling_x_left = compute_rambling_trembling(cop_x_left, force_x_left, sampling_rate)
        rambling_y_left, trembling_y_left = compute_rambling_trembling(cop_y_left, force_y_left, sampling_rate)
        rambling_x_right, trembling_x_right = compute_rambling_trembling(cop_x_right, force_x_right, sampling_rate)
        rambling_y_right, trembling_y_right = compute_rambling_trembling(cop_y_right, force_y_right, sampling_rate)

        # Combine left and right data (average them)
        cop_x_combined = np.nanmean([cop_x_left, cop_x_right], axis=0)
        cop_y_combined = np.nanmean([cop_y_left, cop_y_right], axis=0)
        rambling_x_combined = np.nanmean([rambling_x_left, rambling_x_right], axis=0)
        rambling_y_combined = np.nanmean([rambling_y_left, rambling_y_right], axis=0)
        trembling_x_combined = np.nanmean([trembling_x_left, trembling_x_right], axis=0)
        trembling_y_combined = np.nanmean([trembling_y_left, trembling_y_right], axis=0)

        # Compute PSD and integrated power for the X-axis signals
        freq_rambling_x, psd_rambling_x = compute_psd(rambling_x_combined, sampling_rate)
        freq_trembling_x, psd_trembling_x = compute_psd(trembling_x_combined, sampling_rate)

        rambling_power_x_bands = integrate_power(freq_rambling_x, psd_rambling_x, FREQ_BANDS)
        trembling_power_x_bands = integrate_power(freq_trembling_x, psd_trembling_x, FREQ_BANDS)

        # Compute PSD and integrated power for the Y-axis signals
        freq_rambling_y, psd_rambling_y = compute_psd(rambling_y_combined, sampling_rate)
        freq_trembling_y, psd_trembling_y = compute_psd(trembling_y_combined, sampling_rate)

        rambling_power_y_bands = integrate_power(freq_rambling_y, psd_rambling_y, FREQ_BANDS)
        trembling_power_y_bands = integrate_power(freq_trembling_y, psd_trembling_y, FREQ_BANDS)

        # Compute PSD and integrated power for the combined COPx and COPy signals
        freq_cop_x, psd_cop_x = compute_psd(cop_x_combined, sampling_rate)
        freq_cop_y, psd_cop_y = compute_psd(cop_y_combined, sampling_rate)

        cop_x_power_bands = integrate_power(freq_cop_x, psd_cop_x, FREQ_BANDS)
        cop_y_power_bands = integrate_power(freq_cop_y, psd_cop_y, FREQ_BANDS)

        result_data = {
            "Subject_ID": subject_num,
            "Condition": condition_num,
            "Trial": trial_in_condition_num,
            "Study": study,
            "Path_Length": np.sum(np.sqrt(np.diff(cop_x_combined)**2 + np.diff(cop_y_combined)**2)),
            **{f'Rambling_X_{band}_Power': val for band, val in rambling_power_x_bands.items()},
            **{f'Trembling_X_{band}_Power': val for band, val in trembling_power_x_bands.items()},
            **{f'Rambling_Y_{band}_Power': val for band, val in rambling_power_y_bands.items()},
            **{f'Trembling_Y_{band}_Power': val for band, val in trembling_power_y_bands.items()}
        }

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

def analyze_all_subjects(base_path):
    results_list = []
    cop_power_summary_list = []
    base_path_abs = os.path.abspath(base_path)
    
    for subject_folder in os.listdir(base_path_abs):
        subject_folder_path = os.path.join(base_path_abs, subject_folder)
        if os.path.isdir(subject_folder_path):
            for root_dirpath, _, filenames in os.walk(subject_folder_path):
                for file in filenames:
                    if file.endswith('.csv'):
                        full_file_path = os.path.join(root_dirpath, file)
                        processed_trial_data, cop_power_summary = process_csv_file(full_file_path)
                        if processed_trial_data is not None:
                            results_list.append(processed_trial_data)
                        if cop_power_summary is not None:
                            cop_power_summary_list.append(cop_power_summary)

    return pd.DataFrame(results_list), pd.DataFrame(cop_power_summary_list)

def export_cop_frequency_power_summary(results, output_dir="./cop_frequency_power_summary"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create DataFrame for results
    summary_list = []
    for result in results:
        summary = {
            "Subject_ID": result["Subject_ID"],
            "Condition": result["Condition"],
            "Trial": result["Trial"],
            "COP_X_LF_Power": result.get("COP_X_LF_Power", 0),
            "COP_X_MF_Power": result.get("COP_X_MF_Power", 0),
            "COP_X_HF_Power": result.get("COP_X_HF_Power", 0),
            "COP_Y_LF_Power": result.get("COP_Y_LF_Power", 0),
            "COP_Y_MF_Power": result.get("COP_Y_MF_Power", 0),
            "COP_Y_HF_Power": result.get("COP_Y_HF_Power", 0),
        }
        summary_list.append(summary)
    
    summary_df = pd.DataFrame(summary_list)
    
def export_rambling_trembling_stats(results_df, output_path):
    """
    Compute mean and standard deviation for trembling and rambling integrated power columns
    and save the stats to an Excel file.
    """
    if results_df.empty:
        print("No results available to compute stats.")
        return

    # Collect columns for integrated power of rambling and trembling (X-axis)
    stats_data = {}
    for band in FREQ_BANDS:  # e.g., 'LF', 'MF', 'HF'
        rambling_col = f'Rambling_X_{band}_Power'
        trembling_col = f'Trembling_X_{band}_Power'
        
        # Compute overall mean and std for the column (across all trials)
        stats_data[rambling_col + '_mean'] = [results_df[rambling_col].mean()]
        stats_data[rambling_col + '_std'] = [results_df[rambling_col].std()]
        stats_data[trembling_col + '_mean'] = [results_df[trembling_col].mean()]
        stats_data[trembling_col + '_std'] = [results_df[trembling_col].std()]
    
    stats_df = pd.DataFrame(stats_data)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_path, f'rambling_trembling_stats_{timestamp_str}.xlsx')
    stats_df.to_excel(output_file, index=False)
    print(f"Rambling/Trembling stats saved to {output_file}")




if __name__ == "__main__":
    print("Starting analysis...")
    
    results_df, cop_power_summary_df = analyze_all_subjects(BASE_PATH)
    
    if not results_df.empty:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(OUTPUT_PATH, f'pcd_results_{timestamp_str}.xlsx')
        
        results_df.to_excel(output_file_path, index=False)
        
        print(f"Analysis completed successfully. Results saved to {output_file_path}.")
        
        # Export mean/SD of trembling and rambling integrated power
        export_rambling_trembling_stats(results_df, OUTPUT_PATH)
        
    else:
        print("Analysis completed but no data was processed.")

    if not cop_power_summary_df.empty:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        cop_power_summary_file_path = os.path.join(OUTPUT_PATH, f'PCD_cop_power_summary_{timestamp_str}.xlsx')
        cop_power_summary_df.to_excel(cop_power_summary_file_path, index=False)
        print(f"COP power summary saved to {cop_power_summary_file_path}.")
        
        # Export COP frequency power summary to a separate Excel file
        export_cop_frequency_power_summary(cop_power_summary_df.to_dict('records'), OUTPUT_PATH)
        
    else:
        print("No COP power summary data was processed.")    