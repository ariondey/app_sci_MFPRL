import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
from datetime import datetime

BASE_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL\COP_Clinical_curated"
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

def compute_rambling_trembling(cop_signal, sampling_rate=100):
    rambling = butter_lowpass_filter(cop_signal, cutoff=2.0, fs=sampling_rate)
    trembling = cop_signal - rambling
    return rambling, trembling

def process_csv_file(file_path, sampling_rate=100):
    try:
        filename = os.path.basename(file_path)
        pattern = r".*__(\d+)__(\d+)__(\d+)__.*\.csv"
        match = re.match(pattern, filename)
        if not match:
            print(f"Filename '{filename}' does not match pattern.")
            return None

        trial_global_num = int(match.group(1))
        subject_num = int(match.group(2))
        condition_num = int(match.group(3))

        # Swap the values as per the requirement
        subject_num, condition_num, trial_global_num = condition_num, trial_global_num, subject_num

        df_raw = pd.read_csv(file_path)

        cop_x_left = pd.to_numeric(df_raw['L.COFx'], errors='coerce').interpolate().values
        cop_y_left = pd.to_numeric(df_raw['L.COFy'], errors='coerce').interpolate().values
        cop_x_right = pd.to_numeric(df_raw['R.COFx'], errors='coerce').interpolate().values
        cop_y_right = pd.to_numeric(df_raw['R.COFy'], errors='coerce').interpolate().values

        cop_x_combined = np.nanmean([cop_x_left, cop_x_right], axis=0)
        cop_y_combined = np.nanmean([cop_y_left, cop_y_right], axis=0)

        rambling_x, trembling_x = compute_rambling_trembling(cop_x_combined, sampling_rate)

        freq_rambling_x, psd_rambling_x = compute_psd(rambling_x, sampling_rate)
        freq_trembling_x, psd_trembling_x = compute_psd(trembling_x, sampling_rate)

        rambling_power_x_bands = integrate_power(freq_rambling_x, psd_rambling_x, FREQ_BANDS)
        trembling_power_x_bands = integrate_power(freq_trembling_x, psd_trembling_x, FREQ_BANDS)

        result_data = {
            "Subject_ID": condition_num,
            "Condition": trial_global_num,
            "Trial": subject_num,
            "Path_Length": np.sum(np.sqrt(np.diff(cop_x_combined)**2 + np.diff(cop_y_combined)**2)),
            **{f'Rambling_X_{band}_Power': val for band, val in rambling_power_x_bands.items()},
            **{f'Trembling_X_{band}_Power': val for band, val in trembling_power_x_bands.items()}
        }

        return result_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def analyze_all_subjects(base_path):
    results_list = []
    base_path_abs = os.path.abspath(base_path)
    
    for root_dirpath, _, filenames in os.walk(base_path_abs):
        for file in filenames:
            if file.endswith('.csv'):
                full_file_path = os.path.join(root_dirpath, file)
                processed_trial_data = process_csv_file(full_file_path)
                if processed_trial_data is not None:
                    results_list.append(processed_trial_data)

    return pd.DataFrame(results_list)

if __name__ == "__main__":
    print("Starting analysis...")
    
    results_df = analyze_all_subjects(BASE_PATH)

    if not results_df.empty:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(OUTPUT_PATH, f'tcoa_results_{timestamp_str}.xlsx')
        
        results_df.to_excel(output_file_path, index=False)
        
        print(f"Analysis completed successfully. Results saved to {output_file_path}.")
        
    else:
        print("Analysis completed but no data was processed.")
