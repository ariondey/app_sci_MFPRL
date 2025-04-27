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

BASE_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL\COP_Clinical_curated"
OUTPUT_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL"
SAMPLING_RATE = 100  # Hz

FREQ_BANDS = {'LF': (0, 0.3), 'MF': (0.3, 1), 'HF': (1, 3)}

def butter_lowpass_filter(data, cutoff, fs, order=4):
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
    """
    Computes rambling as COP at IEPs (force = 0) and trembling as deviations from the IEP trajectory.
    """
    # Detect zero-crossings in horizontal force signal
    zero_crossings = np.where(np.diff(np.sign(force_signal)))[0]
    
    # Find exact IEP times and COP values via linear interpolation
    iep_times, iep_cop = [], []
    for i in zero_crossings:
        t1, t2 = i, i + 1
        f1, f2 = force_signal[t1], force_signal[t2]
        
        if f1 == 0:
            iep_times.append(t1)
            iep_cop.append(cop_signal[t1])
        else:
            alpha = -f1 / (f2 - f1)
            iep_time = t1 + alpha
            iep_cop_val = cop_signal[t1] + alpha * (cop_signal[t2] - cop_signal[t1])
            iep_times.append(iep_time)
            iep_cop.append(iep_cop_val)
    
    # Handle insufficient IEPs by extrapolating the rambling trajectory
    if len(iep_times) < 2:
        print("Insufficient IEPs detected. Extrapolating rambling trajectory.")
        rambling = np.interp(
            np.arange(len(cop_signal)) / sampling_rate,
            np.array(iep_times) / sampling_rate,
            iep_cop,
            left=iep_cop[0],
            right=iep_cop[-1]
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

def process_csv_file(file_path, sampling_rate=100):
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

        # Combine forces and COP across plates
        force_x_combined = force_x_left + force_x_right
        force_y_combined = force_y_left + force_y_right
        cop_x_combined = np.nanmean([cop_x_left, cop_x_right], axis=0)
        cop_y_combined = np.nanmean([cop_y_left, cop_y_right], axis=0)

        # Detrend and filter the force signal
        force_x_combined = detrend(force_x_combined)
        force_x_combined = butter_lowpass_filter(force_x_combined, cutoff=3, fs=sampling_rate)

        force_y_combined = detrend(force_y_combined)
        force_y_combined = butter_lowpass_filter(force_y_combined, cutoff=3, fs=sampling_rate)

        # Detect near-zero crossings with a small threshold
        threshold = 0.01
        zero_crossings_x = np.where(np.abs(force_x_combined) < threshold)[0]
        zero_crossings_y = np.where(np.abs(force_y_combined) < threshold)[0]

        # Compute rambling and trembling components
        rambling_x, trembling_x = compute_rambling_trembling(cop_x_combined, force_x_combined, sampling_rate)
        rambling_y, trembling_y = compute_rambling_trembling(cop_y_combined, force_y_combined, sampling_rate)

        # Compute PSD and integrated power for the X-axis signals
        freq_rambling_x, psd_rambling_x = compute_psd(rambling_x, sampling_rate)
        freq_trembling_x, psd_trembling_x = compute_psd(trembling_x, sampling_rate)

        rambling_power_x_bands = integrate_power(freq_rambling_x, psd_rambling_x, FREQ_BANDS)
        trembling_power_x_bands = integrate_power(freq_trembling_x, psd_trembling_x, FREQ_BANDS)

        # Compute PSD and integrated power for the Y-axis signals
        freq_rambling_y, psd_rambling_y = compute_psd(rambling_y, sampling_rate)
        freq_trembling_y, psd_trembling_y = compute_psd(trembling_y, sampling_rate)

        rambling_power_y_bands = integrate_power(freq_rambling_y, psd_rambling_y, FREQ_BANDS)
        trembling_power_y_bands = integrate_power(freq_trembling_y, psd_trembling_y, FREQ_BANDS)

        result_data = {
            "Subject_ID": subject_num,
            "Condition": condition_num,
            "Trial": trial_in_condition_num,
            "Study": study,
            "Path_Length": np.sum(np.sqrt(np.diff(cop_x_combined)**2 + np.diff(cop_y_combined)**2)),
            **{f'Rambling_X_{band}_Power': val for band, val in rambling_power_x_bands.items()},
            **{f'Trembling_X_{band}_Power': val for band, val in trembling_power_x_bands.items()},
            **{f'Rambling_Y_{band}_Power': val for band, val in rambling_power_y_bands.items()},
            **{f'Trembling_Y_{band}_Power': val for band, val in trembling_power_y_bands.items()},
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