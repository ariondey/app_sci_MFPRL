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
    zero_crossings = np.where(np.diff(np.sign(force_signal)))[0]
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
    t_full = np.arange(len(cop_signal)) / sampling_rate
    iep_times_sec = np.array(iep_times) / sampling_rate
    rambling = CubicSpline(iep_times_sec, iep_cop, extrapolate=False)(t_full)
    rambling = pd.Series(rambling).ffill().bfill().values
    trembling = cop_signal - rambling
    return rambling, trembling

def process_csv_file(file_path, sampling_rate=100):
    try:
        filename = os.path.basename(file_path)
        pattern = r"(\d+)__(\w+)__(\d+)__(\d+)__(\d+)__.*\.csv"
        match = re.match(pattern, filename)
        if not match:
            print(f"Filename '{filename}' does not match pattern.")
            return None
        trial_global_num = int(match.group(1))
        study = match.group(2)
        subject_num = int(match.group(3))
        condition_num = int(match.group(4))
        trial_in_condition_num = int(match.group(5))
        df_raw = pd.read_csv(file_path)
        cop_x_left = pd.to_numeric(df_raw['L.COFx'], errors='coerce').interpolate().values
        cop_y_left = pd.to_numeric(df_raw['L.COFy'], errors='coerce').interpolate().values
        cop_x_right = pd.to_numeric(df_raw['R.COFx'], errors='coerce').interpolate().values
        cop_y_right = pd.to_numeric(df_raw['R.COFy'], errors='coerce').interpolate().values
        force_x_left = pd.to_numeric(df_raw['L.Fx'], errors='coerce').ffill().bfill().values
        force_y_left = pd.to_numeric(df_raw['L.Fy'], errors='coerce').ffill().bfill().values
        force_x_right = pd.to_numeric(df_raw['R.Fx'], errors='coerce').ffill().bfill().values
        force_y_right = pd.to_numeric(df_raw['R.Fy'], errors='coerce').ffill().bfill().values
        force_x_combined = force_x_left + force_x_right
        force_y_combined = force_y_left + force_y_right
        cop_x_combined = np.nanmean([cop_x_left, cop_x_right], axis=0)
        cop_y_combined = np.nanmean([cop_y_left, cop_y_right], axis=0)
        force_x_combined = detrend(force_x_combined)
        force_x_combined = butter_lowpass_filter(force_x_combined, cutoff=3, fs=sampling_rate)
        force_y_combined = detrend(force_y_combined)
        force_y_combined = butter_lowpass_filter(force_y_combined, cutoff=3, fs=sampling_rate)
        rambling_x, trembling_x = compute_rambling_trembling(cop_x_combined, force_x_combined, sampling_rate)
        rambling_y, trembling_y = compute_rambling_trembling(cop_y_combined, force_y_combined, sampling_rate)
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

def compute_mean_sd_trembling_rambling(results_df):
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

def export_mean_sd_to_excel(mean_sd_df, output_dir="./mean_sd_summary"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "TCOA_mean_sd_trembling_rambling.xlsx")
    mean_sd_df.to_excel(output_file, index=False)
    print(f"Mean/SD of trembling and rambling components exported to: {output_file}")

if __name__ == "__main__":
    print("Starting analysis...")
    results_df = analyze_all_subjects(BASE_PATH)
    if not results_df.empty:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(OUTPUT_PATH, f'tcoa_results_{timestamp_str}.xlsx')
        results_df.to_excel(output_file_path, index=False)
        print(f"Analysis completed successfully. Results saved to {output_file_path}.")
        mean_sd_df = compute_mean_sd_trembling_rambling(results_df)
        export_mean_sd_to_excel(mean_sd_df, OUTPUT_PATH)
    else:
        print("Analysis completed but no data was processed.")