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

def compute_rambling_trembling(cop_signal, force_signal, sampling_rate=100):
    # Use force_signal in the computation if needed
    rambling = butter_lowpass_filter(cop_signal, cutoff=2.0, fs=sampling_rate)
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
            **{f'Trembling_Y_{band}_Power': val for band, val in trembling_power_y_bands.items()},
            "Rambling_X": np.mean(rambling_x_combined),
            "Trembling_X": np.mean(trembling_x_combined),
            "Rambling_Y": np.mean(rambling_y_combined),
            "Trembling_Y": np.mean(trembling_y_combined)
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

def compute_mean_sd_trembling_rambling(results_df):
    """Compute the mean and standard deviation of trembling and rambling components."""
    mean_sd_data = {
        "Component": [],
        "Direction": [],
        "Mean": [],
        "Standard_Deviation": []
    }

    for component in ["Rambling", "Trembling"]:
        for direction in ["X", "Y"]:
            column_name = f"{component}_{direction}"
            if column_name in results_df.columns:
                mean_sd_data["Component"].append(component)
                mean_sd_data["Direction"].append(direction)
                mean_sd_data["Mean"].append(results_df[column_name].mean())
                mean_sd_data["Standard_Deviation"].append(results_df[column_name].std())

    return pd.DataFrame(mean_sd_data)

def compute_mean_sd_trembling_rambling_by_cohort(results_df):
    """Compute the mean and standard deviation of trembling and rambling components, separated by cohort."""
    mean_sd_data = {
        "Cohort": [],
        "Component": [],
        "Direction": [],
        "Mean": [],
        "Standard_Deviation": [],
        "Participants": []  # Add a column for the number of participants
    }

    # Define cohorts based on Subject_ID
    results_df["Cohort"] = results_df["Subject_ID"].apply(
        lambda x: "100s" if 100 <= x < 200 else "200s" if 200 <= x < 300 else "400s" if 400 <= x < 500 else "Other"
    )

    for cohort in ["100s", "200s", "400s"]:
        cohort_df = results_df[results_df["Cohort"] == cohort]
        num_participants = cohort_df["Subject_ID"].nunique()  # Count unique participants in the cohort
        for component in ["Rambling", "Trembling"]:
            for direction in ["X", "Y"]:
                column_name = f"{component}_{direction}"
                if column_name in cohort_df.columns:
                    mean_sd_data["Cohort"].append(cohort)
                    mean_sd_data["Component"].append(component)
                    mean_sd_data["Direction"].append(direction)
                    mean_sd_data["Mean"].append(cohort_df[column_name].mean())
                    mean_sd_data["Standard_Deviation"].append(cohort_df[column_name].std())
                    mean_sd_data["Participants"].append(num_participants)  # Add the participant count

    return pd.DataFrame(mean_sd_data)

def export_mean_sd_to_excel(mean_sd_df, output_dir="./mean_sd_summary"):
    """Export the mean and standard deviation of trembling and rambling components to an Excel file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "TCOA_mean_sd_trembling_rambling.xlsx")
    mean_sd_df.to_excel(output_file, index=False)
    print(f"Mean/SD of trembling and rambling components exported to: {output_file}")

def export_mean_sd_by_cohort_to_excel(mean_sd_df, output_dir="./mean_sd_summary_by_cohort"):
    """Export the mean and standard deviation of trembling and rambling components by cohort to an Excel file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "TCOA_mean_sd_trembling_rambling_by_cohort.xlsx")
    mean_sd_df.to_excel(output_file, index=False)
    print(f"Mean/SD of trembling and rambling components by cohort exported to: {output_file}")

if __name__ == "__main__":
    print("Starting analysis...")
    
    results_df, cop_power_summary_df = analyze_all_subjects(BASE_PATH)

    if not results_df.empty:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(OUTPUT_PATH, f'tcoa_results_{timestamp_str}.xlsx')
        
        results_df.to_excel(output_file_path, index=False)
        
        print(f"Analysis completed successfully. Results saved to {output_file_path}.")
        
        # Compute and export mean/SD of trembling and rambling components
        mean_sd_df = compute_mean_sd_trembling_rambling(results_df)
        export_mean_sd_to_excel(mean_sd_df, OUTPUT_PATH)
        
        # Compute and export mean/SD of trembling and rambling components by cohort
        mean_sd_by_cohort_df = compute_mean_sd_trembling_rambling_by_cohort(results_df)
        export_mean_sd_by_cohort_to_excel(mean_sd_by_cohort_df, OUTPUT_PATH)
        
    else:
        print("Analysis completed but no data was processed.")

    if not cop_power_summary_df.empty:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        cop_power_summary_file_path = os.path.join(OUTPUT_PATH, f'TCOA_cop_power_summary_{timestamp_str}.xlsx')
        
        cop_power_summary_df.to_excel(cop_power_summary_file_path, index=False)
        
        print(f"COP power summary saved to {cop_power_summary_file_path}.")
        
    else:
        print("No COP power summary data was processed.")