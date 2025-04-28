import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

BASE_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL\PCD_SOT_data"
OUTPUT_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL"
SAMPLING_RATE = 100  # Hz
FREQ_BANDS = {'LF': (0, 0.3), 'MF': (0.3, 1), 'HF': (1, 3)}

def butter_lowpass_filter(data, cutoff, fs, order=4):  # Increase filter order to 4
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
    Computes rambling as COP excursion (COP - mean(COP)) at IEPs (force = 0),
    resampled to match the original sampling rate, and trembling as deviations
    from the IEP trajectory.
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

        # Combine forces and COP across plates (critical for bipedal stance)
        force_x_combined = force_x_left + force_x_right  # Total horizontal force
        force_y_combined = force_y_left + force_y_right
        cop_x_combined = np.nanmean([cop_x_left, cop_x_right], axis=0)  # Simplified COP merge
        cop_y_combined = np.nanmean([cop_y_left, cop_y_right], axis=0)

        # Preprocess the force signal with a low-pass filter (cutoff > 5 Hz)
        force_x_combined = detrend(force_x_combined)
        force_x_combined = butter_lowpass_filter(force_x_combined, cutoff=6, fs=sampling_rate)

        force_y_combined = detrend(force_y_combined)
        force_y_combined = butter_lowpass_filter(force_y_combined, cutoff=6, fs=sampling_rate)

        # Detect zero-crossings in combined force signals
        zero_crossings_x = np.where(np.diff(np.sign(force_x_combined)))[0]
        zero_crossings_y = np.where(np.diff(np.sign(force_y_combined)))[0]
        print(f"Zero-crossings in Force X: {len(zero_crossings_x)}")
        print(f"Zero-crossings in Force Y: {len(zero_crossings_y)}")

        # Weight IEPs based on the magnitude of the force signal
        weights = np.abs(force_x_combined[zero_crossings_x])
        weighted_iep_cop_x = np.average(cop_x_combined[zero_crossings_x], weights=weights)

        weights = np.abs(force_y_combined[zero_crossings_y])
        weighted_iep_cop_y = np.average(cop_y_combined[zero_crossings_y], weights=weights)

        # Compute rambling/trembling using COMBINED signals
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
            "Rambling_X": rambling_x.mean(),
            "Rambling_Y": rambling_y.mean(),
            "Trembling_X": trembling_x.mean(),
            "Trembling_Y": trembling_y.mean(),
            "Weighted_IEP_COP_X": weighted_iep_cop_x,
            "Weighted_IEP_COP_Y": weighted_iep_cop_y
        }

        cop_power_summary = {
            "Subject_ID": subject_num,
            "Condition": condition_num,
            "Trial": trial_in_condition_num,
            "Study": study,
            **{f'COP_X_{band}_Power': val for band, val in cop_x_power_bands.items()},
            **{f'COP_Y_{band}_Power': val for band, val in cop_y_power_bands.items()}
        }

        # # Plot COP, rambling, and trembling signals for X and Y axes
        # plot_cop_rambling_trembling(
        #     cop_signal=cop_x_combined,
        #     rambling_signal=rambling_x,
        #     trembling_signal=trembling_x,
        #     title="COP, Rambling, and Trembling Signals (X-Axis)"
        # )
        # plot_cop_rambling_trembling(
        #     cop_signal=cop_y_combined,
        #     rambling_signal=rambling_y,
        #     trembling_signal=trembling_y,
        #     title="COP, Rambling, and Trembling Signals (Y-Axis)"
        # )

        # # Plot combined force signals
        # plt.plot(force_x_combined, label="Force X Combined")
        # plt.plot(force_y_combined, label="Force Y Combined")
        # plt.legend()
        # plt.show()

        # # Analyze the frequency spectrum of the combined COP signals
        # analyze_cop_frequency_spectrum(cop_x_combined, sampling_rate)
        # analyze_cop_frequency_spectrum(cop_y_combined, sampling_rate)

        # # Plot PSDs for COP, rambling, and trembling signals
        # freq_cop, psd_cop = compute_psd(cop_x_combined, sampling_rate)
        # freq_rambling, psd_rambling = compute_psd(rambling_x, sampling_rate)
        # freq_trembling, psd_trembling = compute_psd(trembling_x, sampling_rate)

        # plt.figure(figsize=(10, 6))
        # plt.plot(freq_cop, psd_cop, label="COP PSD")
        # plt.plot(freq_rambling, psd_rambling, label="Rambling PSD")
        # plt.plot(freq_trembling, psd_trembling, label="Trembling PSD")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Power Spectral Density")
        # plt.legend()
        # plt.grid()
        # plt.show()

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

def compute_mean_sd_trembling_rambling(results_df):
    """Compute the mean and standard deviation of trembling and rambling components, separated by cohort."""
    mean_sd_data = {
        "Cohort": [],
        "Component": [],
        "Direction": [],
        "Mean": [],
        "Standard_Deviation": [],
        "Participants": []  # Add column for number of participants
    }

    # Determine cohorts based on Subject_ID
    results_df['Cohort'] = results_df['Subject_ID'].apply(lambda x: f"{x // 100 * 100}s")

    for cohort, cohort_df in results_df.groupby('Cohort'):
        participant_count = cohort_df['Subject_ID'].nunique()  # Count unique participants
        for component in ["Rambling", "Trembling"]:
            for direction in ["X", "Y"]:
                column_name = f"{component}_{direction}"
                if column_name in cohort_df.columns:
                    mean_sd_data["Cohort"].append(cohort)
                    mean_sd_data["Component"].append(component)
                    mean_sd_data["Direction"].append(direction)
                    mean_sd_data["Mean"].append(cohort_df[column_name].mean())
                    mean_sd_data["Standard_Deviation"].append(cohort_df[column_name].std())
                    mean_sd_data["Participants"].append(participant_count)  # Add participant count

    return pd.DataFrame(mean_sd_data)

def export_mean_sd_to_excel(mean_sd_df, output_dir="./mean_sd_summary"):
    """Export the mean and standard deviation of trembling and rambling components to an Excel file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "PCD_mean_sd_trembling_rambling.xlsx")
    mean_sd_df.to_excel(output_file, index=False)
    print(f"Mean/SD of trembling and rambling components exported to: {output_file}")

# def plot_cop_rambling_trembling(cop_signal, rambling_signal, trembling_signal, title="COP, Rambling, and Trembling Signals"):
#     """Plot the time series of COP, rambling, and trembling signals."""
#     time = np.arange(len(cop_signal))  # Assuming the sampling rate is constant and time is in samples
#     plt.figure(figsize=(12, 6))
#     plt.plot(cop_signal, label="COP Signal", color="blue")
#     plt.plot(rambling_signal, label="Rambling Component", color="green")
#     plt.plot(trembling_signal, label="Trembling Component", color="red")
#     plt.xlabel("Time (samples)")
#     plt.ylabel("Signal Amplitude")
#     plt.legend()
#     plt.grid()
#     plt.show()

# def analyze_cop_frequency_spectrum(cop_signal, sampling_rate):
#     """Analyze and plot the frequency spectrum of the COP signal."""
#     freq, psd = compute_psd(cop_signal, sampling_rate)
#     plt.figure(figsize=(10, 6))
#     plt.plot(freq, psd, color='blue', label='COP Signal PSD')
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Power Spectral Density")
#     plt.title("Frequency Spectrum of COP Signal")
#     plt.grid()
#     plt.legend()
#     plt.show()

if __name__ == "__main__":
    print("Starting analysis...")
    
    results_df, cop_power_summary_df = analyze_all_subjects(BASE_PATH)
    
    if not results_df.empty:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(OUTPUT_PATH, f'pcd_results_{timestamp_str}.xlsx')
        
        results_df.to_excel(output_file_path, index=False)
        
        print(f"Analysis completed successfully. Results saved to {output_file_path}.")
        
        # Compute and export mean/SD of trembling and rambling
        mean_sd_df = compute_mean_sd_trembling_rambling(results_df)
        export_mean_sd_to_excel(mean_sd_df, OUTPUT_PATH)
        

        
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