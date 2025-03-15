import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from datetime import datetime
import re

# Define constants
FREQ_BANDS = {'LF': (0, 0.3), 'MF': (0.3, 1), 'HF': (1, 3)}
SAMPLING_RATE = 100  # Hz (from the data file, data_rate is 100Hz)
BASE_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\MFPRL_PCD-Decomposition\PCD_SOT_data"
OUTPUT_PATH = r"C:\Users\ari\Documents\UIUC\MFPRL\app_sci_MFPRL"

# Helper function: Butterworth low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

# Helper function: Compute power spectral density (PSD)
def compute_psd(data, fs):
    n = len(data)
    freq = fftfreq(n, d=1/fs)[:n//2]
    fft_values = np.abs(fft(data))[:n//2]
    power_density = (fft_values ** 2) / n
    return freq, power_density

# Helper function: Integrate PSD over frequency bands
def integrate_power(freq, power, bands):
    power_band = {}
    for band, (low, high) in bands.items():
        band_mask = (freq >= low) & (freq <= high)
        if np.any(band_mask):  # Check if there are any frequencies in this band
            power_band[band] = trapezoid(power[band_mask], freq[band_mask])
        else:
            power_band[band] = 0
    return power_band

def compute_rambling_trembling(cop, force, sampling_rate, cutoff=2.0):
    # Using a simple lowpass filter for rambling; trembling is the difference.
    rambling = butter_lowpass_filter(cop, cutoff=cutoff, fs=sampling_rate, order=2)
    trembling = cop - rambling
    return rambling, trembling

def process_excel_file(file_path, sampling_rate=100):
    """Process an Excel file to extract COP data, calculate rambling/trembling, integrated band power, and add metadata."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Find the row containing 'DP'
        dp_row_index = None
        for i, value in enumerate(df.iloc[:, 0]):
            if isinstance(value, str) and value.strip() == 'DP':
                dp_row_index = i
                break
                
        if dp_row_index is None:
            print(f"No 'DP' row found in file: {file_path}")
            return None
            
        headers = df.iloc[dp_row_index].values
        data_df = df.iloc[dp_row_index + 1:].copy()
        data_df.columns = headers
        data_df = data_df.dropna(how='all')
        
        # Extract COP and force data
        cop_x_left = pd.to_numeric(data_df['FP(L).COFx'], errors='coerce').ffill().bfill().values
        cop_y_left = pd.to_numeric(data_df['FP(L).COFy'], errors='coerce').ffill().bfill().values
        cop_x_right = pd.to_numeric(data_df['FP(R).COFx'], errors='coerce').ffill().bfill().values
        cop_y_right = pd.to_numeric(data_df['FP(R).COFy'], errors='coerce').ffill().bfill().values
        
        force_x_left = pd.to_numeric(data_df['FP(L).Fx'], errors='coerce').ffill().bfill().values
        force_y_left = pd.to_numeric(data_df['FP(L).Fy'], errors='coerce').ffill().bfill().values
        force_x_right = pd.to_numeric(data_df['FP(R).Fx'], errors='coerce').ffill().bfill().values
        force_y_right = pd.to_numeric(data_df['FP(R).Fy'], errors='coerce').ffill().bfill().values
        
        # Compute rambling and trembling components for each direction
        rambling_x_left, trembling_x_left = compute_rambling_trembling(cop_x_left, force_x_left, sampling_rate)
        rambling_y_left, trembling_y_left = compute_rambling_trembling(cop_y_left, force_y_left, sampling_rate)
        rambling_x_right, trembling_x_right = compute_rambling_trembling(cop_x_right, force_x_right, sampling_rate)
        rambling_y_right, trembling_y_right = compute_rambling_trembling(cop_y_right, force_y_right, sampling_rate)
        
        # Combine left and right data (average them)
        cop_x = np.mean([cop_x_left, cop_x_right], axis=0)
        cop_y = np.mean([cop_y_left, cop_y_right], axis=0)
        rambling_x = np.mean([rambling_x_left, rambling_x_right], axis=0)
        rambling_y = np.mean([rambling_y_left, rambling_y_right], axis=0)
        trembling_x = np.mean([trembling_x_left, trembling_x_right], axis=0)
        trembling_y = np.mean([trembling_y_left, trembling_y_right], axis=0)
        
        # Compute PSD and integrated power for the X-axis signals
        freq, psd_rambling_x = compute_psd(rambling_x, sampling_rate)
        _, psd_trembling_x = compute_psd(trembling_x, sampling_rate)
        
        # Integrate power using defined bands
        rambling_x_power = integrate_power(freq, psd_rambling_x, FREQ_BANDS)
        trembling_x_power = integrate_power(freq, psd_trembling_x, FREQ_BANDS)
        
        # Build the time series dictionary and add integrated power for each band.
        time_series = {
            'cop_x': cop_x,
            'cop_y': cop_y,
            'rambling_x': rambling_x,
            'rambling_y': rambling_y,
            'trembling_x': trembling_x,
            'trembling_y': trembling_y,
            'freq_rambling_x': freq,
            'psd_rambling_x': psd_rambling_x,
            'psd_trembling_x': psd_trembling_x
        }
        
        for band in FREQ_BANDS:
            col_key = f'Rambling_X_{band}_Power'
            time_series[col_key] = rambling_x_power.get(band, 0)
            
        for band in FREQ_BANDS:
            col_key = f'Trembling_X_{band}_Power'
            time_series[col_key] = trembling_x_power.get(band, 0)
        
        # Extract metadata from the file name.
        file_name = os.path.basename(file_path)
        # Pattern expects something like: PCD_106_SOT_C3_T3_11102018_03_14_59.xlsx
        pattern = r"PCD_(\d+)_SOT_C(\d+)_T(\d+)_.*\.(xlsx|xls)"
        match = re.match(pattern, file_name)
        if match:
            subject_id = int(match.group(1))
            condition  = int(match.group(2))
            trial_num  = int(match.group(3))
        else:
            print(f"Could not extract metadata from file name: {file_name}")
            subject_id = None
            condition = None
            trial_num = None
        
        time_series['subject_id'] = subject_id
        time_series['condition'] = condition
        time_series['trial'] = trial_num
        
        return time_series
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Main function to analyze all subjects
def analyze_all_subjects(base_path, sampling_rate):
    all_results = []
    raw_data_collection = {}  # To store raw data for validation

    # Iterate over sub-folders in base_path (each sub-folder is a different subject)
    for subject_dir in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_dir)
        if os.path.isdir(subject_path):
            subject_id = subject_dir
            print(f"Processing subject: {subject_id}")

            for file in os.listdir(subject_path):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(subject_path, file)
                    print(f"  Processing file: {file}")

                    # Use the process_excel_file function to get trial_data
                    trial_data = process_excel_file(file_path, sampling_rate)

                    if trial_data is not None:
                        # Store raw data for validation
                        file_key = f"{trial_data['subject_id']}_{trial_data['condition']}_{trial_data['trial']}"
                        raw_data_collection[file_key] = {
                            'subject_id': trial_data['subject_id'],
                            'condition': trial_data['condition'],
                            'trial': trial_data['trial'],
                            'cop_x': trial_data['cop_x'],
                            'cop_y': trial_data['cop_y'],
                            'rambling_x': trial_data['rambling_x'],
                            'rambling_y': trial_data['rambling_y'],
                            'trembling_x': trial_data['trembling_x'],
                            'trembling_y': trial_data['trembling_y']
                        }
                        
                        # Calculate total path length as before
                        path_length = np.sum(np.sqrt(np.diff(trial_data['cop_x'])**2 +
                                                     np.diff(trial_data['cop_y'])**2))
                        # Build result dictionary including the integrated power columns
                        result = {
                            'Subject_ID': trial_data['subject_id'],
                            'Condition': trial_data['condition'],
                            'Trial': trial_data['trial'],
                            'File_Name': os.path.basename(file_path),
                            'Path_Length': path_length
                        }
                        
                        # Add integrated power columns from FREQ_BANDS
                        for band in FREQ_BANDS:
                            key = f'Rambling_X_{band}_Power'
                            result[key] = trial_data.get(key, 0)
                        
                        for band in FREQ_BANDS:
                            key = f'Trembling_X_{band}_Power'
                            result[key] = trial_data.get(key, 0)
                        
                        all_results.append(result)
    return all_results, raw_data_collection

# Function to save results to Excel with multiple sheets for validation
def save_results_to_excel(results, raw_data_collection, output_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_path, f"pcd_analysis_results_{timestamp}.xlsx")
    
    # Create a pandas Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Convert results to DataFrame and save to main sheet
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_excel(writer, sheet_name='Summary_Results', index=False)
            
            # Create pivot table for path length
            pivot_path_length = results_df.pivot_table(
                values='Path_Length', 
                index=['Subject_ID', 'Trial'], 
                columns=['Condition']
            )
            pivot_path_length.to_excel(writer, sheet_name='Pivot_PathLength')
            
            # Create pivot tables for each frequency band if the column exists
            for band in FREQ_BANDS:
                col_name = f'Rambling_X_{band}_Power'
                if col_name in results_df.columns:
                    pivot_rambling_x = results_df.pivot_table(
                        values=col_name,
                        index=['Subject_ID', 'Trial'],
                        columns=['Condition']
                    )
                    pivot_rambling_x.to_excel(writer, sheet_name=f'Pivot_Rambling_X_{band}')
            
            for band in FREQ_BANDS:
                col_name = f'Trembling_X_{band}_Power'
                if col_name in results_df.columns:
                    pivot_trembling_x = results_df.pivot_table(
                        values=col_name,
                        index=['Subject_ID', 'Trial'],
                        columns=['Condition']
                    )
                    pivot_trembling_x.to_excel(writer, sheet_name=f'Pivot_Trembling_X_{band}')
        
        # Save a sample of raw data for validation (first 5 files)
        sample_keys = list(raw_data_collection.keys())[:5]
        for i, key in enumerate(sample_keys):
            data = raw_data_collection[key]
            
            # Create a DataFrame with the raw COP data
            raw_df = pd.DataFrame({
                'COP_X': data['cop_x'],
                'COP_Y': data['cop_y'],
                'Rambling_X': data['rambling_x'],
                'Rambling_Y': data['rambling_y'],
                'Trembling_X': data['trembling_x'],
                'Trembling_Y': data['trembling_y']
            })
            raw_df.to_excel(writer, sheet_name=f'Raw_Data_{i+1}', index=False)
            
            # Create a DataFrame with the frequency data (if available)
            if 'freq_rambling_x' in data and 'psd_rambling_x' in data and 'psd_trembling_x' in data:
                freq_df = pd.DataFrame({
                    'Frequency': data['freq_rambling_x'],
                    'Rambling_X_PSD': data['psd_rambling_x'],
                    'Trembling_X_PSD': data['psd_trembling_x']
                })
                freq_df = freq_df[freq_df['Frequency'] <= 30]  # Limit to relevant frequency range
                freq_df.to_excel(writer, sheet_name=f'Freq_Data_{i+1}', index=False)
            
            # Save metadata
            metadata_df = pd.DataFrame({
                'Property': ['Subject_ID', 'Condition', 'Trial'],
                'Value': [data['subject_id'], data['condition'], data['trial']]
            })
            metadata_df.to_excel(writer, sheet_name=f'Metadata_{i+1}', index=False)
    
    print(f"Results saved to {output_file}")
    return output_file

# Function to visualize results for a specific subject
def visualize_subject_results(results_df, subject_id):
    if results_df is None or results_df.empty:
        print("No results to visualize.")
        return
    
    # Filter for the specific subject
    subject_data = results_df[results_df['Subject_ID'] == subject_id]
    
    if subject_data.empty:
        print(f"No data found for subject {subject_id}")
        return
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Path Length by Condition and Trial
    subject_data.pivot(index='Trial', columns='Condition', values='Path_Length').plot(
        kind='bar', ax=axes[0], title=f'Path Length by Condition and Trial - Subject {subject_id}'
    )
    axes[0].set_ylabel('Path Length (cm)')
    axes[0].grid(True)
    
    # Plot 2: Rambling Power (X-axis)
    for band in FREQ_BANDS:
        subject_data.pivot(index='Trial', columns='Condition', values=f'Rambling_X_{band}_Power').plot(
            ax=axes[1], marker='o', label=f'X-{band}'
        )
    axes[1].set_title(f'Rambling Power (X-axis) - Subject {subject_id}')
    axes[1].set_ylabel('Power')
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot 3: Trembling Power (X-axis)
    for band in FREQ_BANDS:
        subject_data.pivot(index='Trial', columns='Condition', values=f'Trembling_X_{band}_Power').plot(
            ax=axes[2], marker='o', label=f'X-{band}'
        )
    axes[2].set_title(f'Trembling Power (X-axis) - Subject {subject_id}')
    axes[2].set_xlabel('Trial')
    axes[2].set_ylabel('Power')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save the figure (commented out because it makes an image for every subject)
    output_file = os.path.join(OUTPUT_PATH, f"subject_{subject_id}_results.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Run the analysis
    results, raw_data_collection = analyze_all_subjects(BASE_PATH, SAMPLING_RATE)
    
    if results:
        # Save results to Excel with multiple sheets
        output_file = save_results_to_excel(results, raw_data_collection, OUTPUT_PATH)
        
        # Convert to DataFrame for visualization
        results_df = pd.DataFrame(results)
        
        # Visualize results for each subject
        for subject_id in results_df['Subject_ID'].unique():
            visualize_subject_results(results_df, subject_id)
            
        print("Analysis complete!")
