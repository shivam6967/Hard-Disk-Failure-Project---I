"""
Hard Disk Failure Prediction and Detailed Report
================================================
This script predicts hard disk failures using machine learning models and generates
a detailed report on each failed or failing disk.

Directory Structure:
- dataset2/ - Contains all CSV data files
- models/ - Contains this script

Usage:
- Update the DATA_DIR variable at the top of the script to point to your dataset directory
- Run this script from any location
"""

# Set the path to your dataset directory here
# Try different paths if you're getting file not found errors
DATA_DIR = "../dataset2"  # Default directory name
# Other possible paths to try:
# DATA_DIR = "./dataset2"  # Relative path from current directory
# DATA_DIR = "../dataset2"  # Go up one level and then to dataset2
# DATA_DIR = "C:/path/to/your/dataset2"  # Absolute path (Windows)
# DATA_DIR = "/path/to/your/dataset2"  # Absolute path (Unix/Linux/Mac)


# Install required packages
# !pip install pandas numpy scikit-learn xgboost imbalanced-learn

# Load Packages
import pandas as pd
import numpy as np
from time import time
import os
import sys
import datetime
import json
from tabulate import tabulate

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
from collections import Counter
from sklearn import metrics 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Uncomment if you want to use these
# from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE

# Reading data from a combined file with both good and failed drives
def split_train_val_test_data(root=DATA_DIR, drive_file="ST8000DM002_last_10_day_all_q_raw.csv",  
                          ignore_cols=["date", "serial_number", "model", "capacity_bytes", "failure"], 
                          resample_data=False, smote_data=False):
    """
    Load and split data into training and testing sets
    
    Parameters:
    -----------
    root : str
        Root directory where data is stored
    drive_file : str
        Name of the CSV file containing the data
    ignore_cols : list
        Columns to ignore in the analysis
    resample_data : bool
        Whether to resample the minority class
    smote_data : bool
        Whether to apply SMOTE for oversampling
        
    Returns:
    --------
    tuple
        Training and testing data splits
    """
    # Check if file exists
    file_path = os.path.join(root, drive_file)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        print("Available files:", os.listdir(root))
        return None, None, None, None
        
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {len(df)} rows and {len(df.columns)} columns.")
        
        # Check if required columns exist
        for col in ignore_cols:
            if col not in df.columns:
                if col == "failure":
                    print(f"Error: Required column '{col}' not found in the dataset.")
                    return None, None, None, None
                else:
                    # If it's not a critical column, remove it from ignore_cols
                    ignore_cols.remove(col)
                    print(f"Warning: Column '{col}' not found in the dataset.")
                
        # Print column names
        print("Columns in dataset:", df.columns.tolist())
        
        # Print class distribution
        print("Class distribution:", Counter(df['failure']))
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None
    
    # Filter good and failed drives
    df_good = df.loc[df['failure'] == 0]
    df_bad = df.loc[df['failure'] == 1]
    
    # Check if we have both good and bad drives
    if len(df_good) == 0 or len(df_bad) == 0:
        print("Error: Dataset doesn't contain both good and failed drives.")
        print(f"Good drives: {len(df_good)}, Failed drives: {len(df_bad)}")
        return None, None, None, None
     
    # Sort by date if available
    if "date" in df.columns:
        df_good = df_good.sort_values(["date"])
        df_bad = df_bad.sort_values(["date"])

    good_y = df_good["failure"]
    bad_y = df_bad["failure"]

    # Split into train (80%) and test (20%)
    X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(
        df_good, good_y, train_size=0.8, shuffle=False)
    X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
        df_bad, bad_y, train_size=0.8, shuffle=False)
        
    if resample_data:
        print(f"Resampling minority class. Before: {len(X_train_bad)} samples")
        X_train_bad = resample(df_bad, replace=True, n_samples=len(X_train_good), random_state=1)
        if "date" in df.columns:
            X_train_bad = X_train_bad.sort_values(["date"])
        print(f"After resampling: {len(X_train_bad)} samples")

    y_train_bad = X_train_bad["failure"]

    # Combine the datasets
    X_train = pd.concat([X_train_good, X_train_bad], axis=0)
    y_train = pd.concat([y_train_good, y_train_bad], axis=0)
    X_test = pd.concat([X_test_good, X_test_bad], axis=0)
    y_test = pd.concat([y_test_good, y_test_bad], axis=0)

    # Save a reference to the original dataframe (for detailed reports)
    original_df = df.copy()

    # Remove ignored columns
    valid_ignore_cols = [col for col in ignore_cols if col in X_train.columns]
    if valid_ignore_cols:
        X_train = X_train.drop(columns=valid_ignore_cols)
        X_test = X_test.drop(columns=valid_ignore_cols)

    # Apply SMOTE if requested
    if smote_data:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"Applied SMOTE. New training set shape: {X_train.shape}")
        except ImportError:
            print("Warning: imblearn not installed. SMOTE not applied.")
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
            print("Continuing with original data.")

    print(f"Final training set: {X_train.shape}, positive samples: {sum(y_train)}")
    print(f"Final test set: {X_test.shape}, positive samples: {sum(y_test)}")

    return X_train, X_test, y_train, y_test, original_df

def sort_data_by_date(file_path):
    """Sort data by date"""
    try:
        df = pd.read_csv(file_path)
        if "date" not in df.columns:
            print("Warning: 'date' column not found in the dataset.")
            return df
        sorted_df = df.sort_values(["date"])
        return sorted_df
    except Exception as e:
        print(f"Error sorting data: {e}")
        return None

def random_tune_randomforest():
    """Configure RandomForest hyperparameter tuning"""
    rf = RandomForestClassifier(random_state=1)
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    #Entropy calculations
    criterion = ["gini", "entropy"]
    
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap,
        'criterion': criterion
    }
    
    rf_random = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=random_grid, 
        n_iter=100, 
        cv=3, 
        verbose=2, 
        random_state=1, 
        n_jobs=-1, 
        scoring=["f1", "accuracy"], 
        refit="f1"
    )
    
    return rf_random

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Calculate precision, recall, and F1 for the positive class
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision (failure detection): {precision:.4f}")
    print(f"Recall (failure detection): {recall:.4f}")
    print(f"F1 Score (failure detection): {f1:.4f}")
    
    return acc, report, conf_matrix, y_pred

def generate_failure_report(df, y_pred=None, threshold=0.7, file_name="failure_report.html"):
    """
    Generate a detailed report on failed and failing disks
    
    Parameters:
    -----------
    df : pandas DataFrame
        The original dataset with all disk information
    y_pred : numpy array or None
        Model predictions if available
    threshold : float
        Probability threshold for failure prediction if applicable
    file_name : str
        Name of the output HTML report file
    """
    # Create a copy to avoid modifying the original
    report_df = df.copy()
    
    # Add a column for prediction if available
    if y_pred is not None and len(y_pred) == len(report_df):
        report_df['predicted_failure'] = y_pred
    
    # Filter only the failed drives
    failed_drives = report_df[report_df['failure'] == 1]
    
    # If no failed drives, check for predicted failures
    if len(failed_drives) == 0 and 'predicted_failure' in report_df.columns:
        failed_drives = report_df[report_df['predicted_failure'] == 1]
        if len(failed_drives) == 0:
            print("No actual or predicted disk failures found in the dataset.")
            return
    elif len(failed_drives) == 0:
        print("No disk failures found in the dataset.")
        return
    
    # Generate a timestamp for the report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Disk Failure Detailed Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .report-header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
            .disk-section {{ margin-top: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .failure {{ color: #cc0000; font-weight: bold; }}
            .warning {{ color: #ff9900; font-weight: bold; }}
            .normal {{ color: #008800; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .summary {{ margin-top: 20px; background-color: #e9f7fe; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <h1>Disk Failure Detailed Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Total drives analyzed:</strong> {len(report_df)}</p>
            <p><strong>Failed drives detected:</strong> {len(failed_drives)}</p>
        </div>
        
        <div class="summary">
            <h2>Summary of Failed Disks</h2>
            <table>
                <tr>
                    <th>Serial Number</th>
                    <th>Model</th>
                    <th>Capacity</th>
                    <th>Date of Failure</th>
                    <th>Status</th>
                </tr>
    """
    
    # Add summary rows for each failed disk
    for _, drive in failed_drives.iterrows():
        serial = drive.get('serial_number', 'Unknown')
        model = drive.get('model', 'Unknown')
        capacity = drive.get('capacity_bytes', 'Unknown')
        if capacity != 'Unknown':
            # Convert bytes to TB if available
            capacity = f"{capacity / (1024**4):.2f} TB"
        date = drive.get('date', 'Unknown')
        
        status = "Confirmed Failure" if drive['failure'] == 1 else "Predicted Failure"
        status_class = "failure" if drive['failure'] == 1 else "warning"
        
        html_content += f"""
                <tr>
                    <td>{serial}</td>
                    <td>{model}</td>
                    <td>{capacity}</td>
                    <td>{date}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Add detailed section for each failed disk
    html_content += """
        <h2>Detailed Information for Each Failed Disk</h2>
    """
    
    # Define SMART attributes that are strong indicators of failure
    critical_attributes = [
        'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw',
        'smart_5_normalized', 'smart_187_normalized', 'smart_188_normalized', 'smart_197_normalized', 'smart_198_normalized'
    ]
    
    attribute_descriptions = {
        'smart_1': 'Read Error Rate',
        'smart_3': 'Spin-Up Time',
        'smart_4': 'Start/Stop Count',
        'smart_5': 'Reallocated Sectors Count',
        'smart_7': 'Seek Error Rate',
        'smart_9': 'Power-On Hours',
        'smart_10': 'Spin Retry Count',
        'smart_12': 'Power Cycle Count',
        'smart_183': 'Runtime Bad Block',
        'smart_184': 'End-to-End Error',
        'smart_187': 'Reported Uncorrectable Errors',
        'smart_188': 'Command Timeout',
        'smart_189': 'High Fly Writes',
        'smart_190': 'Temperature Difference',
        'smart_191': 'G-Sense Error Rate',
        'smart_192': 'Power-off Retract Count',
        'smart_193': 'Load Cycle Count',
        'smart_194': 'Temperature',
        'smart_195': 'Hardware ECC Recovered',
        'smart_196': 'Reallocation Event Count',
        'smart_197': 'Current Pending Sector Count',
        'smart_198': 'Offline Uncorrectable Sector Count',
        'smart_199': 'UltraDMA CRC Error Count',
        'smart_200': 'Write Error Rate',
        'smart_240': 'Head Flying Hours',
        'smart_241': 'Total LBAs Written',
        'smart_242': 'Total LBAs Read'
    }
    
    # Add detailed section for each failed disk
    for i, (_, drive) in enumerate(failed_drives.iterrows(), 1):
        serial = drive.get('serial_number', 'Unknown')
        model = drive.get('model', 'Unknown')
        capacity = drive.get('capacity_bytes', 'Unknown')
        if capacity != 'Unknown':
            capacity = f"{capacity / (1024**4):.2f} TB"
        date = drive.get('date', 'Unknown')
        
        html_content += f"""
        <div class="disk-section">
            <h3>Failed Disk #{i}: {serial}</h3>
            <p><strong>Model:</strong> {model}</p>
            <p><strong>Capacity:</strong> {capacity}</p>
            <p><strong>Date of Failure:</strong> {date}</p>
            <p><strong>Status:</strong> <span class="failure">{'Confirmed Failure' if drive['failure'] == 1 else 'Predicted Failure'}</span></p>
            
            <h4>SMART Attributes</h4>
            <table>
                <tr>
                    <th>Attribute Name</th>
                    <th>Raw Value</th>
                    <th>Normalized Value</th>
                    <th>Status</th>
                </tr>
        """
        
        # Add all SMART attributes in the dataset
        smart_cols = [col for col in drive.index if col.startswith('smart_')]
        for col in smart_cols:
            # Skip failure column
            if col == 'failure':
                continue
                
            # Extract attribute number and type (raw vs normalized)
            parts = col.split('_')
            if len(parts) >= 3:
                attr_num = parts[1]
                attr_type = parts[2]
                
                # Only display if we have both raw and normalized values
                raw_col = f'smart_{attr_num}_raw'
                norm_col = f'smart_{attr_num}_normalized'
                
                if raw_col in drive and norm_col in drive and col == raw_col:
                    raw_val = drive[raw_col]
                    norm_val = drive[norm_col]
                    
                    # Determine if this attribute indicates failure
                    status_class = "normal"
                    status_text = "Normal"
                    
                    # Check if this is a critical attribute with abnormal values
                    if raw_col in critical_attributes or norm_col in critical_attributes:
                        if attr_num in ['5', '187', '188', '197', '198'] and raw_val > 0:
                            status_class = "failure"
                            status_text = "Critical"
                        elif attr_num in ['5', '187', '188', '197', '198'] and norm_val < 100:
                            status_class = "warning"
                            status_text = "Warning"
                    
                    # Get attribute description
                    attr_name = attribute_descriptions.get(f'smart_{attr_num}', f'Attribute {attr_num}')
                    
                    html_content += f"""
                    <tr>
                        <td>{attr_name}</td>
                        <td>{raw_val}</td>
                        <td>{norm_val}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
                    """
        
        html_content += """
            </table>
        </div>
        """
    
    # Close the HTML document
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML to a file
    with open(file_name, 'w') as f:
        f.write(html_content)
    
    print(f"Detailed failure report generated: {file_name}")
    
    # Also generate a plain text version
    generate_plain_text_report(failed_drives, file_name.replace('.html', '.txt'))

def generate_plain_text_report(failed_drives, file_name="failure_report.txt"):
    """Generate a plain text version of the failure report"""
    with open(file_name, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DISK FAILURE DETAILED REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Failed drives detected: {len(failed_drives)}\n\n")
        
        f.write("SUMMARY OF FAILED DISKS\n")
        f.write("-" * 80 + "\n")
        
        # Create a summary table
        summary_data = []
        for _, drive in failed_drives.iterrows():
            serial = drive.get('serial_number', 'Unknown')
            model = drive.get('model', 'Unknown')
            capacity = drive.get('capacity_bytes', 'Unknown')
            if capacity != 'Unknown':
                capacity = f"{capacity / (1024**4):.2f} TB"
            date = drive.get('date', 'Unknown')
            status = "Confirmed Failure" if drive['failure'] == 1 else "Predicted Failure"
            
            summary_data.append([serial, model, capacity, date, status])
        
        # Use tabulate for better formatting
        try:
            summary_table = tabulate(
                summary_data,
                headers=["Serial Number", "Model", "Capacity", "Date", "Status"],
                tablefmt="grid"
            )
            f.write(summary_table + "\n\n")
        except:
            # Fallback if tabulate is not available
            f.write("Serial Number\tModel\tCapacity\tDate\tStatus\n")
            for row in summary_data:
                f.write("\t".join(str(col) for col in row) + "\n")
            f.write("\n\n")
        
        # Define SMART attributes that are strong indicators of failure
        critical_attributes = [
            'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw'
        ]
        
        attribute_descriptions = {
            'smart_1': 'Read Error Rate',
            'smart_5': 'Reallocated Sectors Count',
            'smart_9': 'Power-On Hours',
            'smart_187': 'Reported Uncorrectable Errors',
            'smart_188': 'Command Timeout',
            'smart_197': 'Current Pending Sector Count',
            'smart_198': 'Offline Uncorrectable Sector Count',
        }
        
        # Add detailed section for each failed disk
        for i, (_, drive) in enumerate(failed_drives.iterrows(), 1):
            f.write(f"FAILED DISK #{i}\n")
            f.write("-" * 80 + "\n")
            
            serial = drive.get('serial_number', 'Unknown')
            model = drive.get('model', 'Unknown')
            capacity = drive.get('capacity_bytes', 'Unknown')
            if capacity != 'Unknown':
                capacity = f"{capacity / (1024**4):.2f} TB"
            date = drive.get('date', 'Unknown')
            
            f.write(f"Serial Number: {serial}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Capacity: {capacity}\n")
            f.write(f"Date of Failure: {date}\n")
            f.write(f"Status: {'Confirmed Failure' if drive['failure'] == 1 else 'Predicted Failure'}\n\n")
            
            f.write("CRITICAL SMART ATTRIBUTES:\n")
            
            # Add critical SMART attributes 
            smart_data = []
            for attr_num in ['5', '187', '188', '197', '198']:
                raw_col = f'smart_{attr_num}_raw'
                norm_col = f'smart_{attr_num}_normalized'
                
                if raw_col in drive and norm_col in drive:
                    raw_val = drive[raw_col]
                    norm_val = drive[norm_col]
                    attr_name = attribute_descriptions.get(f'smart_{attr_num}', f'Attribute {attr_num}')
                    
                    status = "Normal"
                    if raw_val > 0:
                        status = "CRITICAL"
                    elif norm_val < 100:
                        status = "Warning"
                        
                    smart_data.append([attr_name, raw_val, norm_val, status])
            
            try:
                smart_table = tabulate(
                    smart_data,
                    headers=["Attribute", "Raw Value", "Normalized Value", "Status"],
                    tablefmt="grid"
                )
                f.write(smart_table + "\n\n")
            except:
                f.write("Attribute\tRaw Value\tNormalized Value\tStatus\n")
                for row in smart_data:
                    f.write("\t".join(str(col) for col in row) + "\n")
                f.write("\n\n")
            
            f.write("OTHER SMART ATTRIBUTES:\n")
            other_smart_data = []
            
            # Add other important SMART attributes
            for col in ['smart_1_raw', 'smart_9_raw', 'smart_194_raw', 'smart_241_raw', 'smart_242_raw']:
                if col in drive:
                    attr_num = col.split('_')[1]
                    raw_val = drive[col]
                    norm_col = f'smart_{attr_num}_normalized'
                    norm_val = drive[norm_col] if norm_col in drive else "N/A"
                    attr_name = attribute_descriptions.get(f'smart_{attr_num}', f'Attribute {attr_num}')
                    
                    other_smart_data.append([attr_name, raw_val, norm_val])
            
            try:
                other_smart_table = tabulate(
                    other_smart_data,
                    headers=["Attribute", "Raw Value", "Normalized Value"],
                    tablefmt="grid"
                )
                f.write(other_smart_table + "\n\n")
            except:
                f.write("Attribute\tRaw Value\tNormalized Value\n")
                for row in other_smart_data:
                    f.write("\t".join(str(col) for col in row) + "\n")
                f.write("\n\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"Plain text failure report generated: {file_name}")

def run_model(model_type="RF", file_path="ST8000DM002_last_10_day_all_q_raw.csv", 
              root="./", tune_model=False, resample_data=True, generate_report=True):
    """
    Run a specific model on the given data and generate a detailed report
    
    Parameters:
    -----------
    model_type : str
        Type of model to run ('RF' for RandomForest)
    file_path : str
        Path to the data file
    root : str
        Root directory containing the data
    tune_model : bool
        Whether to tune the model hyperparameters
    resample_data : bool
        Whether to resample the minority class
    generate_report : bool
        Whether to generate a detailed disk failure report
    """
    print(f"\n{'='*50}")
    print(f"Running {model_type} on {file_path}")
    print(f"{'='*50}")
    
    # Load and split data
    X_train, X_test, y_train, y_test, original_df = split_train_val_test_data(
        root=root, 
        drive_file=file_path, 
        resample_data=resample_data
    )
    
    if X_train is None:
        print("Failed to load data. Exiting.")
        return
    
    if tune_model:
        if model_type == "RF":
            model = random_tune_randomforest()
            print("Using RandomForest with hyperparameter tuning")
        else:
            print(f"Model type {model_type} with tuning not supported. Using RandomForest.")
            model = random_tune_randomforest()
    else:
        if model_type == "RF":
            model = RandomForestClassifier(
                n_estimators=2000, 
                min_samples_split=5, 
                min_samples_leaf=4,
                max_features='auto', 
                max_depth=40, 
                criterion='entropy',
                bootstrap=True,
                random_state=1
            )
            print("Using RandomForest with preset parameters")
        else:
            print(f"Model type {model_type} not supported. Using RandomForest.")
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=1
            )
    
    # Train model
    print("\nTraining model...")
    start = time()
    model.fit(X_train, y_train)
    end = time()
    print(f"Training completed in {(end - start)/60:.2f} minutes")
    
    # If model was tuned, print best parameters
    if hasattr(model, 'best_params_'):
        print("\nBest parameters:", model.best_params_)
    
    # Evaluate model
    print("\nEvaluating model on test set:")
    _, _, _, y_pred = evaluate_model(model, X_test, y_test)
    
    # Generate detailed report for failed disks
    if generate_report:
        report_file = f"disk_failure_report_{file_path.replace('.csv', '')}.html"
        print(f"\nGenerating detailed disk failure report: {report_file}")
        generate_failure_report(original_df, y_pred, file_name=report_file)
    
    return model, original_df, y_pred

def run_on_all_files(data_dir=DATA_DIR, model_type="RF", tune_model=False, resample_data=True, generate_report=True):
    """Run the model on all available data files and generate reports"""
    # List all CSV files in the directory
    try:
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not all_files:
            print(f"No CSV files found in {data_dir}")
            return
        
        print(f"Found {len(all_files)} CSV files: {all_files}")
        
        # Run model on each file
        results = {}
        all_failed_disks = []
        
        for file in all_files:
            if "_last_" in file:  # Only process files with the expected naming pattern
                try:
                    model, df, y_pred = run_model(
                        model_type=model_type,
                        file_path=file,
                        root=data_dir,
                        tune_model=tune_model,
                        resample_data=resample_data,
                        generate_report=generate_report
                    )
                    
                    # Store results
                    results[file] = {
                        "model": model,
                        "dataframe": df,
                        "predictions": y_pred
                    }
                    
                    # Collect failed disks for consolidated report
                    failed_df = df[df['failure'] == 1].copy()
                    failed_df['source_file'] = file
                    all_failed_disks.append(failed_df)
                    
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        # Generate consolidated report with all failed disks
        if generate_report and all_failed_disks:
            consolidated_df = pd.concat(all_failed_disks, ignore_index=True)
            print(f"\nGenerating consolidated report with all {len(consolidated_df)} failed disks")
            generate_failure_report(consolidated_df, file_name="consolidated_disk_failure_report.html")
        
        return results
    
    except Exception as e:
        print(f"Error listing files in directory {data_dir}: {e}")
        return None

def analyze_failed_disks(data_dir=DATA_DIR, output_file="failed_disks_detailed_analysis.txt"):
    """
    Analyze all CSV files to extract comprehensive information about each failed disk
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the CSV files
    output_file : str
        File to save the detailed analysis
    """
    try:
        # Get list of all CSV files
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not all_files:
            print(f"No CSV files found in {data_dir}")
            return
        
        print(f"Found {len(all_files)} CSV files")
        
        # List to store all failed disks across all files
        all_failed_disks = []
        all_drive_types = set()
        
        # Process each file
        for file in all_files:
            file_path = os.path.join(data_dir, file)
            try:
                print(f"Processing {file}...")
                df = pd.read_csv(file_path)
                
                # Check if this file contains the necessary columns
                if 'failure' not in df.columns:
                    print(f"  Skipping {file} - no 'failure' column found")
                    continue
                
                # Extract failed drives
                failed_df = df[df['failure'] == 1].copy()
                
                # Add source file information
                failed_df['source_file'] = file
                
                # Extract drive type from filename (e.g., ST8000DM002)
                drive_type = file.split('_')[0] if '_' in file else 'Unknown'
                failed_df['drive_type'] = drive_type
                all_drive_types.add(drive_type)
                
                print(f"  Found {len(failed_df)} failed drives in {file}")
                all_failed_disks.append(failed_df)
                
            except Exception as e:
                print(f"  Error processing {file}: {e}")
        
        # Combine all failed disks into one dataframe
        if not all_failed_disks:
            print("No failed disks found in any file.")
            return
            
        combined_df = pd.concat(all_failed_disks, ignore_index=True)
        print(f"Total failed disks found: {len(combined_df)}")
        
        # Generate a detailed report
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DETAILED ANALYSIS OF ALL FAILED DISKS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Failed Disks: {len(combined_df)}\n")
            f.write(f"Drive Types: {', '.join(sorted(all_drive_types))}\n\n")
            
            # Group by drive type for summary
            f.write("SUMMARY BY DRIVE TYPE\n")
            f.write("-" * 80 + "\n")
            
            for drive_type in sorted(all_drive_types):
                count = len(combined_df[combined_df['drive_type'] == drive_type])
                f.write(f"{drive_type}: {count} failed disks\n")
            
            f.write("\n")
            
            # Get serial numbers if available
            if 'serial_number' in combined_df.columns:
                serials = combined_df['serial_number'].unique()
                f.write(f"Unique Failed Disks by Serial Number: {len(serials)}\n\n")
            
            # Write detailed information for each failed disk
            f.write("INDIVIDUAL FAILED DISK DETAILS\n")
            f.write("=" * 80 + "\n\n")
            
            # Iterate through each failed disk
            for i, (_, disk) in enumerate(combined_df.iterrows(), 1):
                f.write(f"FAILED DISK #{i}\n")
                f.write("-" * 80 + "\n")
                
                # Basic information
                f.write(f"Source File: {disk['source_file']}\n")
                f.write(f"Drive Type: {disk['drive_type']}\n")
                
                # Add all available identification information
                for col in ['serial_number', 'model', 'capacity_bytes', 'date']:
                    if col in disk:
                        value = disk[col]
                        if col == 'capacity_bytes' and isinstance(value, (int, float)):
                            value = f"{value / (1024**4):.2f} TB"
                        f.write(f"{col.replace('_', ' ').title()}: {value}\n")
                
                f.write("\nSMART Attributes:\n")
                
                # Find all SMART attributes
                smart_cols = [col for col in disk.index if col.startswith('smart_')]
                
                # Group by attribute number
                smart_by_number = {}
                for col in smart_cols:
                    parts = col.split('_')
                    if len(parts) >= 2:
                        attr_num = parts[1]
                        attr_type = parts[2] if len(parts) > 2 else "unknown"
                        
                        if attr_num not in smart_by_number:
                            smart_by_number[attr_num] = {}
                        
                        smart_by_number[attr_num][attr_type] = disk[col]
                
                # Define critical attributes
                critical_attrs = ['5', '187', '188', '197', '198']
                
                # Write critical attributes first
                f.write("\nCritical SMART Attributes:\n")
                for attr in critical_attrs:
                    if attr in smart_by_number:
                        attr_info = smart_by_number[attr]
                        raw = attr_info.get('raw', 'N/A')
                        norm = attr_info.get('normalized', 'N/A')
                        
                        # Get attribute name
                        attr_name = {
                            '5': 'Reallocated Sectors Count',
                            '187': 'Reported Uncorrectable Errors',
                            '188': 'Command Timeout',
                            '197': 'Current Pending Sector Count',
                            '198': 'Offline Uncorrectable Sector Count'
                        }.get(attr, f'Attribute {attr}')
                        
                        f.write(f"  {attr_name} (SMART #{attr}):\n")
                        f.write(f"    Raw Value: {raw}\n")
                        f.write(f"    Normalized Value: {norm}\n")
                
                # Write other common attributes
                f.write("\nOther Important SMART Attributes:\n")
                other_attrs = ['1', '9', '12', '194', '241', '242']
                for attr in other_attrs:
                    if attr in smart_by_number:
                        attr_info = smart_by_number[attr]
                        raw = attr_info.get('raw', 'N/A')
                        norm = attr_info.get('normalized', 'N/A')
                        
                        # Get attribute name
                        attr_name = {
                            '1': 'Read Error Rate',
                            '9': 'Power-On Hours',
                            '12': 'Power Cycle Count',
                            '194': 'Temperature',
                            '241': 'Total LBAs Written',
                            '242': 'Total LBAs Read'
                        }.get(attr, f'Attribute {attr}')
                        
                        f.write(f"  {attr_name} (SMART #{attr}):\n")
                        f.write(f"    Raw Value: {raw}\n")
                        f.write(f"    Normalized Value: {norm}\n")
                
                # Additional attributes (uncomment if needed)
                # f.write("\nAll SMART Attributes:\n")
                # for attr, attr_info in sorted(smart_by_number.items()):
                #     if attr not in critical_attrs and attr not in other_attrs:
                #         raw = attr_info.get('raw', 'N/A')
                #         norm = attr_info.get('normalized', 'N/A')
                #         f.write(f"  Attribute {attr}:\n")
                #         f.write(f"    Raw Value: {raw}\n")
                #         f.write(f"    Normalized Value: {norm}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
            
        print(f"Detailed analysis of all failed disks saved to {output_file}")
        
        # Also generate HTML report
        html_output = output_file.replace('.txt', '.html')
        generate_failure_report(combined_df, file_name=html_output)
        
        return combined_df
    
    except Exception as e:
        print(f"Error analyzing failed disks: {e}")
        return None

# Main execution
if __name__ == "__main__":
    print("Hard Disk Failure Prediction and Detailed Analysis")
    print("=================================================")
    
    # List all available data files in the dataset directory
    data_dir = DATA_DIR
    print(f"\nListing available data files in {data_dir}:")
    try:
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"Found {len(all_files)} CSV files: {all_files}")
    except Exception as e:
        print(f"Error listing files in {data_dir}: {e}")
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Make sure the dataset directory exists")
        print("2. Update the DATA_DIR variable at the top of the script")
        print("3. Try these alternative paths:")
        print("   - './dataset2'")
        print("   - '../dataset2'")
        print("   - The absolute path to your dataset directory")
        print("\nCurrent working directory:", os.getcwd())
        
        # Try to list the current directory to help with debugging
        try:
            print("\nFiles in current directory:", os.listdir("./"))
        except Exception as e:
            print("Could not list current directory:", e)
    
    # Options menu
    print("\nOPTIONS:")
    print("1: Run predictive model and generate report for a specific file")
    print("2: Run predictive model on all files")
    print("3: Generate detailed report of all failed disks (no modeling)")
    print("4: Exit")
    
    user_input = input("\nSelect an option [1-4]: ")
    
    if user_input == "1":
        # Run on a specific file
        file_name = input("Enter the file name (e.g., ST8000DM002_last_10_day_all_q_raw.csv): ")
        tune = input("Tune model hyperparameters? (may take longer) [y/N]: ").lower() == 'y'
        
        run_model(
            model_type="RF",
            file_path=file_name,
            root=DATA_DIR,
            tune_model=tune,
            resample_data=True,
            generate_report=True
        )
    elif user_input == "2":
        # Run on all files
        tune = input("Tune model hyperparameters? (may take longer) [y/N]: ").lower() == 'y'
        
        run_on_all_files(
            data_dir=DATA_DIR,
            model_type="RF",
            tune_model=tune,
            resample_data=True,
            generate_report=True
        )
    elif user_input == "3":
        # Generate detailed report only
        output_file = input("Enter output file name [failed_disks_detailed_analysis.txt]: ")
        if not output_file:
            output_file = "failed_disks_detailed_analysis.txt"
            
        analyze_failed_disks(
            data_dir=DATA_DIR,
            output_file=output_file
        )
    else:
        print("Exiting program.")
    
    print("\nAnalysis complete!")