#!/usr/bin/env python3
"""
Hard Disk Drive Failure Prediction

This script analyzes hard disk drive SMART data to predict drive failures.
"""

import pandas as pd
import numpy as np 
from time import time
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import metrics 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest

# Reading data from a combined file with both good and failed drives
def split_train_val_test_data(root="../dataset2/",
                              drive_file="ST12000NM0007_last_10_day_all_q_raw.csv",  
                              ignore_cols=["date", "serial_number", "model", "capacity_bytes", "failure"], 
                              resample_data=False, smote_data=False):
    """
    Split the data into training, validation, and test sets.
    
    Parameters:
    - root: Path to the dataset directory
    - drive_file: Name of the dataset file
    - ignore_cols: Columns to ignore during training
    - resample_data: Whether to resample data to balance classes
    - smote_data: Whether to apply SMOTE for synthetic minority samples
    
    Returns:
    Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    # Check if file exists
    file_path = os.path.join(root, drive_file)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print(f"Available files in {root}:")
        for file in os.listdir(root):
            print(f"- {file}")
        raise FileNotFoundError(f"Dataset file {drive_file} not found in {root}")

    # Read the dataset
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset: {drive_file}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Print some dataset statistics
    if 'failure' in df.columns:
        print(f"Failure distribution: {df['failure'].value_counts()}")
    else:
        print("Warning: No 'failure' column found in dataset")
        print("Available columns:", df.columns.tolist())
        raise ValueError("Dataset does not contain required 'failure' column")

    # Split into good and bad drives
    df_good = df.loc[df['failure'] == 0]
    df_bad = df.loc[df['failure'] == 1]
    
    print(f"Good drives: {len(df_good)}, Bad drives: {len(df_bad)}")
    
    # Sort by date if date column exists
    if 'date' in df.columns:
        df_good = df_good.sort_values(["date"])
        df_bad = df_bad.sort_values(["date"])
    else:
        print("Warning: 'date' column not found, skipping date-based sorting")
        # We'll still proceed without date sorting

    good_y = df_good["failure"]
    bad_y = df_bad["failure"]

    # If there are very few failed drives, we might need special handling
    if len(df_bad) < 3:
        print(f"Warning: Only {len(df_bad)} failed drives found. Using special handling.")
        # Split good drives
        X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(
            df_good, good_y, train_size=0.8, test_size=0.2, random_state=42)
        
        # If we have at least one failed drive, put it in training
        if len(df_bad) > 0:
            X_train_bad = df_bad
            y_train_bad = bad_y
            X_test_bad = pd.DataFrame(columns=df_bad.columns)
            y_test_bad = pd.Series([], dtype=int)
        else:
            # Create empty DataFrames if no failed drives
            X_train_bad = pd.DataFrame(columns=df_bad.columns)
            y_train_bad = pd.Series([], dtype=int)
            X_test_bad = pd.DataFrame(columns=df_bad.columns)
            y_test_bad = pd.Series([], dtype=int)
        
        # Split train into train and validation (only for good drives since we don't have enough bad drives)
        X_train_good, X_val_good, y_train_good, y_val_good = train_test_split(
            X_train_good, y_train_good, train_size=0.75, test_size=0.25, random_state=42)
        
        X_val_bad = pd.DataFrame(columns=df_bad.columns)
        y_val_bad = pd.Series([], dtype=int)
    else:
        # Original splitting logic
        # Split into train (80%) and test (20%)
        X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(
            df_good, good_y, train_size=0.8, test_size=0.2, random_state=42)
        X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
            df_bad, bad_y, train_size=0.8, test_size=0.2, random_state=42)

        # Split train into train and validation
        # Train(60%), Val(20%), Test(20%)
        X_train_good, X_val_good, y_train_good, y_val_good = train_test_split(
            X_train_good, y_train_good, train_size=0.75, test_size=0.25, random_state=42)
        X_train_bad, X_val_bad, y_train_bad, y_val_bad = train_test_split(
            X_train_bad, y_train_bad, train_size=0.75, test_size=0.25, random_state=42)
    
    # Resample data if requested
    if resample_data and len(X_train_bad) > 0:
        print("Resampling data to balance classes...")
        try:
            # Handle case where X_train_good might be empty
            if len(X_train_good) > 0:
                X_train_bad = resample(X_train_bad, replace=True, n_samples=len(X_train_good), random_state=1)
                y_train_bad = X_train_bad["failure"]
            else:
                print("Warning: Cannot resample, X_train_good is empty")
        except Exception as e:
            print(f"Warning: Error during resampling: {e}")
            print("Proceeding without resampling")

    # Concatenate datasets
    X_train = pd.concat([X_train_good, X_train_bad], axis=0)
    y_train = pd.concat([y_train_good, y_train_bad], axis=0)
    X_val = pd.concat([X_val_good, X_val_bad], axis=0)
    y_val = pd.concat([y_val_good, y_val_bad], axis=0)
    X_test = pd.concat([X_test_good, X_test_bad], axis=0)
    y_test = pd.concat([y_test_good, y_test_bad], axis=0)

    print(f"Final dataset sizes:")
    print(f"Training: {X_train.shape[0]} samples (Failures: {sum(y_train)})")
    print(f"Validation: {X_val.shape[0]} samples (Failures: {sum(y_val)})")
    print(f"Testing: {X_test.shape[0]} samples (Failures: {sum(y_test)})")

    # Drop columns we don't need for training
    for col in ignore_cols:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
        if col in X_val.columns:
            X_val = X_val.drop(columns=[col])
        if col in X_test.columns:
            X_test = X_test.drop(columns=[col])

    # Apply SMOTE if requested
    if smote_data and len(X_train) > 0 and sum(y_train) > 0:
        print("Applying SMOTE to balance classes...")
        try:
            # Only apply SMOTE if we have enough samples
            if len(X_train) >= 10 and sum(y_train) >= 5 and (1 - sum(y_train)/len(y_train)) >= 0.1:
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                print(f"After SMOTE - Training: {X_train.shape[0]} samples (Failures: {sum(y_train)})")
            else:
                print("Not enough samples for SMOTE, skipping")
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
            print("Proceeding without SMOTE")

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def tune_n_estimators(model, X_train, y_train):
    """Find optimal n_estimators for XGBoost model"""
    print("Getting optimal n_estimators...")
    
    # Check if we have enough data to perform CV
    if X_train.shape[0] < 10 or sum(y_train) < 5:
        print("Warning: Not enough data for cross-validation. Using default n_estimators.")
        return model.get_params()['n_estimators']
    
    try:
        import xgboost as xgb
        # 1: Set learning rate and n_estimators
        xgtrain = xgb.DMatrix(X_train, y_train)
        params = model.get_xgb_params()
        num_boost_round = model.get_params()['n_estimators'] 
        metrics = 'auc'
        
        # Adjust nfold based on data size
        if X_train.shape[0] < 50:
            nfold = 3
        else:
            nfold = 5
            
        early_stopping_rounds = 50
        
        cvresult = xgb.cv(params, xgtrain, num_boost_round=num_boost_round, nfold=nfold,
                          metrics=metrics, early_stopping_rounds=early_stopping_rounds)
        
        n_estimators = cvresult.shape[0]
        print("Optimal n_estimators:", n_estimators)
        return n_estimators
    except Exception as e:
        print(f"Error tuning n_estimators: {e}")
        print("Using default n_estimators instead")
        return model.get_params()['n_estimators']


def tune_xgb(model, X_train, y_train):
    """Tune all parameters for XGBoost model"""
    print("Starting XGBoost basic tuning...")
    
    # Check if we have enough data for tuning
    if X_train.shape[0] < 30 or sum(y_train) < 10:
        print("Warning: Not enough data for comprehensive tuning. Using default parameters.")
        return model.get_params()
    
    try:
        # Simple tuning of n_estimators only
        n_estimators = tune_n_estimators(model, X_train, y_train)
        
        # Return tuned parameters
        tuned_params = {
            'n_estimators': n_estimators
        }
        
        print(f"Tuned parameters: {tuned_params}")
        return tuned_params
    except Exception as e:
        print(f"Error during XGBoost tuning: {e}")
        print("Using default parameters instead")
        return model.get_params()


def run(models=[RandomForestClassifier(max_depth=2, random_state=0)], 
        dataset_file="ST12000NM0007_last_10_day_all_q_raw.csv",
        tune_model=False,
        dataset_path="../dataset2/"):
    """
    Run the hard disk failure prediction model
    
    Parameters:
    - models: List of models to train and evaluate
    - dataset_file: Name of the dataset file to use
    - tune_model: Whether to tune the model parameters
    - dataset_path: Path to the dataset directory
    """
    print(f"Running prediction on dataset: {dataset_file}")
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_data(
            root=dataset_path, 
            drive_file=dataset_file, 
            resample_data=True,
            smote_data=False  # Set to True if you want to use SMOTE
        )
        print("Data loaded successfully...\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    tuned_params = {}
    trained_models = {}
    feature_importances_dict = {}
    
    for model in models:  
        model_name = type(model).__name__
        print(f"\n\n * Training {model_name}...")
        
        if model_name == "XGBClassifier" and tune_model:
            try:
                tuned_params = tune_xgb(model, X_train, y_train)
                if isinstance(tuned_params, dict):
                    # Update model with tuned parameters
                    for param, value in tuned_params.items():
                        setattr(model, param, value)
                    print(f"Updated model with tuned parameters: {tuned_params}")
            except Exception as e:
                print(f"Error during model tuning: {e}")
                print("Proceeding with default parameters")

        try:
            start = time()
            model.fit(X_train, y_train)
            end = time()
            print(f"\nTime to train: {(end - start):.2f} seconds")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                # Get feature names and importances as a list of tuples
                feature_importances = [(feature, importance) for feature, importance in zip(X_train.columns, importance)]
                # Sort by importance
                feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
                feature_importances_dict[model_name] = feature_importances
                print("\nTop 10 most important features:")
                for feature, importance in feature_importances[:10]:
                    print(f"{feature}: {importance:.4f}")
            
            # Validation set results
            print("\n- Results on validation set: ")
            try:
                y_val_pred = model.predict(X_val)
                print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
                print("Scores:\n", classification_report(y_val, y_val_pred))
            except Exception as e:
                print(f"Error evaluating on validation set: {e}")
            
            # Test set results
            print("\n- Results on test set: ")
            try:
                y_pred = model.predict(X_test)
                print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                print("Scores:\n", classification_report(y_test, y_pred))
            except Exception as e:
                print(f"Error evaluating on test set: {e}")
                
            trained_models[model_name] = model
                
        except Exception as e:
            print(f"Error training or evaluating model: {e}")
    
    return trained_models, feature_importances_dict, tuned_params


# Entry point of function
if __name__ == "__main__":
    # Path to dataset directory
    dataset_path = "../dataset2/"
    
    # Check command line arguments for dataset path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    # List available dataset files to help user choose
    try:
        print("Available datasets:")
        available_datasets = []
        for file in os.listdir(dataset_path):
            if file.endswith('.csv'):
                available_datasets.append(file)
                print(f"{len(available_datasets)}. {file}")
                
        if not available_datasets:
            print("No CSV files found. Please check your dataset directory.")
            sys.exit(1)
    except Exception as e:
        print(f"Error listing dataset files: {e}")
        print("Please check if the dataset directory exists and contains CSV files.")
        sys.exit(1)
    
    # Let user choose a dataset
    try:
        choice = int(input("\nEnter the number of the dataset you want to use (1-{}): ".format(len(available_datasets))))
        if choice < 1 or choice > len(available_datasets):
            print("Invalid choice. Using the first dataset.")
            choice = 1
    except:
        print("Invalid input. Using the first dataset.")
        choice = 1
    
    dataset_file = available_datasets[choice-1]
    print(f"\nSelected dataset: {dataset_file}")
    
    # Choose model
    print("\nAvailable models:")
    print("1. XGBoost Classifier (recommended for better accuracy)")
    print("2. Random Forest Classifier (faster)")
    print("3. Neural Network (MLP Classifier)")
    
    try:
        model_choice = int(input("Enter the number of the model you want to use (1-3): "))
        if model_choice < 1 or model_choice > 3:
            print("Invalid choice. Using XGBoost Classifier.")
            model_choice = 1
    except:
        print("Invalid input. Using XGBoost Classifier.")
        model_choice = 1
    
    # Configure model based on choice
    if model_choice == 1:
        model = XGBClassifier(
            learning_rate=0.1,
            n_estimators=100,  # Reduced for faster training
            max_depth=3,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1,
            seed=27
        )
        print("Using XGBoost Classifier")
    elif model_choice == 2:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        print("Using Random Forest Classifier")
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            max_iter=200,
            random_state=42
        )
        print("Using Neural Network (MLP Classifier)")
    
    # Ask if user wants to tune model parameters
    try:
        tune_choice = input("\nDo you want to tune model parameters? (yes/no, tuning may take a long time): ").lower()
        tune_model = tune_choice in ['yes', 'y', 'true', '1']
    except:
        print("Invalid input. Proceeding without tuning.")
        tune_model = False
    
    # Run the model
    print(f"\nRunning {type(model).__name__} on {dataset_file}...")
    print(f"Model tuning: {'Enabled' if tune_model else 'Disabled'}")
    
    # Create list with the selected model
    models_list = [model]
    
    # Run the model
    trained_models, feature_importances, tuned_params = run(
        models=models_list,
        dataset_file=dataset_file,
        tune_model=tune_model,
        dataset_path=dataset_path
    )
    
    if trained_models:
        print("\nModel training and evaluation complete!")
        
        # Display a summary of the most important features for each model
        for model_name, importances in feature_importances.items():
            print(f"\nTop 5 most important features for {model_name}:")
            for i, (feature, importance) in enumerate(importances[:5], 1):
                print(f"{i}. {feature}: {importance:.4f}")
    else:
        print("\nModel training failed. Please check the errors above.")