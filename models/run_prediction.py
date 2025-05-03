import os
import sys
from fixed_hdd_failure_prediction import run
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    """
    Main function to run the hard disk failure prediction model
    """
    # Path to dataset directory
    dataset_path = "../dataset2/"
    
    # List available datasets
    print("Available datasets:")
    available_datasets = []
    try:
        for file in os.listdir(dataset_path):
            if file.endswith('.csv'):
                available_datasets.append(file)
                print(f"{len(available_datasets)}. {file}")
    except Exception as e:
        print(f"Error listing datasets: {e}")
        print("Please make sure the dataset2 directory exists and contains CSV files.")
        return
    
    if not available_datasets:
        print("No CSV files found in the dataset directory.")
        return
    
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
    print("1. XGBoost Classifier")
    print("2. Random Forest Classifier")
    
    try:
        model_choice = int(input("Enter the number of the model you want to use (1-2): "))
        if model_choice < 1 or model_choice > 2:
            print("Invalid choice. Using XGBoost Classifier.")
            model_choice = 1
    except:
        print("Invalid input. Using XGBoost Classifier.")
        model_choice = 1
    
    # Configure model based on choice
    if model_choice == 1:
        model = XGBClassifier(
            learning_rate=0.2,
            n_estimators=100,  # Reduced for faster training
            max_depth=3,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1,
            seed=27
        )
        print("Using XGBoost Classifier")
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        print("Using Random Forest Classifier")
    
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
    trained_model, tuned_params = run(
        models=models_list,
        dataset_file=dataset_file,
        tune_model=tune_model,
        dataset_path=dataset_path
    )
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()