#!/bin/bash

# Define the datasets, models, and missing types
datasets=("breast" "australian" "heart" "student")
models=("genrbf")
missing_types=("mar" "mnar")


# Function to check if the result CSV file exists
csv_exists() {
    dataset=$1
    missing_type=$2
    model=$3
    output_path="results/${missing_type}/${dataset}/${model}.csv"
    
    if [ -f "$output_path" ]; then
        return 0  # CSV exists
    else
        return 1  # CSV does not exist
    fi
}

# Loop through datasets, models, and missing types
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for missing_type in "${missing_types[@]}"; do
            # Check if the CSV file exists
            if ! csv_exists "$dataset" "$missing_type" "$model"; then
                echo "Missing or improperly generated CSV: results/${missing_type}/${dataset}/${model}.csv"
                echo "Rerunning test for dataset: ${dataset}, missing_type: ${missing_type}, model: ${model}..."
                
                # Construct and run the command
                python main.py --datasets "$dataset" --models "$model" --missing_types "$missing_type" --save
                
                if [ $? -eq 0 ]; then
                    echo "Test completed successfully for dataset: ${dataset}, missing_type: ${missing_type}, model: ${model}."
                else
                    echo "Error running the test for dataset: ${dataset}, missing_type: ${missing_type}, model: ${model}."
                fi
            else
                echo "CSV exists for dataset: ${dataset}, missing_type: ${missing_type}, model: ${model}."
            fi
        done
    done
done