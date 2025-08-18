import os
import pandas as pd
from datasets import load_dataset

# 1. Load the dataset from Hugging Face
dataset = load_dataset("newfacade/LeetCodeDataset")

# 2. Access train and test splits
train_data = dataset["train"]
test_data = dataset["test"]

# 3. Convert to pandas DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# 4. Create a folder to save CSVs
output_folder = "/Users/anand/Desktop/Ml_deep_notes_for_myself/LeetCodeDatasetCSV"
os.makedirs(output_folder, exist_ok=True)

# 5. Save as CSV files
train_csv_path = os.path.join(output_folder, "train.csv")
test_csv_path = os.path.join(output_folder, "test.csv")

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Train dataset saved to: {train_csv_path}")
print(f"Test dataset saved to: {test_csv_path}")
