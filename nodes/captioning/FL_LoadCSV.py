import os
import csv
import io

class FL_LoadCSV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("CSV", "INT")
    RETURN_NAMES = ("csv_data", "row_count")
    FUNCTION = "load_csv"
    CATEGORY = "🏵️Fill Nodes/Captioning"

    def load_csv(self, csv_path):
        if not csv_path:
            raise ValueError("CSV file path is not provided.")

        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file does not exist: {csv_path}")

        if not csv_path.lower().endswith('.csv'):
            raise ValueError("File must have .csv extension")

        try:
            # Read the CSV file as binary data to match the expected CSV type
            with open(csv_path, 'rb') as f:
                csv_data = f.read()

            # Also read as text to count rows for the row_count output
            with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                row_count = sum(1 for row in reader)

            print(f"CSV file loaded successfully: {csv_path} ({row_count} rows)")

        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            raise ValueError(f"Failed to load CSV file: {str(e)}")

        return (csv_data, row_count)

    @classmethod
    def IS_CHANGED(cls, csv_path):
        if not csv_path or not os.path.exists(csv_path):
            return float("NaN")
        return os.path.getmtime(csv_path)