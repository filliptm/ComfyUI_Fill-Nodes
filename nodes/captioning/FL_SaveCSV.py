import os
import comfy.utils

class FL_SaveCSV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_data": ("CSV",),
                "output_directory": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": "captions.csv"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_csv"
    CATEGORY = "🏵️Fill Nodes/Captioning"
    OUTPUT_NODE = True

    def save_csv(self, csv_data, output_directory, filename):
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Construct the full file path
        file_path = os.path.join(output_directory, filename)

        # Ensure the filename ends with .csv
        if not file_path.lower().endswith('.csv'):
            file_path += '.csv'

        # Write the CSV data to the file
        try:
            with open(file_path, 'wb') as f:
                f.write(csv_data)
            print(f"CSV file saved successfully: {file_path}")
        except Exception as e:
            print(f"Error saving CSV file: {str(e)}")

        return ()

    @classmethod
    def IS_CHANGED(cls, csv_data, output_directory, filename):
        return float("NaN")