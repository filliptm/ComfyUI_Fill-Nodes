import csv
import io
import random

class FL_CSVExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_data": ("CSV",),
                "seed": ("INT", {"default": 666666, "min": 0, "max": 999999}),
                "column_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            },
            "optional": {
                "skip_header": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("extracted_text", "selected_row", "full_row_data")
    FUNCTION = "extract_csv_row"
    CATEGORY = "🏵️Fill Nodes/Captioning"

    def extract_csv_row(self, csv_data, seed, column_index, skip_header=True):
        if not csv_data:
            raise ValueError("CSV data is not provided.")

        try:
            # Convert binary CSV data to string
            csv_string = csv_data.decode('utf-8')

            # Create a CSV reader from the string
            csv_reader = csv.reader(io.StringIO(csv_string))
            rows = list(csv_reader)

            if not rows:
                raise ValueError("CSV data is empty")

            # Skip header if requested
            data_rows = rows[1:] if skip_header and len(rows) > 1 else rows

            if not data_rows:
                raise ValueError("No data rows found in CSV")

            # Use seed to select row (similar to FL_Image_Randomizer pattern)
            num_rows = len(data_rows)
            selected_row_index = seed % num_rows
            selected_row = data_rows[selected_row_index]

            # Check if column index is valid
            if column_index >= len(selected_row):
                raise ValueError(f"Column index {column_index} is out of range. Row has {len(selected_row)} columns.")

            # Extract the text from the specified column
            extracted_text = selected_row[column_index].strip()

            # Create full row data as comma-separated string
            full_row_data = ",".join(f'"{cell}"' for cell in selected_row)

            # Calculate the actual row number (accounting for header if skipped)
            actual_row_number = selected_row_index + (1 if skip_header else 0)

            print(f"Extracted from row {actual_row_number + 1}, column {column_index}: {extracted_text[:50]}{'...' if len(extracted_text) > 50 else ''}")

        except Exception as e:
            print(f"Error extracting CSV data: {str(e)}")
            raise ValueError(f"Failed to extract CSV data: {str(e)}")

        return (extracted_text, actual_row_number + 1, full_row_data)

    @classmethod
    def IS_CHANGED(cls, csv_data, seed, column_index, skip_header=True):
        return float("NaN")