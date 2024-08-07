import os
import csv
import io


class FL_CaptionToCSV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("CSV",)
    FUNCTION = "create_csv"
    CATEGORY = "🏵️Fill Nodes/Captioning"
    OUTPUT_NODE = True

    def create_csv(self, image_directory):
        # Get all image files and their corresponding caption files
        image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Sort files to ensure consistent order

        # Prepare CSV data
        csv_data = []
        for image_file in image_files:
            caption_file = os.path.splitext(image_file)[0] + '.txt'
            caption_path = os.path.join(image_directory, caption_file)
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            except FileNotFoundError:
                caption = "No caption found"

            csv_data.append([image_file, caption])

        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['image_file', 'caption'])  # Header
        writer.writerows(csv_data)

        # Get the CSV content as a string
        csv_content = output.getvalue()

        # Convert to bytes for compatibility with other nodes
        csv_bytes = csv_content.encode('utf-8')

        return (csv_bytes,)

    @classmethod
    def IS_CHANGED(cls, image_directory):
        return float("NaN")