import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import re
from PIL import Image
from pathlib import Path

from comfy.utils import ProgressBar

class FL_WordFrequencyGraph:
    DESCRIPTION = """
FL_WordFrequencyGraph scans a directory for .txt files, analyzes word frequency across all files,
and generates a visual bar graph showing word usage statistics. The graph displays the most frequently
used words on the left side and least used on the right, creating an attractive visualization of
text data patterns.
"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "",
                    "description": "Path to directory containing .txt files"
                }),
                "title_name": ("STRING", {
                    "default": "Word Frequency Analysis",
                    "description": "Custom title for the graph"
                }),
                "max_words": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 1,
                    "description": "Maximum number of words to display"
                }),
                "min_word_length": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "description": "Minimum word length to include"
                }),
                "image_width": ("INT", {
                    "default": 1920,
                    "min": 800,
                    "max": 4096,
                    "step": 32,
                    "description": "Output image width"
                }),
                "image_height": ("INT", {
                    "default": 1080,
                    "min": 600,
                    "max": 2160,
                    "step": 32,
                    "description": "Output image height"
                }),
                "color_scheme": (["blue", "green", "red", "purple", "orange", "rainbow"], {
                    "default": "blue",
                    "description": "Color scheme for the graph"
                }),
                "exclude_common_words": ("BOOLEAN", {
                    "default": True,
                    "description": "Exclude common words (the, and, or, etc.)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_word_frequency_graph"
    CATEGORY = "🏵️Fill Nodes/Captioning"

    def generate_word_frequency_graph(self, directory_path, title_name, max_words, min_word_length,
                                    image_width, image_height, color_scheme, exclude_common_words):
        
        # Validate directory path
        if not directory_path or not os.path.exists(directory_path):
            raise ValueError(f"Directory path does not exist: {directory_path}")
        
        # Common words to exclude
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'among', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'its', 'our', 'their', 'is', 'am', 'not', 'no', 'yes', 'so', 'if', 'when',
            'where', 'why', 'how', 'what', 'who', 'which', 'than', 'then', 'now', 'here',
            'there', 'more', 'most', 'some', 'any', 'all', 'each', 'every', 'both', 'either',
            'neither', 'one', 'two', 'first', 'last', 'next', 'other', 'another', 'same',
            'different', 'new', 'old', 'good', 'bad', 'big', 'small', 'long', 'short', 'high',
            'low', 'right', 'left', 'only', 'just', 'also', 'even', 'still', 'already', 'yet'
        } if exclude_common_words else set()
        
        # Find all .txt files in directory
        txt_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        
        if not txt_files:
            raise ValueError(f"No .txt files found in directory: {directory_path}")
        
        print(f"Found {len(txt_files)} .txt files")
        
        # Process all text files
        word_counter = Counter()
        pbar = ProgressBar(len(txt_files))
        
        for i, file_path in enumerate(txt_files):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().lower()
                    # Extract words using regex (letters only)
                    words = re.findall(r'[a-zA-Z]+', text)
                    
                    # Filter words by length and common words
                    filtered_words = [
                        word for word in words 
                        if len(word) >= min_word_length and word not in common_words
                    ]
                    
                    word_counter.update(filtered_words)
                    
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            pbar.update_absolute(i)
        
        if not word_counter:
            raise ValueError("No valid words found in the text files")
        
        # Get top words
        top_words = word_counter.most_common(max_words)
        words, counts = zip(*top_words)
        
        # Create the graph
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(image_width/100, image_height/100), dpi=100)
        
        # Set colors based on scheme
        if color_scheme == "rainbow":
            colors = plt.cm.rainbow(np.linspace(0, 1, len(words)))
        else:
            color_map = {
                "blue": plt.cm.Blues_r,
                "green": plt.cm.Greens_r,
                "red": plt.cm.Reds_r,
                "purple": plt.cm.Purples_r,
                "orange": plt.cm.Oranges_r
            }
            colors = color_map[color_scheme](np.linspace(0.3, 1.0, len(words)))
        
        # Create bars
        bars = ax.barh(range(len(words)), counts, color=colors)
        
        # Customize the plot
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=max(8, min(14, 300 // len(words))))
        ax.set_xlabel('Word Frequency', fontsize=14, color='white')
        ax.set_title(f'{title_name}\n{len(txt_files)} files, {sum(counts)} total words',
                    fontsize=16, color='white', pad=20)
        
        # Invert y-axis so most frequent words are at the top
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                   str(count), ha='left', va='center', 
                   fontsize=max(6, min(10, 200 // len(words))), color='white')
        
        # Style the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set background color
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to PIL Image using buffer_rgba method
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        buf = np.asarray(buf)
        # Remove alpha channel and convert RGBA to RGB
        buf = buf[:, :, :3]
        
        plt.close(fig)
        
        # Convert to torch tensor
        image_tensor = torch.from_numpy(buf.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        print(f"Generated word frequency graph with {len(words)} words")
        
        return (image_tensor,)