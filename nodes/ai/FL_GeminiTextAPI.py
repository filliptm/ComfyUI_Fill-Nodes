import os
import time
import random
import torch
import traceback
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any


class FL_GeminiTextAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ([
                    "gemini-2.5-pro",
                    "gemini-2.5-flash", 
                    "gemini-2.5-flash-lite",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-2.5-pro-preview-06-05",
                    "gemini-2.5-flash-preview-05-20",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash"
                ], {"default": "gemini-2.5-flash"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffff}),
            },
            "optional": {
                "system_instructions": ("STRING", {"multiline": True, "default": ""}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "🏵️Fill Nodes/AI"

    def __init__(self):
        """Initialize logging system"""
        self.log_messages = []

    def _log(self, message):
        """Global logging function: record to log list"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[FL_GeminiTextAPI] {timestamp}: {message}"
        print(formatted_message)
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def _call_gemini_api(self, client, model, contents, gen_config, retry_count=0, max_retries=3):
        """Call Gemini API with retry logic using the updated generate_content method"""
        try:
            self._log(f"API call attempt #{retry_count + 1}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )

            # Check if response is valid
            if hasattr(response, 'text'):
                self._log("Valid API response received")
                return response
            else:
                self._log("Invalid API response format")
                if retry_count < max_retries - 1:
                    self._log(f"Retrying in 2 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)  # Wait 2 seconds before retry
                    return self._call_gemini_api(client, model, contents, gen_config, retry_count + 1, max_retries)
                else:
                    self._log(f"Maximum retries ({max_retries}) reached. Giving up.")
                    return None

        except Exception as e:
            self._log(f"API call error: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = 2 * (retry_count + 1)  # Progressive backoff: 2s, 4s, 6s...
                self._log(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._call_gemini_api(client, model, contents, gen_config, retry_count + 1, max_retries)
            else:
                self._log(f"Maximum retries ({max_retries}) reached. Giving up.")
                return None

    def generate_text(self, prompt, api_key, model, temperature, max_output_tokens, seed,
                      system_instructions="", top_p=0.95, top_k=64):
        """Generate text response from Gemini API using the new client structure"""
        # Reset log messages
        self.log_messages = []

        try:
            # Check if API key is provided
            if not api_key:
                error_message = "Error: No API key provided. Please enter Google API key in the node."
                self._log(error_message)
                return (f"## ERROR: {error_message}\n\nPlease provide a valid Google API key.",)

            # Create client instance with API key
            client = genai.Client(api_key=api_key)

            # Set random seeds for reproducibility
            random.seed(seed)
            torch.manual_seed(seed)

            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
                candidate_count=1
            )

            # Add system instructions if provided
            if system_instructions and system_instructions.strip():
                self._log(f"Using system instructions: {system_instructions[:50]}...")
                gen_config.system_instruction = system_instructions

            self._log(f"Sending prompt to Gemini API (model: {model}, temp: {temperature})")

            # Make API call with contents parameter
            response = self._call_gemini_api(
                client=client,
                model=model,
                contents=[prompt],  # Contents expects a list
                gen_config=gen_config,
                max_retries=3
            )

            # Check if we got a valid response
            if response is None:
                error_text = "Failed to get response from Gemini API after multiple attempts."
                self._log(error_text)
                return (f"## API Error\n{error_text}\n\n## Debug Log\n" + "\n".join(self.log_messages),)

            # Extract and return the raw text from the response
            result_text = response.text.strip()  # Remove any leading/trailing whitespace
            
            self._log(f"Received response ({len(result_text)} characters)")
            
            return (result_text,)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            self._log(error_message)
            traceback.print_exc()

            # Return error message and debug log
            return (f"## Error\n{error_message}\n\n## Debug Log\n" + "\n".join(self.log_messages),)