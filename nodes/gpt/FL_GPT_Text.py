import aiohttp
import asyncio
import os
import sys
from tqdm import tqdm

class FL_GPT_Text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Enter your OpenAI API key here"}),
                "model": (["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant that provides accurate and concise information.",
                    "multiline": True}),
                "user_prompt": ("STRING", {"default": "Hello, can you help me with something?", "multiline": True}),
                "max_tokens": ("INT", {"default": 500, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "save_to_file": ("BOOLEAN", {"default": False}),
                "output_directory": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": "gpt_response.txt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "🏵️Fill Nodes/GPT"

    async def call_openai_api(self, session, model, system_prompt, user_prompt, max_tokens, temperature, top_p, 
                             frequency_penalty, presence_penalty):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        try:
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data['choices'][0]['message']['content']
        except aiohttp.ClientResponseError as e:
            print(f"API Error: {str(e)}")
            return f"Error: {str(e)}"
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return f"Error: {str(e)}"

    def generate_text(self, api_key, model, system_prompt, user_prompt, max_tokens, temperature, top_p,
                     frequency_penalty, presence_penalty, save_to_file=False, output_directory="", filename="gpt_response.txt"):
        # Use provided API key or fall back to environment variable
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            
        try:
            if not api_key:
                raise ValueError("API key is not provided and not set as an environment variable. Please provide an API key.")

            async def main():
                headers = {"Authorization": f"Bearer {api_key}"}
                async with aiohttp.ClientSession(headers=headers) as session:
                    return await self.call_openai_api(
                        session, model, system_prompt, user_prompt, max_tokens,
                        temperature, top_p, frequency_penalty, presence_penalty
                    )

            response = asyncio.run(main())

            # Save to file if requested
            if save_to_file and output_directory:
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                
                file_path = os.path.join(output_directory, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                print(f"Response saved to: {file_path}")

            return (response,)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(error_message)
            return (error_message,)