import aiohttp
import asyncio
from PIL import Image
import io
import base64
import os

#removed api key from input for safer use
class FL_SimpleGPTVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["gpt-4o-mini", "gpt-4o", "gpt-4-vision-preview"],),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant that describes images accurately and concisely.",
                    "multiline": True}),
                "request_prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "max_tokens": ("INT", {"default": 300, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "detail": (["auto", "low", "high"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_caption"
    CATEGORY = "🏵️Fill Nodes/GPT"

    async def process_image(self, session, img, model, system_prompt, request_prompt, max_tokens, temperature, detail):
        # Encode image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": request_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}",
                                "detail": detail
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                        print(f"Rate limited. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    data = await response.json()
                    return data['choices'][0]['message']['content']
            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    retry_after = int(e.headers.get('Retry-After', base_delay * (2 ** attempt)))
                    print(f"Rate limited. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                else:
                    return f"Error processing image: {str(e)}"
            except Exception as e:
                return f"Unexpected error: {str(e)}"

        return "Failed to process image after multiple retries due to rate limiting."

    def generate_caption(self, image, model, system_prompt, request_prompt, max_tokens, temperature, detail):
        api_key = os.getenv("OPENAI_API_KEY") # changed to look for env variable
        if not api_key:
            return ("API key is not set as an environment variable",)

        # Convert tensor to PIL Image
        pil_img = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype('uint8'))

        async def main():
            async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {api_key}"}) as session:
                result = await self.process_image(session, pil_img, model, system_prompt, request_prompt, max_tokens,
                                                  temperature, detail)
                return result

        try:
            result = asyncio.run(main())
            return (result,)
        except Exception as e:
            error_message = f"Error in API request: {str(e)}"
            print(error_message)
            return (error_message,)