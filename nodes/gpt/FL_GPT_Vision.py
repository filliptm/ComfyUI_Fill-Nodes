import aiohttp
import asyncio
from PIL import Image
import io
import os
import sys
from tqdm import tqdm
import base64

# removed api key from input for safer use
class FL_GPT_Vision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["gpt-4o-mini", "gpt-4o", "gpt-4-vision-preview"],),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant that describes images accurately and concisely.",
                    "multiline": True}),
                "request_prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "output_directory": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 300, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "detail": (["auto", "low", "high"],),
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 20}),
            },
            "optional": {
                "images": ("IMAGE",),
                "input_directory": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("message", "output_directory")
    FUNCTION = "generate_captions"
    CATEGORY = "🏵️Fill Nodes/GPT"

    async def process_image(self, session, img, img_filename, output_directory, overwrite, api_key, model,
                            system_prompt, request_prompt, max_tokens, temperature, detail):
        caption_filename = os.path.splitext(img_filename)[0] + ".txt"
        img_path = os.path.join(output_directory, img_filename)
        caption_path = os.path.join(output_directory, caption_filename)

        if not overwrite and os.path.exists(caption_path):
            return None

        # Save the image
        img.save(img_path)

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

        try:
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                caption = data['choices'][0]['message']['content']

            # Save the caption
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)

            return caption
        except aiohttp.ClientResponseError as e:
            print(f"Error processing {img_filename}: {str(e)}")
            return None

    async def process_batch(self, batch, session, *args):
        tasks = [self.process_image(session, img, filename, *args) for img, filename in batch]
        return await asyncio.gather(*tasks)

    def generate_captions(self, model, system_prompt, request_prompt, output_directory, overwrite, max_tokens,
                          temperature, detail, batch_size, images=None, input_directory=None):
        api_key = os.getenv("OPENAI_API_KEY") #looks for api key in env variable
        try:
            if not api_key:
                raise ValueError("API key is not set as an environment variable")

            if images is None and not input_directory:
                raise ValueError("Either 'images' or 'input_directory' must be provided")

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            image_list = []
            if images is not None:
                for i, img in enumerate(images):
                    pil_img = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype('uint8'))
                    image_list.append((pil_img, f"image_{i}.jpg"))

            if input_directory:
                if not os.path.exists(input_directory):
                    raise ValueError(f"Input directory does not exist: {input_directory}")
                for filename in os.listdir(input_directory):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        img_path = os.path.join(input_directory, filename)
                        pil_img = Image.open(img_path)
                        image_list.append((pil_img, filename))

            total_images = len(image_list)
            if total_images == 0:
                raise ValueError("No images found to process")

            batches = [image_list[i:i + batch_size] for i in range(0, total_images, batch_size)]

            async def main():
                async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {api_key}"}) as session:
                    all_captions = []
                    for batch in tqdm(batches, desc="Processing batches", file=sys.stdout):
                        batch_captions = await self.process_batch(batch, session, output_directory, overwrite, api_key,
                                                                  model, system_prompt, request_prompt, max_tokens,
                                                                  temperature, detail)
                        all_captions.extend(batch_captions)
                    return all_captions

            captions = asyncio.run(main())

            # Print summary
            print(f"\nTotal images processed: {total_images}")
            print(f"Images and captions saved in: {output_directory}")

            return (f"Captions generated and saved in {output_directory}", output_directory)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(error_message)
            return (error_message, "")