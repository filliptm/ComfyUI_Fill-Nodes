import openai
import base64
import io
import os
import json
import asyncio
import aiohttp

import torch
from PIL import Image
from torchvision.transforms import functional as TF


class FL_Dalle3:
    def __init__(self):
        self.__client = openai.AsyncOpenAI()
        self.__previous_params = None
        self.__cache_images = Nonee
        self.__cache_revised_prompts = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (["1024x1024", "1024x1792", "1792x1024"],),
                "dummy_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "great picture"
                }),
                "quality": (["HD", "Standard"],),
                "style": (["vivid", "natural"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10}),
                "retry": ("INT", {"default": 0, "min": 0, "max": 5}),
            },
            "optional": {
                "auto_save": ("BOOLEAN", {"default": False}),
                "auto_save_dir": ("STRING", {"default": "./output_dalle3"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGES", "WIDTH", "HEIGHT", "REVISED_PROMPTS")
    FUNCTION = "generate_images"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/GPT"

    async def generate_single_image(self, prompt, resolution, quality, style, retry):
        for retry_count in range(retry + 1):
            try:
                response = await self.__client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=resolution,
                    quality="hd" if quality == "HD" else "standard",
                    style="vivid" if style == "vivid" else "natural",
                    n=1,
                    response_format="b64_json"
                )
                return response
            except openai.BadRequestError as ex:
                if retry_count >= retry:
                    raise ex
                print(
                    f"FL_OpenAiDalle3: received BadRequestError, retrying... #{retry_count + 1} : {json.dumps(ex.response.json())}")
        return None

    async def generate_batch(self, prompt, resolution, quality, style, batch_size, retry):
        tasks = [self.generate_single_image(prompt, resolution, quality, style, retry) for _ in range(batch_size)]
        return await asyncio.gather(*tasks)

    def generate_images(self, resolution, dummy_seed, prompt, quality, style, batch_size, retry, auto_save=False,
                        auto_save_dir="./output_dalle3"):
        current_params = (resolution, dummy_seed, prompt, quality, style, batch_size)

        if self.__cache_images is None or self.__previous_params != current_params:
            responses = asyncio.run(self.generate_batch(prompt, resolution, quality, style, batch_size, retry))

            images = []
            revised_prompts = []

            for i, r0 in enumerate(responses):
                if r0 is None:
                    continue

                im0 = Image.open(io.BytesIO(base64.b64decode(r0.data[0].b64_json)))

                if auto_save:
                    os.makedirs(auto_save_dir, exist_ok=True)
                    next_index = len([f for f in os.listdir(auto_save_dir) if f.endswith('.png')]) + 1
                    image_file_name = os.path.join(auto_save_dir, f"dalle3_output_{next_index:06d}.png")
                    state_file_name = os.path.join(auto_save_dir, f"dalle3_output_{next_index:06d}.json")
                    im0.save(image_file_name)
                    with open(state_file_name, "wt") as f:
                        json.dump({
                            "resolution": resolution,
                            "prompt": prompt,
                            "quality": quality,
                            "style": style,
                            "batch_index": i
                        }, f, indent=2, ensure_ascii=False)

                im1 = TF.to_tensor(im0.convert("RGBA"))
                im1[:3, im1[3, :, :] == 0] = 0
                images.append(im1)
                revised_prompts.append(r0.data[0].revised_prompt)

            self.__previous_params = current_params
            self.__cache_images = images
            self.__cache_revised_prompts = revised_prompts
        else:
            images = self.__cache_images
            revised_prompts = self.__cache_revised_prompts

        images_tensor = torch.stack(images)
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        images_tensor = images_tensor[:, :, :, :3]
        width, height = map(int, resolution.split("x"))
        return images_tensor, width, height, ", ".join(revised_prompts)