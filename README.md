# ComfyUI_Fill-Nodes

Image randomizer from directory, Image Captioning saver.

Image randomizer: - A load image directory node that allows you to pull images either in sequence (Per que render) or at random (also per que render)
-
## Video


https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/8d222109-cda1-40f8-9558-6f8136e07617

It uses a dummy int value that you attach a seed to to enure that it will continue to pull new images from your directory even if the seed is fixed. In order to do this right click the node and turn the run trigger
into an input and connect a seed generator of your choice set to random.

interesting uses for this


  -loading up a directory and letting it cycle through all your images in order

  -connecting this node to something like IPAdapter, while being set to random, allowing you to cycle through styles via images

  -batch processing of any kind on large amounts of images

Image Captioning saver: - takes an input image (single or batch) and saves a matching .txt file with the image with desired captioning.
-
Both files will be over written for continuous experimentation. Required to have an output attached for monitoring. Will overwrite images and text on each run. Built this node to save Lora captions from my Dataset Creator

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/69ab3151-2e16-4b54-b9ae-17e4bf0f0157)

Dimension Display: - Simply shows the dimension of an image in a string for monitoring. No need for INTS.
-

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/05286d8f-bf8b-4737-b2f8-635a14f42d7a)



Pixelator: - Custom effect that bit reduces an image and makes it black and white. See examples
-

current implementation requires you to break batches into a list and back into a batch if you want to use it on video. for VRAM management.

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/6806e256-0f57-48eb-be96-02f880f68de0)


Audio Tools (WIP): - Load audio, scans for BPM, crops audio to desired bars and duration
-

other nodes that are a work in progress take the sliced audio/bpm/fps and hold an image for the duration.
There is also a VHS converter node that allows you to load audio into the VHS video combine for audio insertion on the fly!

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/e1b642e2-29d7-442a-a657-a32ca0fac9c4)

Directory Crawler: - Simple node that loads all images in a directory and any subdirectories as well
-

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/7f6862c7-60dc-4561-8b58-72b489903107)


Raw Code Node: - Simple node that loads Python and allows you to dev inside comfy without having to reload the instance every time
-
Great for developing ideas and writing custom stuff quickly


![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/db439865-e3c5-4e52-b37c-c3ba601c0840)

Glitch: Video and image effect
-
Slices up your image or video to make a glitching feel

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/b9bc2f82-19e3-4877-bb98-0801c4ceb96f)

Ripple: Video and image effect
-
Ripples your video or image

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/660983b1-1090-400b-9c92-4e5d3a1eb2b6)

Pixel Sort: Video and image
-
CAUTION: This node is a very heavy operation. It takes 5-10 seconds per frame. (WIP)

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/7ab1785a-fab7-4206-bf9b-fef48896b518)

Hexagon: Video an image
-
This one is really fun. It masks your image and video in slices, but thats not all! Each slice acts as its own video or image when you start rotating and messing with the parameters.

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/06ddaae1-2c1e-41c0-af7f-d713f1bc6d91)

Ascii: Video and image
-
This one allows for a TON of different styles. This node also works with Alt Codes like this: alt+3 = ♥ or alt+219 = █
If you play with the spacing of 219 you can actually get a pixel art effect. ALSO, the last character in the list will always be applied to the highest luminance areas of the image. This is useful because you can do silly things like leave the last character as a blank space, allowing for negative space to be applied to light areas.

#### SYSTEM vs. LOCAL FONTS

The default font list is populated from the fonts located within the extension/fonts folder. You  can add more fonts to this location and when ComfyUI is started it will load those fonts into the list.

You also have the option to use system fonts. You can set the env var:

`SET FL_USE_SYSTEM_FONTS=true (default: false)`

And the dropdown will populate with all the available system ttf and otf fonts.

<img width="1136" alt="Screenshot 2024-04-29 192646" src="https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/926287e9-e22a-4e64-9e4f-7fd6e096b558">

Prompt Selector: - simple prompt selector/randomizer
-
Great for iterating through a lot of prompts or randomizing a list of things you already know works.

<img width="1021" alt="Screenshot 2024-05-15 144223" src="https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/1e96e75a-e9f4-4056-a55a-0ee1cc316dff">

Random Number Range: - Randomize numbers within a selected range.
-
Has a lot of uses where you want numbers randomized, but you need them in a specific range for error purposes.

<img width="683" alt="Screenshot 2024-05-15 144242" src="https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/b2d15b5c-aeda-44d1-b341-075940d83a3f">

Half Tone FX: - Creates a black and white half-tone effect.
-
Lots of fun with this one. Get interesting effects on both images and video.

<img width="953" alt="Screenshot 2024-05-15 144011" src="https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/486e0e4f-ad37-4353-8e92-42e85767f882">

Infinite Zoom: - Creates a zooming effect for both images and video.
-
![Screenshot 2024-05-27 222117](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/7874b38d-59ad-416d-acbc-ef9cdcc78abd)

Paper Drawn: - Filter effect that makes your images and videos look like pencil drawn.
-
![Screenshot 2024-05-27 222434](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/32fd03ff-5f8b-4d55-b616-6365528fb218)

Image Notes: - Adds a black bar with a string input to save images with notes.
-
![Screenshot 2024-06-07 223629](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/84123521-6263-498f-ae57-4949f76e67a9)

