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

Various FX: - Video effects (ACTIVE WIP) These effects both work with video and images
-
CAUTION THE PIXEL SORT NODE IS A HEAVY PROCESS, TAKES 5-10 seconds per frame depending on the resolution
Play around with these nodes, there is a lot to play with within each of them, and when you chain them together!
Soon, I will be adding a masking feature that will allow you to mask the effect right inside the node.
I will also in the grander scheme of things be adding variable frame parameters based on sine or any wave inputs to make the effects "Breath" and change over time.

![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/8b6bcfa5-f050-44ff-abe1-1f18afd75ecc)
