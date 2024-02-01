# ComfyUI_Fill-Nodes

Image randomizer from directory, Image Captioning saver.

Image randomizer: - A load image directory node that allows you to pull images either in sequence (Per que render) or at random (also per que render)
-
Image Captioning saver: - takes an input image (single or batch) and saves a matching .txt file with the image with desired captioning. Both files will be over written for continuous experimentation
-
## Video


https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/8d222109-cda1-40f8-9558-6f8136e07617

It uses a dummy int value that you attach a seed to to enure that it will continue to pull new images from your directory even if the seed is fixed. In order to do this right click the node and turn the run trigger
into an input and connect a seed generator of your choice set to random.

interesting uses for this


  -loading up a directory and letting it cycle through all your images in order
  
  -connecting this node to something like IPAdapter, while being set to random, allowing you to cycle through styles via images
  
  -batch processing of any kind on large amounts of images



![image](https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/69ab3151-2e16-4b54-b9ae-17e4bf0f0157)
