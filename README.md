# ComfyUI_Fill-Nodes

The start of a pack that I will continue to build out to fill the gaps of nodes and functionality that I feel is missing in comfyUI

For now this pack has a single node. A load image directory node that allows you to pull images either in sequence (Per que render) or at random (also per que render)

## Video


https://github.com/filliptm/ComfyUI_Fill-Nodes/assets/55672949/8d222109-cda1-40f8-9558-6f8136e07617

It uses a dummy int value that you attach a seed to to enure that it will continue to pull new images from your directory even if the seed is fixed. In order to do this right click the node and turn the run trigger
into an input and connect a seed generator of your choice set to random.

interesting uses for this
  -loading up a directory and letting it cycle through all your images in order
  -connecting this node to something like IPAdapter, while being set to random, allowing you to cycle through styles via images
  -batch processing of any kind on large amounts of images
