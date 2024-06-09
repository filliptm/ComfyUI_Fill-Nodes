import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Fill-Nodes.appearance", // Extension name
    nodeCreated(node) {

        switch (node.comfyClass) {
            case "FL_ImageRandomizer":
            case "FL_ImageCaptionSaver":
            case "FL_ImageDimensionDisplay":
            case "FL_AudioPreview":
            case "FL_ImageDurationSync":
            case "FL_AudioConverter":
            case "FL_AudioFrameCalculator":
            case "FL_CodeNode":
            case "FL_ImagePixelator":
            case "FL_DirectoryCrawl":
            case "FL_Ascii":
            case "FL_Glitch":
            case "FL_Ripple":
            case "FL_PixelSort":
            case "FL_HexagonalPattern":
            case "FL_NFTGenerator":
            case "FL_HalftonePattern":
            case "FL_RandomNumber":
            case "FL_PromptSelector":
            case "FL_Shadertoy":
            case "FL_PixelArtShader":
            case "FL_InfiniteZoom":
            case "FL_PaperDrawn":
            case "FL_ImageNotes":
            case "FL_ImageCollage":

                // node.setSize([200, 58]);
                node.color = "#16727c";
                node.bgcolor = "#4F0074";

                break;
        }
    }
});
