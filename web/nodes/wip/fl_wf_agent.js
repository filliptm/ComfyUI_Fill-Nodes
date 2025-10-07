import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com';
const GEMINI_API_VERSION = 'v1';
const GEMINI_MODEL = 'gemini-2.0-flash';
const SYSTEM_PROMPT = `You are a JavaScript code generator for ComfyUI workflow automation.

Your task is to generate code based on the user's prompt that utilizes the available workflow manipulation functions.
Only return the JavaScript code itself, no explanations or additional text DO NOT HAVE mark down astrics around the code. ONLY
respond with the raw text of the code.
Focus on creating clean, efficient code that accomplishes the requested workflow automation.

Available functions and capabilities:
- Node Management: find(), findLast(), create(), remove(), bypass(), unbypass(), pin(), unpin()
- Connection Management: connect(), getValues(), setValues()
- Layout Management: putOnLeft(), putOnRight(), putOnTop(), putOnBottom(), moveToRight(), moveToBottom()
- Workflow Control: generate(), cancel(), enableAutoQueue(), disableAutoQueue(), setBatchCount()
- System Control: enableSleep(), disableSleep(), enableScreenSaver(), disableScreenSaver(), sendImages()

When generating code:
1. Use proper error handling where appropriate
2. Leverage the provided utility functions
3. Keep the code concise but readable
4. Focus on achieving the specific workflow automation goal
5. Consider edge cases and validation

Remember: Only output the JavaScript code itself - no explanations, comments or additional text unless they are code comments.

Here is an example of how to structure the code. The user prompt for this example was "make the basic comfy workflow that builds itself slowly, that deletes
itself. 

// Animation Timing
const CREATE_DELAY = 100;      // Delay between creating nodes (ms)
const DELETE_DELAY = 100;      // Delay between deleting nodes (ms)
const PAUSE_DELAY = 100;      // Pause before dismantling (ms)

// Node Positioning
const START_X = 100;           // Starting X position
const START_Y = 100;           // Starting Y position
const SPACING_X = 400;         // Horizontal spacing between nodes
const SPACING_Y = 250;         // Vertical spacing (for neg prompt)

// Generation Parameters
const CHECKPOINT = "v1-5-pruned.ckpt";
const POSITIVE_PROMPT = "beautiful landscape, mountains, lake, sunset";
const NEGATIVE_PROMPT = "blur, haze, ugly, text";
const STEPS = 20;
const CFG = 7;
const SAMPLER = "euler";
const WIDTH = 512;
const HEIGHT = 512;
const BATCH_SIZE = 1;

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function buildAndDestroyWorkflow() {
    const nodes = [];
    
    // Create Checkpoint Loader
    const checkpoint = create("CheckpointLoaderSimple", {
        ckpt_name: CHECKPOINT
    });
    checkpoint.pos[0] = START_X;
    checkpoint.pos[1] = START_Y;
    nodes.push(checkpoint);
    await sleep(CREATE_DELAY);

    // Create Positive CLIP
    const clipPos = create("CLIPTextEncode", {
        text: POSITIVE_PROMPT
    });
    clipPos.pos[0] = START_X + SPACING_X;
    clipPos.pos[1] = START_Y;
    connect(checkpoint, "CLIP", clipPos, "clip");
    nodes.push(clipPos);
    await sleep(CREATE_DELAY);

    // Create Negative CLIP
    const clipNeg = create("CLIPTextEncode", {
        text: NEGATIVE_PROMPT
    });
    clipNeg.pos[0] = START_X + SPACING_X;
    clipNeg.pos[1] = START_Y + SPACING_Y;
    connect(checkpoint, "CLIP", clipNeg, "clip");
    nodes.push(clipNeg);
    await sleep(CREATE_DELAY);

    // Create KSampler
    const sampler = create("KSampler", {
        steps: STEPS,
        cfg: CFG,
        sampler_name: SAMPLER,
        scheduler: "normal",
        denoise: 1,
        seed: generateSeed()
    });
    sampler.pos[0] = START_X + SPACING_X * 2;
    sampler.pos[1] = START_Y;
    connect(checkpoint, "MODEL", sampler, "model");
    connect(clipPos, "CONDITIONING", sampler, "positive");
    connect(clipNeg, "CONDITIONING", sampler, "negative");
    nodes.push(sampler);
    await sleep(CREATE_DELAY);

    // Create VAE Decode
    const vae = create("VAEDecode", {});
    vae.pos[0] = START_X + SPACING_X * 3;
    vae.pos[1] = START_Y;
    connect(checkpoint, "VAE", vae, "vae");
    connect(sampler, "LATENT", vae, "samples");
    nodes.push(vae);
    await sleep(CREATE_DELAY);

    // Create Save Image
    const save = create("SaveImage", {
        filename_prefix: "ComfyUI"
    });
    save.pos[0] = START_X + SPACING_X * 4;
    save.pos[1] = START_Y;
    connect(vae, "IMAGE", save, "images");
    nodes.push(save);
    await sleep(CREATE_DELAY);

    // Pause before dismantling
    await sleep(PAUSE_DELAY);

    // Remove nodes in sequence
    for (const node of nodes) {
        remove(node);
        await sleep(DELETE_DELAY);
    }
}

// Execute the workflow
buildAndDestroyWorkflow();


ONLY respond with raw text. do not use any markdown ticks or reply with anything else other than the code itself. again, i need the code
as raw text

When creating nodes in the code, always ensure that you put a 100ms to help the user visually understand whats happening when they click exdcute
`;





const MIN_SEED = 0;
const MAX_SEED = parseInt("0xffffffffffffffff", 16);
const STEPS_OF_SEED = 10;
const DEFAULT_MARGIN_X = 32;
const DEFAULT_MARGIN_Y = 64;
let SLEEP = true;

function getNodes(type) {
  return app.graph._nodes.filter(e => e.comfyClass === "FL_WF_Agent" &&
    e.widgets?.find(e => e.name === "event")?.value === type);
}

function execNodes(type, args) {
  const nodes = getNodes(type);
  for (const node of nodes) {
    execNode(node, args);
  }
}

function execNode(node, args) {
  try {
    if (!node.__eventHandler__) {
      node.__eventHandler__ = {};
    }
    if (!node.properties) {
      node.properties = {};
    }
    if (!node.properties.__eventHandler__) {
      node.properties.__eventHandler__ = {};
    }

    const SELF = node;
    const COMMAND = node.widgets?.find(e => e.name === "javascript")?.value;
    const STATE = node.__eventHandler__;
    const PROPS = node.properties.__eventHandler__;
    const NODES = app.graph._nodes;
    const LINKS = app.graph.links;
    const ARGS = args ?? [];

    const DATE = new Date();
    const YEAR = DATE.getFullYear();
    const MONTH = DATE.getMonth() + 1;
    const DAY = DATE.getDay();
    const HOURS = DATE.getHours();
    const MINUTES = DATE.getMinutes();
    const SECONDS = DATE.getSeconds();

    const BATCH_COUNT = getBatchCount();
    const QUEUE_MODE = getQueueMode();
    const AUTO_QUEUE = getQueueMode() !== "disabled";

    const create = (className, values, options) => createNode.apply(node, [className, values, options]);

    eval(COMMAND);
  } catch(err) {
    console.error(err);
  }
}

function getDefaultCommand() {
  const nodes = app.graph._nodes
    .filter(e => e && e.comfyClass !== "FL_WF_Agent")
    .sort((a, b) => a.id - b.id);

  let text = "";
  for (const node of nodes) {
    const nodeId = node.id;
    const nodeTitle = node.title;
    text += `var n${nodeId} = find(${nodeId}); // ${nodeTitle}\n`;
  }
  text += `\n// You can use javascript code here!`;
  text += `\n// COMMAND: string`;
  text += `\n// STATE: <string, any>`;
  text += `\n// PROPS: <string, any>`;
  text += `\n// NODES: Node[]`;
  text += `\n// LINKS: Link[]`;
  text += `\n// ARGS: any[]`;
  text += `\n// DATE: Date`;
  text += `\n// YEAR: number`;
  text += `\n// MONTH: number`;
  text += `\n// DAY: number`;
  text += `\n// HOURS: number`;
  text += `\n// MINUTES: number`;
  text += `\n// SECONDS: number`;
  text += `\n// QUEUE_MODE: "disabled"|"instant"|"change"`;
  text += `\n// AUTO_QUEUE: boolean`;
  text += `\n// BATCH_COUNT: number`;
  text += `\n// SLEEP: boolean`;
  text += `\n`;
  text += `\n// find(ID|TITLE|TYPE) => Node`;
  text += `\n// findLast(ID|TITLE|TYPE) => Node`;
  text += `\n// exec(...NODE)`;
  text += `\n// getValues(NODE) => { key: value, ... }`;
  text += `\n// setValues(NODE, { key: value, ... })`;
  text += `\n// connect(OUTPUT_NODE, OUTPUT_SLOT_NAME, INPUT_NODE, INPUT_SLOT_NAME?)`;
  text += `\n// create(TYPE, { key: value, ... }) => Node`;
  text += `\n// generateSeed() => number`;
  text += `\n// generateFloat(min, max) => number`;
  text += `\n// generateInt(min, max) => number`;
  text += `\n// random(...any[]) => any`;
  text += `\n// bypass(...NODE)`;
  text += `\n// unbypass(...NODE)`;
  text += `\n// pin(...NODE)`;
  text += `\n// unpin(...NODE)`;
  text += `\n// remove(...NODE)`;
  text += `\n// select(...NODE)`;
  text += `\n// putOnLeft(NODE, TARGET_NODE)`;
  text += `\n// putOnRight(NODE, TARGET_NODE)`;
  text += `\n// putOnTop(NODE, TARGET_NODE)`;
  text += `\n// putOnBottom(NODE, TARGET_NODE)`;
  text += `\n// moveToRight(NODE)`;
  text += `\n// moveToBottom(NODE)`;
  text += `\n// getRect(NODE) => [x, y, width, height]`;
  text += `\n// setRect(NODE, x?, y?, width?, height?)`;
  text += `\n// generate() // Start generation.`;
  text += `\n// cancel() // Cancel current generation.`;
  text += `\n// enableAutoQueue()`;
  text += `\n// disableAutoQueue()`;
  text += `\n// setBatchCount(number)`;
  text += `\n// disableSleep()`;
  text += `\n// enableSleep()`;
  text += `\n// disableScreenSaver()`;
  text += `\n// enableScreenSaver()`;
  text += `\n// sendImages(url, field, ...PREVIEW_NODE)`;
  return text;
}

function findNode(n) {
  if (typeof n === "number") {
    return find(n);
  } else if (typeof n === "string") {
    return find(n);
  } else if (typeof n === "object") {
    return n;
  }
}

function enableAutoQueue() {
  return setQueueMode("instant");
}

function disableAutoQueue() {
  return setQueueMode("disabled");
}

function exec(...nodes) {
  nodes = nodes.map(findNode);
  for (const node of nodes) {
    node.exec();
  }
}

function getQueueMode() {
  return app.extensionManager.queueSettings.mode;
}

function setQueueMode(v) {
  if (typeof v === "boolean") {
    app.extensionManager.queueSettings.mode = v ? "instant" : "disabled";
  } else {
    app.extensionManager.queueSettings.mode = v;
  }
}

function getBatchCount() {
  return app.extensionManager.queueSettings.batchCount;
}

function setBatchCount(v) {
  app.extensionManager.queueSettings.batchCount = v;
}

const createNode = function(className, values, options) {
  values = values ?? {};
  options = { select: true, shiftY: 0, before: false, ...(options || {}) };
  const node = LiteGraph.createNode(className);
  if (!node) {
    throw new Error(`${className} not found.`);
  }

  if (node.widgets) {
    for (const [key, value] of Object.entries(values)) {
      const widget = node.widgets.find(e => e.name === key);
      if (widget) {
        widget.value = value;
      }
    }
  }

  app.graph.add(node);

  if (options.select) {
    app.canvas.selectNode(node, false);
  }

  putOnRight(node, this);
  moveToBottom(node);

  return node;
}

// methods
const match = function(node, query) {
  if (typeof query === "number") {
    return node.id === query;
  } else if (typeof query === "string") {
    return node.title === query || node.comfyClass === query || node.type === query;
  } else if (typeof query === "object") {
    return node.id === query.id;
  } else {
    return false;
  }
}

const find = function(query) {
  for (let i = 0; i < app.graph._nodes.length; i++) {
    const n = app.graph._nodes[i];
    if (match(n, query)) {
      return n;
    }
  }
}

const findLast = function(query) {
  for (let i = app.graph._nodes.length - 1; i >= 0; i--) {
    const n = app.graph._nodes[i];
    if (match(n, query)) {
      return n;
    }
  }
}

const getValues = function(node) {
  node = findNode(node);
  let result = {};
  if (node.widgets) {
    for (const widget of node.widgets) {
      result[widget.name] = widget.value;
    }
  }
  return result;
}

const getValue = getValues;

const setValues = function(node, values) {
  node = findNode(node);
  if (node.widgets) {
    for (const [key, value] of Object.entries(values)) {
      const widget = node.widgets.find(e => e.name === key);
      if (widget) {
        widget.value = value;
      }
    }
  }
}

const setValue = setValues;

const connect = function(outputNode, outputName, inputNode, inputName) {
  outputNode = findNode(outputNode);
  inputNode = findNode(inputNode);

  if (!inputName) {
    inputName = outputName;
  }

  outputName = outputName.toUpperCase();
  inputName = inputName.toLowerCase();

  let output = outputName ? outputNode.outputs?.find(e => e.name.toUpperCase() === outputName) : null;
  let outputSlot;
  let input = inputName ? inputNode.inputs?.find(e => e.name.toLowerCase() === inputName) : null;
  let inputSlot;

  if (output) {
    outputSlot = outputNode.findOutputSlot(output.name);
    if (!input) {
      input = inputNode.inputs?.find(e => e.type === output.type);
      if (input) {
        inputSlot = inputNode.findInputSlot(input.name);
      }
    }
  }

  if (input) {
    inputSlot = inputNode.findInputSlot(input.name);
    if (!output) {
      output = outputNode.outputs?.find(e => e.type === input.type);
      if (output) {
        outputSlot = outputNode.findOutputSlot(output.name);
      }
    }
  }

  if (typeof inputSlot === "number" && typeof outputSlot === "number") {
    outputNode.connect(outputSlot, inputNode.id, inputSlot);
  }
}

const generateFloat = function(min, max) {
  if (typeof min !== "number") {
    min = Number.MIN_SAFE_INTEGER;
  }
  if (typeof max !== "number") {
    max = Number.MAX_SAFE_INTEGER;
  }
  return Math.random() * (max - min) + min;
}

const generateInt = function(min, max) {
  return Math.floor(generateFloat(min, max));
}

const generateSeed = function() {
  let max = Math.min(1125899906842624, MAX_SEED);
  let min = Math.max(-1125899906842624, MIN_SEED);
  let range = (max - min) / (STEPS_OF_SEED / 10);
  return Math.floor(Math.random() * range) * (STEPS_OF_SEED / 10) + min;
}

const random = function(...args) {
  return args[generateInt(0, args.length)];
}

const bypass = function(...nodes) {
  nodes = nodes.map(findNode);
  for (const node of nodes) {
    node.mode = 4;
  }
}

const unbypass = function(...nodes) {
  nodes = nodes.map(findNode);
  for (const node of nodes) {
    node.mode = 0;
  }
}

const pin = function(...nodes) {
  nodes = nodes.map(findNode);
  for (const node of nodes) {
    node.pin(true);
  }
}

const unpin = function(...nodes) {
  nodes = nodes.map(findNode);
  for (const node of nodes) {
    node.pin(false);
  }
}

const remove = function(...nodes) {
  nodes = nodes.map(findNode);
  for (const node of nodes) {
    app.graph.remove(node);
  }
}

const select = function(...nodes) {
  nodes = nodes.map(findNode);
  app.canvas.deselectAll();
  app.canvas.selectNodes(nodes);
}

const generate = function() {
  app.queuePrompt(0, getBatchCount());
}

const cancel = function() {
  api.interrupt();
}

const putOnLeft = function(targetNode, anchorNode) {
  targetNode = findNode(targetNode);
  anchorNode = findNode(anchorNode);
  targetNode.pos[0] = anchorNode.pos[0] - targetNode.size[0] - DEFAULT_MARGIN_X;
  targetNode.pos[1] = anchorNode.pos[1];
}

const putOnRight = function(targetNode, anchorNode) {
  targetNode = findNode(targetNode);
  anchorNode = findNode(anchorNode);
  targetNode.pos[0] = anchorNode.pos[0] + anchorNode.size[0] + DEFAULT_MARGIN_X;
  targetNode.pos[1] = anchorNode.pos[1];
}

const putOnTop = function(targetNode, anchorNode) {
  targetNode = findNode(targetNode);
  anchorNode = findNode(anchorNode);
  targetNode.pos[0] = anchorNode.pos[0];
  targetNode.pos[1] = anchorNode.pos[1] - targetNode.size[1] - DEFAULT_MARGIN_Y;
}

const putOnBottom = function(targetNode, anchorNode) {
  targetNode = findNode(targetNode);
  anchorNode = findNode(anchorNode);
  targetNode.pos[0] = anchorNode.pos[0];
  targetNode.pos[1] = anchorNode.pos[1] + anchorNode.size[1] + DEFAULT_MARGIN_Y;
}

const moveToRight = function(targetNode) {
  targetNode = findNode(targetNode);
  let isChanged = true;
  while(isChanged) {
    isChanged = false;
    for (const node of app.graph._nodes) {
      if (node.id === targetNode.id) {
        continue;
      }
      const top = node.pos[1];
      const bottom = node.pos[1] + node.size[1];
      const left = node.pos[0];
      const right = node.pos[0] + node.size[0];
      const isCollisionX = left <= node.pos[0] + targetNode.size[0] &&
        right >= targetNode.pos[0];
      const isCollisionY = top <= node.pos[1] + targetNode.size[1] &&
        bottom >= targetNode.pos[1];

      if (isCollisionX && isCollisionY) {
        targetNode.pos[0] = right + DEFAULT_MARGIN_X;
        isChanged = true;
      }
    }
  }
}

const moveToBottom = function(targetNode) {
  targetNode = findNode(targetNode);
  let isChanged = true;
  while(isChanged) {
    isChanged = false;
    for (const node of app.graph._nodes) {
      if (node.id === targetNode.id) {
        continue;
      }
      const top = node.pos[1];
      const bottom = node.pos[1] + node.size[1];
      const left = node.pos[0];
      const right = node.pos[0] + node.size[0];
      const isCollisionX = left <= targetNode.pos[0] + targetNode.size[0] &&
        right >= targetNode.pos[0];
      const isCollisionY = top <= targetNode.pos[1] + targetNode.size[1] &&
        bottom >= targetNode.pos[1];

      if (isCollisionX && isCollisionY) {
        targetNode.pos[1] = bottom + DEFAULT_MARGIN_Y;
        isChanged = true;
      }
    }
  }
}

const getX = function(node) {
  return findNode(node).pos[0];
}

const getY = function(node) {
  return findNode(node).pos[1];
}

const getWidth = function(node) {
  return findNode(node).size[0];
}

const getHeight = function(node) {
  return findNode(node).size[1];
}

const getRect = function(node) {
  node = findNode(node);
  return [
    node.pos[0],
    node.pos[1],
    node.size[0],
    node.size[1],
  ];
}

const setX = function(node, n) {
  node = findNode(node);
  node.pos[0] = n;
}

const setY = function(node, n) {
  node = findNode(node);
  node.pos[1] = n;
}

const setWidth = function(node, w) {
  node = findNode(node);
  node.size[0] = w;
  node.onResize(node.size);
}

const setHeight = function(node, h) {
  node = findNode(node);
  node.size[1] = h;
  node.onResize(node.size);
}

const setRect = function(node, x, y, width, height) {
  node = findNode(node);
  if (typeof x !== "number") {
    x = getX(node);
  }
  if (typeof y !== "number") {
    y = getY(node);
  }
  if (typeof width !== "number") {
    width = getWidth(node);
  }
  if (typeof height !== "number") {
    height = getHeight(node);
  }
  node.pos[0] = x;
  node.pos[1] = y;
  node.size[0] = width;
  node.size[1] = height;
  node.onResize(node.size);
}

const disableSleep = async function() {
  const response = await api.fetchApi(`/shinich39/event-handler/disable-sleep`, {
    method: "GET",
  });

  if (response.status !== 200) {
    throw new Error(response.statusText);
  }
  SLEEP = false;
  console.log("[FL_WF_Agent] Disable system sleep");
}

const enableSleep = async function() {
  const response = await api.fetchApi(`/shinich39/event-handler/enable-sleep`, {
    method: "GET",
  });

  if (response.status !== 200) {
    throw new Error(response.statusText);
  }
  SLEEP = true;
  console.log("[FL_WF_Agent] Enable system sleep");
}

const disableScreenSaver = async function() {
  const response = await api.fetchApi(`/shinich39/event-handler/disable-screen-saver`, {
    method: "GET",
  });

  if (response.status !== 200) {
    throw new Error(response.statusText);
  }
  SLEEP = false;
  console.log("[FL_WF_Agent] Disable screen saver");
}

const enableScreenSaver = async function() {
  const response = await api.fetchApi(`/shinich39/event-handler/enable-screen-saver`, {
    method: "GET",
  });

  if (response.status !== 200) {
    throw new Error(response.statusText);
  }
  SLEEP = true;
  console.log("[FL_WF_Agent] Enable screen saver");
}

const getPathFromImg = function(img) {
  const url = new URL(img.src);
  let filename = url.searchParams.get("filename");
  if (filename && filename !== "") {
    filename = "/" + filename;
  }
  let subdir = url.searchParams.get("subfolder");
  if (subdir && subdir !== "") {
    subdir = "/" + subdir;
  }
  let dir = url.searchParams.get("type");
  if (dir && dir !== "") {
    dir = "/" + dir;
  }
  return `ComfyUI${dir}${subdir}${filename}`;
}

const sendImages = async function(url, field, ...filePaths) {
  filePaths = filePaths.reduce((acc, cur) => {
    if (typeof cur === "object" && cur.comfyClass === "PreviewImage") {
      if (cur.imgs) {
        acc.push(
          ...cur.imgs.map((img) => getPathFromImg(img))
        );
      }
    } else if (typeof cur === "string") {
      acc.push(cur);
    }
    return acc;
  }, []);

  if (filePaths.length < 1) {
    throw new Error("Images not found");
  }

  const response = await api.fetchApi(`/shinich39/event-handler/send-images`, {
    method: "POST",
    body: JSON.stringify({
      url: url,
      field: field,
      files: filePaths,
    }),
  });

  if (response.status !== 200) {
    throw new Error(response.statusText);
  }
}

app.registerExtension({
  name: "Comfy.FL_WF_Agent",
  async setup() {
    setTimeout(() => {
      const origQueuePrompt = app.queuePrompt;
      app.queuePrompt = async function(...args) {
        execNodes("before_queued", args);
        const r = await origQueuePrompt.apply(this, arguments);
        return r;
      }

      api.addEventListener("promptQueued", function(...args) {
        execNodes("after_queued", args);
      });

      api.addEventListener("status", function(...args) {
        execNodes("status", args);
      });

      api.addEventListener("progress", function(...args) {
        execNodes("progress", args);
      });

      api.addEventListener("executing", function(...args) {
        execNodes("executing", args);
      });

      api.addEventListener("executed", function(...args) {
        execNodes("executed", args);
      });

      api.addEventListener("execution_start", function(...args) {
        execNodes("execution_start", args);
      });

      api.addEventListener("execution_success", function(...args) {
        execNodes("execution_success", args);
      });

      api.addEventListener("execution_error", function(...args) {
        execNodes("execution_error", args);
      });

      api.addEventListener("execution_cached", function(...args) {
        execNodes("execution_cached", args);
      });

      console.log("[FL_WF_Agent] Event added.");
    }, 1024);
  },
  nodeCreated(node) {
    if (node.comfyClass === "FL_WF_Agent") {
      const jsWidget = node.widgets?.find(e => e.name === "javascript");
      if (jsWidget && jsWidget.value === "// Generated code will appear here") { // Only set default if it's the placeholder
        console.log("[FL_WF_Agent] Setting default JS command for new node:", node.id);
        jsWidget.value = getDefaultCommand();
      }

      // Helper function to list available Gemini models
      async function listGeminiModels(apiKey) {
          try {
              const response = await fetch(`${GEMINI_BASE_URL}/${GEMINI_API_VERSION}/models?key=${apiKey}`);
              if (!response.ok) {
                  throw new Error(`Failed to list models: ${response.statusText}`);
              }
              const data = await response.json();
              console.log("[FL_WF_Agent] Available Gemini Models:", data.models);
              return data.models;
          } catch (error) {
              console.error("[FL_WF_Agent] Error listing models:", error);
              throw error;
          }
      }

      // Helper function to sanitize Gemini response
      function sanitizeGeminiResponse(text) {
          // Remove markdown code blocks
          text = text.replace(/```javascript\n/g, '');
          text = text.replace(/```js\n/g, '');
          text = text.replace(/```\n/g, '');
          text = text.replace(/```/g, '');
          
          // Remove potential HTML/XML tags that might slip through
          text = text.replace(/<\/?[^>]+(>|$)/g, '');
          
          // Remove any leading/trailing whitespace
          text = text.trim();
          
          // Convert smart quotes to regular quotes
          text = text.replace(/[''‛`]/g, "'");
          text = text.replace(/[""]/g, '"');
          
          return text;
      }

      // Helper function to call Gemini API
      async function callGeminiAPI(prompt, apiKey) {
          console.log("[FL_WF_Agent] Calling Gemini API with prompt:", prompt);
          
          if (!apiKey) {
              throw new Error("API key is required");
          }

          try {
              // Try to read node definitions from cache
              let nodeDefinitions = "No node definitions found. Please run 'Scan Nodes' first.";
              
              // First try the direct absolute path
              const absolutePath = "/custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt";
              console.log("[FL_WF_Agent] Trying to load node definitions from absolute path:", absolutePath);
              
              try {
                  const response = await fetch(absolutePath);
                  console.log("[FL_WF_Agent] Response status:", response.status);
                  
                  if (response.ok) {
                      nodeDefinitions = await response.text();
                      console.log("[FL_WF_Agent] Successfully loaded node definitions, size:", nodeDefinitions.length);
                  } else {
                      console.warn("[FL_WF_Agent] Failed to load from absolute path. Status:", response.status);
                      
                      // Try a relative path
                      const relativePath = "./custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt";
                      console.log("[FL_WF_Agent] Trying relative path:", relativePath);
                      
                      try {
                          const relResponse = await fetch(relativePath);
                          if (relResponse.ok) {
                              nodeDefinitions = await relResponse.text();
                              console.log("[FL_WF_Agent] Successfully loaded from relative path, size:", nodeDefinitions.length);
                          } else {
                              console.warn("[FL_WF_Agent] Failed to load from relative path. Status:", relResponse.status);
                              
                              // Try a direct file path based on the current URL
                              const pathFromOrigin = `${window.location.origin}/custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt`;
                              console.log("[FL_WF_Agent] Trying path from origin:", pathFromOrigin);
                              
                              try {
                                  const originResponse = await fetch(pathFromOrigin);
                                  if (originResponse.ok) {
                                      nodeDefinitions = await originResponse.text();
                                      console.log("[FL_WF_Agent] Successfully loaded from origin path, size:", nodeDefinitions.length);
                                  } else {
                                      console.warn("[FL_WF_Agent] Failed to load from origin path. Status:", originResponse.status);
                                  }
                              } catch (e) {
                                  console.warn("[FL_WF_Agent] Error loading from origin path:", e);
                              }
                          }
                      } catch (e) {
                          console.warn("[FL_WF_Agent] Error loading from relative path:", e);
                      }
                  }
              } catch (e) {
                  console.warn("[FL_WF_Agent] Error loading from absolute path:", e);
              }
              
              // Log server-side file system data to console
              console.log("[FL_WF_Agent] Creating mock node definitions as fallback since file could not be accessed");
              
              // If we couldn't load from the file, create some basic node definitions
              if (nodeDefinitions === "No node definitions found. Please run 'Scan Nodes' first.") {
                  nodeDefinitions = `
Node: CheckpointLoaderSimple
Category: loaders
Inputs:
  Required:
    - ckpt_name: STRING
Outputs:
  - MODEL
  - CLIP
  - VAE
--------------------------------------------------
Node: CLIPTextEncode
Category: conditioning
Inputs:
  Required:
    - text: STRING
    - clip: CLIP
Outputs:
  - CONDITIONING
--------------------------------------------------
Node: KSampler
Category: sampling
Inputs:
  Required:
    - model: MODEL
    - positive: CONDITIONING
    - negative: CONDITIONING
    - latent_image: LATENT
    - seed: INT
    - steps: INT
    - cfg: FLOAT
    - sampler_name: STRING
    - scheduler: STRING
    - denoise: FLOAT
Outputs:
  - LATENT
--------------------------------------------------
Node: VAEDecode
Category: latent
Inputs:
  Required:
    - samples: LATENT
    - vae: VAE
Outputs:
  - IMAGE
--------------------------------------------------
Node: SaveImage
Category: image
Inputs:
  Required:
    - images: IMAGE
    - filename_prefix: STRING
Outputs:
  - IMAGE
`;
                  console.log("[FL_WF_Agent] Using fallback node definitions with basic nodes");
              }
              
              // Use hardcoded system prompt with node definitions
              const systemPrompt = `${SYSTEM_PROMPT}\n\nAvailable ComfyUI Nodes:\n${nodeDefinitions}\n\nUse these nodes when generating code - they represent the actual nodes available in the system.`;
              
              // List available models first
              const models = await listGeminiModels(apiKey);
              const modelName = GEMINI_MODEL;
              
              if (!models.some(m => m.name.includes(modelName))) {
                  throw new Error(`Model ${modelName} not found. Available models: ${models.map(m => m.name).join(', ')}`);
              }

              const requestBody = {
                  contents: [{
                      parts: [{
                          text: `${systemPrompt}\nUser prompt: ${prompt}`
                      }]
                  }]
              };

              const response = await fetch(`${GEMINI_BASE_URL}/${GEMINI_API_VERSION}/models/${modelName}:generateContent?key=${apiKey}`, {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/json',
                  },
                  body: JSON.stringify(requestBody)
              });

              if (!response.ok) {
                  const errorData = await response.json();
                  console.error("[FL_WF_Agent] API Error Response:", errorData);
                  throw new Error(`Gemini API Error (${response.status}): ${JSON.stringify(errorData)}`);
              }

              const data = await response.json();
              
              if (!data.candidates?.[0]?.content?.parts?.[0]?.text) {
                  console.error("[FL_WF_Agent] Unexpected API response format:", data);
                  throw new Error("Invalid response format from Gemini API");
              }

              return data.candidates[0].content.parts[0].text.trim();

          } catch (error) {
              console.error("[FL_WF_Agent] API Call Error:", error);
              throw error;
          }
      }

      // Add code generation button
      const generateCodeWidget = node.addWidget("button", "🤖 Generate Code", null, async () => {
          try {
              const promptWidget = node.widgets?.find(e => e.name === "code_prompt");
              const apiKeyWidget = node.widgets?.find(e => e.name === "api_key");
              const jsWidget = node.widgets?.find(e => e.name === "javascript");

              if (!promptWidget?.value?.trim()) {
                  app.ui.dialog.show("Error", "Please enter a code prompt");
                  return;
              }

              if (!apiKeyWidget?.value?.trim()) {
                  app.ui.dialog.show("Error", "Please enter a valid Gemini API key");
                  return;
              }

              app.ui.dialog.show("Status", "Generating code...");
              
              const generatedCode = await callGeminiAPI(promptWidget.value, apiKeyWidget.value);
              
              if (generatedCode && jsWidget) {
                  // Sanitize the generated code before setting it
                  jsWidget.value = sanitizeGeminiResponse(generatedCode);
                  node.setDirtyCanvas(true);
                  app.ui.dialog.show("Success", "Code generated successfully!");
              } else {
                  throw new Error("Failed to update code widget");
              }

          } catch (error) {
              console.error("[FL_WF_Agent] Code generation error:", error);
              app.ui.dialog.show("Error", `Failed to generate code: ${error.message}`);
          }
      }, {
          serialize: false
      });
      generateCodeWidget.computeSize = () => [140, 26];

      // Add scan nodes button
      const scanNodesWidget = node.addWidget("button", "🔍 Scan Nodes", null, async () => {
          try {
              console.log("[FL_WF_Agent] Starting node scan...");
              app.ui.dialog.show("Status", "Starting node scan... This may take a moment.");
              
              // Log current paths for debugging
              console.log("[FL_WF_Agent] Current location:", window.location.href);
              console.log("[FL_WF_Agent] Expected cache path:", `${window.location.origin}/custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt`);
              console.log("[FL_WF_Agent] Relative cache path:", '/custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt');
              
              // Debug server directory structure
              console.log("[FL_WF_Agent] Checking current server file structure...");
              
              // Set scan_nodes widget to true to trigger scan
              const scanWidget = node.widgets?.find(e => e.name === "scan_nodes");
              if (!scanWidget) {
                  throw new Error("Scan nodes widget not found");
              }
              
              // Set the value and trigger execution
              scanWidget.value = true;
              
              // Queue the node for execution
              if (node.triggerSlot) {
                  node.triggerSlot(0);
              } else {
                  node.onExecuted?.({}); // Fallback execution
              }
              
              app.graph.setDirtyCanvas(true);
              
              // Try to detect if file becomes available after scan completes
              setTimeout(async () => {
                  console.log("[FL_WF_Agent] Checking for cache file after scan...");
                  try {
                      const testResponse = await fetch('/custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt');
                      console.log("[FL_WF_Agent] Cache file check result:", testResponse.status);
                      if (testResponse.ok) {
                          console.log("[FL_WF_Agent] Cache file is now accessible!");
                          const textSample = await testResponse.text();
                          console.log("[FL_WF_Agent] First 100 chars:", textSample.substring(0, 100));
                      } else {
                          console.log("[FL_WF_Agent] Cache file still not accessible after scan!");
                      }
                  } catch (e) {
                      console.log("[FL_WF_Agent] Error checking cache file:", e);
                  }
              }, 5000);
              
          } catch (error) {
              console.error("[FL_WF_Agent] Scan error:", error);
              app.ui.dialog.show("Error", `Failed to start scan: ${error.message}`);
              
              // Reset scan widget on error
              const scanWidget = node.widgets?.find(e => e.name === "scan_nodes");
              if (scanWidget) {
                  scanWidget.value = false;
              }
          }
      }, {
          serialize: false
      });
      scanNodesWidget.computeSize = () => [140, 26];

      // Add handler for scan feedback
      node.onExecuted = function(message) {
          console.log("[FL_WF_Agent] Node executed, message:", message);
          
          if (message?.ui?.scan_feedback) {
              const feedback = message.ui.scan_feedback;
              console.log("[FL_WF_Agent] Received scan feedback:", feedback);
              
              // Display detailed directory information
              if (feedback.directory_info) {
                  console.log("[FL_WF_Agent] Directory info:", feedback.directory_info);
              }
              
              if (feedback.success) {
                  // Try multiple paths to check if the file is accessible
                  const pathsToTry = [
                      '/custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt',
                      './custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt',
                      `${window.location.origin}/custom_nodes/ComfyUI_Fill-Nodes/web/nodes/node_definitions.txt`
                  ];
                  
                  console.log("[FL_WF_Agent] Trying multiple paths to locate the file:");
                  
                  pathsToTry.forEach((path, index) => {
                      fetch(path)
                          .then(response => {
                              console.log(`[FL_WF_Agent] Path ${index+1} (${path}) check result:`, response.status);
                              if (response.ok) {
                                  return response.text().then(text => {
                                      console.log(`[FL_WF_Agent] File at path ${index+1} exists, content length:`, text.length);
                                      console.log(`[FL_WF_Agent] First 100 chars:`, text.substring(0, 100));
                                  });
                              }
                              return `File not found at path ${index+1}`;
                          })
                          .catch(error => {
                              console.error(`[FL_WF_Agent] Error checking path ${index+1}:`, error);
                          });
                  });
                  
                  app.ui.dialog.show("Success", feedback.message);
                  if (feedback.stdout) {
                      console.log("[FL_WF_Agent] Scanner output:", feedback.stdout);
                  }
              } else {
                  app.ui.dialog.show("Error", feedback.message);
                  if (feedback.stderr) {
                      console.error("[FL_WF_Agent] Scanner error:", feedback.stderr);
                  }
              }
              
              // Reset scan_nodes widget
              const scanWidget = this.widgets?.find(e => e.name === "scan_nodes");
              if (scanWidget) {
                  scanWidget.value = false;
              }
          }
      };

      // Add execute button
      const execButton = node.addWidget("button", "Execute", null, () => {}, {
        serialize: false
      });
      execButton.computeSize = () => [140, 26];
      execButton.callback = () => execNode(node, []);

      node.exec = () => execNode(node, []);
    }
  }
});
