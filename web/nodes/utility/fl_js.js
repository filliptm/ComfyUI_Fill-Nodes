import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const MIN_SEED = 0;
const MAX_SEED = parseInt("0xffffffffffffffff", 16);
const STEPS_OF_SEED = 10;
const DEFAULT_MARGIN_X = 32;
const DEFAULT_MARGIN_Y = 64;
let SLEEP = true;

function getNodes(type) {
  return app.graph._nodes.filter(e => e.comfyClass === "FL_JS" &&
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
    .filter(e => e && e.comfyClass !== "FL_JS")
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
  console.log("[FL_JS] Disable system sleep");
}

const enableSleep = async function() {
  const response = await api.fetchApi(`/shinich39/event-handler/enable-sleep`, {
    method: "GET",
  });

  if (response.status !== 200) {
    throw new Error(response.statusText);
  }
  SLEEP = true;
  console.log("[FL_JS] Enable system sleep");
}

const disableScreenSaver = async function() {
  const response = await api.fetchApi(`/shinich39/event-handler/disable-screen-saver`, {
    method: "GET",
  });

  if (response.status !== 200) {
    throw new Error(response.statusText);
  }
  SLEEP = false;
  console.log("[FL_JS] Disable screen saver");
}

const enableScreenSaver = async function() {
  const response = await api.fetchApi(`/shinich39/event-handler/enable-screen-saver`, {
    method: "GET",
  });

  if (response.status !== 200) {
    throw new Error(response.statusText);
  }
  SLEEP = true;
  console.log("[FL_JS] Enable screen saver");
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
  name: "Comfy.FL_JS",
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

      console.log("[FL_JS] Event added.");
    }, 1024);
  },
  nodeCreated(node) {
    if (node.comfyClass === "FL_JS") {
      const w = node.widgets?.find(e => e.name === "javascript");
      if (w) {
        w.value = getDefaultCommand();
      }

      const b = node.addWidget("button", "Execute", null, () => {}, {
        serialize: false,
      });

      b.computeSize = () => [0, 26];
      b.callback = () => execNode(node, []);
      
      node.exec = () => execNode(node, []);
    }
  }
});
