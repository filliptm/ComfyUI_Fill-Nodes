import { app } from "/scripts/app.js";
import {
  getAvifMetadata,
  getPngMetadata,
  getWebpMetadata,
} from "/scripts/pnginfo.js";

app.registerExtension({
  name: "Local.LoadImageDropFix.FillNodesPath",
  setup() {
    console.warn("[LoadImageDropFix] extension loaded v2026-07-20-workflow-sanitize");
    let lastDragOverLogKey = null;
    let patchedDragOverNode = null;
    let lastPatchedDrag = null;

    const findNodeById = (graph, id) => {
      if (!graph || id == null) return null;

      const node = graph.getNodeById?.(id);
      if (node) return node;

      for (const subgraph of graph.subgraphs?.values?.() ?? []) {
        const subgraphNode = findNodeById(subgraph, id);
        if (subgraphNode) return subgraphNode;
      }

      return null;
    };

    const getNodeFromDomEvent = (event) => {
      const element = event.target?.closest?.("[data-node-id]");
      if (!element) return null;

      return findNodeById(app.rootGraph ?? app.canvas?.graph, element.dataset.nodeId);
    };

    const getNodeUnderEvent = (event) => {
      const domNode = getNodeFromDomEvent(event);
      if (domNode) return domNode;

      const canvas = app.canvas;
      const graph = canvas?.graph;
      if (!canvas || !graph) return null;

      canvas.adjustMouseEvent?.(event);
      return graph.getNodeOnPos?.(event.canvasX, event.canvasY) ?? null;
    };

    const isLoadImageNode = (node) => {
      return node?.type === "LoadImage" || node?.type === "LoadImageMask";
    };

    const isImageFile = (file) => file?.type?.startsWith?.("image/");

    const isWorkflowFile = (file) => {
      return (
        file?.type === "application/json" ||
        file?.name?.toLowerCase?.().endsWith(".json")
      );
    };

    const isWorkflowMetadataCandidate = (file) => {
      const name = file?.name?.toLowerCase?.() ?? "";
      return (
        isWorkflowFile(file) ||
        isImageFile(file) ||
        file?.type === "video/mp4" ||
        file?.type === "video/quicktime" ||
        name.endsWith(".mp4") ||
        name.endsWith(".mov") ||
        name.endsWith(".m4v")
      );
    };

    const getFiles = (event) => {
      return Array.from(event.dataTransfer?.files ?? []);
    };

    const getItems = (event) => {
      return Array.from(event.dataTransfer?.items ?? []);
    };

    const isDraggingFiles = (event) => {
      return (
        getItems(event).some((item) => item.kind === "file") ||
        Array.from(event.dataTransfer?.types ?? []).includes("Files")
      );
    };

    const isDraggingImages = (event) => {
      const items = getItems(event).filter((item) => item.kind === "file");
      if (!items.length) return isDraggingFiles(event);
      return items.some(
        (item) => !item.type || item.type.startsWith("image/"),
      );
    };

    const hasWorkflowMetadata = async (file) => {
      try {
        const workflowData = await getWorkflowData(file);
        return !!(
          workflowData?.workflow ||
          workflowData?.Workflow ||
          workflowData?.prompt ||
          workflowData?.Prompt ||
          workflowData?.parameters ||
          workflowData?.templates
        );
      } catch (error) {
        console.warn("[LoadImageDropFix] workflow metadata check failed", error);
        return false;
      }
    };

    const getImageMetadata = async (file) => {
      let metadata = {};
      try {
        if (file.type === "image/png") {
          metadata = await getPngMetadata(file);
        } else if (file.type === "image/webp") {
          metadata = await getWebpMetadata(file);
        } else if (file.type === "image/avif") {
          metadata = await getAvifMetadata(file);
        }
      } catch (error) {
        console.warn("[LoadImageDropFix] metadata read failed", error);
      }

      return metadata;
    };

    const readTextFile = (file) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ""));
        reader.onerror = () => reject(reader.error ?? new Error("read failed"));
        reader.onabort = () => reject(new Error("read aborted"));
        reader.readAsText(file);
      });
    };

    const getWorkflowDataFromServer = async (file) => {
      if (!isWorkflowMetadataCandidate(file)) return {};

      const body = new FormData();
      body.append("file", file);

      return await new Promise((resolve) => {
        const request = new XMLHttpRequest();
        request.open("POST", "/load_image_drop_fix/workflow_metadata");
        request.onload = () => {
          if (request.status < 200 || request.status >= 300) {
            resolve({});
            return;
          }

          try {
            resolve(JSON.parse(request.responseText || "{}"));
          } catch {
            resolve({});
          }
        };
        request.onerror = () => resolve({});
        request.onabort = () => resolve({});
        request.send(body);
      });
    };

    const parseJson = (value) => {
      if (!value) return undefined;
      if (typeof value === "object") return value;
      if (typeof value !== "string") return undefined;
      return JSON.parse(value);
    };

    const cloneJsonData = (value) => {
      if (!value || typeof value !== "object") return value;
      try {
        return structuredClone(value);
      } catch {
        return JSON.parse(JSON.stringify(value));
      }
    };

    const stripPreviewUrl = (value) => {
      if (typeof value !== "string") return value;
      return value.includes("rgthree.compare._temp_") ? "" : value;
    };

    const normalizeMediaReference = (value) => {
      if (!value || typeof value !== "object" || Array.isArray(value)) return 0;

      let changed = 0;
      const filename = value.filename;
      const subfolder = value.subfolder;
      if (typeof filename === "string" && typeof subfolder === "string" && subfolder) {
        const normalizedSubfolder = subfolder.replace(/^\/+|\/+$/g, "");
        const prefix = `${normalizedSubfolder}/`;
        if (filename.startsWith(prefix)) {
          value.filename = filename.slice(prefix.length);
          changed++;
        }
      }

      return changed;
    };

    const sanitizeWorkflowForDrop = (workflow) => {
      const sanitized = cloneJsonData(workflow);
      const stats = { normalizedMediaRefs: 0, clearedTempPreviews: 0 };

      const walk = (value) => {
        if (!value || typeof value !== "object") return value;

        stats.normalizedMediaRefs += normalizeMediaReference(value);

        if (Array.isArray(value)) {
          for (let i = value.length - 1; i >= 0; i--) {
            if (typeof value[i] === "string" && value[i].includes("rgthree.compare._temp_")) {
              value.splice(i, 1);
              stats.clearedTempPreviews++;
            } else {
              const next = walk(value[i]);
              if (next !== value[i]) value[i] = next;
            }
          }
          return value;
        }

        for (const [key, child] of Object.entries(value)) {
          if (typeof child === "string") {
            const stripped = stripPreviewUrl(child);
            if (stripped !== child) {
              value[key] = stripped;
              stats.clearedTempPreviews++;
            }
          } else {
            const next = walk(child);
            if (next !== child) value[key] = next;
          }
        }

        return value;
      };

      walk(sanitized);

      for (const node of sanitized?.nodes ?? []) {
        if (node?.type === "Image Comparer (rgthree)" && Array.isArray(node.widgets_values)) {
          for (let i = 0; i < node.widgets_values.length; i++) {
            const widgetValue = node.widgets_values[i];
            const asJson = JSON.stringify(widgetValue ?? "");
            if (asJson.includes("rgthree.compare._temp_")) {
              node.widgets_values[i] = [];
              stats.clearedTempPreviews++;
            }
          }
        }
      }

      if (stats.normalizedMediaRefs || stats.clearedTempPreviews) {
        console.warn("[LoadImageDropFix] sanitized dropped workflow", stats);
      }

      return sanitized;
    };

    const focusLoadedWorkflow = () => {
      requestAnimationFrame(() => {
        const canvas = app.canvas;
        const graph = canvas?.graph ?? app.rootGraph;
        const firstNode = graph?.nodes?.[0];
        if (firstNode && canvas?.centerOnNode) {
          canvas.centerOnNode(firstNode);
        }
        canvas?.setDirty?.(true, true);
      });
    };

    const isApiJson = (data) => {
      return (
        data &&
        typeof data === "object" &&
        !Array.isArray(data) &&
        Object.keys(data).length > 0 &&
        Object.values(data).every((node) => node?.class_type)
      );
    };

    const getWorkflowData = async (file) => {
      const serverWorkflowData = await getWorkflowDataFromServer(file);
      if (
        serverWorkflowData?.workflow ||
        serverWorkflowData?.prompt ||
        serverWorkflowData?.parameters ||
        serverWorkflowData?.templates
      ) {
        return serverWorkflowData;
      }

      if (isWorkflowFile(file)) {
        const data = parseJson(await readTextFile(file));
        if (data?.workflow || data?.Workflow || data?.prompt || data?.Prompt) {
          return {
            workflow: data.workflow ?? data.Workflow,
            prompt: data.prompt ?? data.Prompt,
            parameters: data.parameters,
            templates: data.templates,
          };
        }
        if (data?.templates) return { templates: data.templates };
        if (isApiJson(data)) return { prompt: data };
        return { workflow: data };
      }

      if (!isImageFile(file)) return {};

      const metadata = await getImageMetadata(file);
      return !!(
        metadata?.workflow ||
        metadata?.Workflow ||
        metadata?.prompt ||
        metadata?.Prompt ||
        metadata?.parameters ||
        metadata?.parametersText ||
        metadata?.templates
      )
        ? {
            workflow: metadata.workflow ?? metadata.Workflow,
            prompt: metadata.prompt ?? metadata.Prompt,
            parameters: metadata.parameters ?? metadata.parametersText,
            templates: metadata.templates,
          }
        : {};
    };

    const hasWorkflowData = (workflowData) => {
      return !!(
        workflowData?.workflow ||
        workflowData?.prompt ||
        workflowData?.parameters ||
        workflowData?.templates
      );
    };

    const loadWorkflowFromFile = async (file, workflowData) => {
      workflowData ??= await getWorkflowData(file);
      const fileName = file.name.replace(/\.\w+$/, "");

      if (workflowData?.templates && app.loadTemplateData) {
        app.loadTemplateData({ templates: workflowData.templates });
        return true;
      }

      if (workflowData?.workflow) {
        const workflow = parseJson(workflowData.workflow);
        if (workflow && typeof workflow === "object" && !Array.isArray(workflow)) {
          const sanitizedWorkflow = sanitizeWorkflowForDrop(workflow);
          console.warn("[LoadImageDropFix] workflow data parsed", {
            file: file.name,
            nodes: sanitizedWorkflow.nodes?.length ?? 0,
            version: sanitizedWorkflow.version,
          });
          await app.loadGraphData(sanitizedWorkflow, true, true, fileName, {
            openSource: "file_drop",
            deferWarnings: true,
            skipAssetScans: true,
            silentAssetErrors: true,
          });
          console.warn("[LoadImageDropFix] workflow loaded", file.name);
          focusLoadedWorkflow();
          return true;
        }
      }

      if (workflowData?.prompt && app.loadApiJson) {
        const prompt = parseJson(workflowData.prompt);
        if (prompt) {
          console.warn("[LoadImageDropFix] api json parsed", {
            file: file.name,
            nodes: Object.keys(prompt).length,
          });
          await app.loadApiJson(prompt, fileName);
          console.warn("[LoadImageDropFix] api json loaded", file.name);
          focusLoadedWorkflow();
          return true;
        }
      }

      if (workflowData?.prompt) {
        console.warn("[LoadImageDropFix] api json fallback to native loader", file.name);
        await app.handleFile(file, "file_drop", { deferWarnings: true });
        focusLoadedWorkflow();
        return true;
      }

      if (workflowData?.parameters) {
        await app.handleFile(file, "file_drop", { deferWarnings: true });
        return true;
      }

      return false;
    };

    const handleWorkflowFile = async (file, workflowData) => {
      try {
        if (await loadWorkflowFromFile(file, workflowData)) return true;
      } catch (error) {
        console.warn("[LoadImageDropFix] direct workflow load failed", error);
      }

      const originalLoadGraphData = app.loadGraphData;
      try {
        app.loadGraphData = function (
          graphData,
          clean,
          restoreView,
          workflow,
          options = {},
        ) {
          return originalLoadGraphData.call(
            this,
            graphData,
            clean,
            restoreView,
            workflow,
            {
              ...options,
              deferWarnings: true,
              skipAssetScans: true,
              silentAssetErrors: true,
            },
          );
        };
        await app.handleFile(file, "file_drop", { deferWarnings: true });
        return true;
      } catch (error) {
        console.warn("[LoadImageDropFix] native workflow load failed", error);
        return false;
      } finally {
        app.loadGraphData = originalLoadGraphData;
      }
    };

    const uploadImage = async (file) => {
      const body = new FormData();
      body.append("image", file);
      body.append("type", "input");
      body.append("overwrite", "true");

      const data = await new Promise((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open("POST", "/upload/image");
        request.onload = () => {
          if (request.status < 200 || request.status >= 300) {
            reject(new Error(`${request.status} ${request.statusText}`));
            return;
          }

          try {
            resolve(JSON.parse(request.responseText));
          } catch (error) {
            reject(error);
          }
        };
        request.onerror = () => reject(new Error("image upload failed"));
        request.onabort = () => reject(new Error("image upload aborted"));
        request.send(body);
      });
      return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
    };

    const addToComboValues = (widget, value) => {
      const values = widget?.options?.values;
      if (!Array.isArray(values)) return;

      const valuesToAdd = Array.isArray(value) ? value : [value];
      for (const item of valuesToAdd) {
        if (!values.includes(item)) values.push(item);
      }
    };

    const setNodeImage = (node, value) => {
      const widget = node?.widgets?.find?.((widget) => widget.name === "image");
      if (!widget) {
        throw new Error(`LoadImage node ${node?.id ?? ""} has no image widget`);
      }

      const oldValue = widget.value;
      addToComboValues(widget, value);
      widget.value = value;
      widget.callback?.(value);
      node.onWidgetChanged?.(widget.name, value, oldValue, widget);
      node.imgs = undefined;
      node.graph?.setDirtyCanvas?.(true, true);
      app.canvas?.setDirty?.(true, true);
    };

    const createLoadImageNode = (event) => {
      const liteGraph = window.LiteGraph;
      const graph = app.canvas?.graph ?? app.rootGraph;
      if (!liteGraph?.createNode || !graph?.add) {
        throw new Error("LiteGraph is not ready");
      }

      app.canvas?.adjustMouseEvent?.(event);
      const node = liteGraph.createNode("LoadImage");
      if (!node) throw new Error("Failed to create LoadImage node");

      node.pos = [
        event.canvasX ?? app.canvas?.graph_mouse?.[0] ?? 0,
        event.canvasY ?? app.canvas?.graph_mouse?.[1] ?? 0,
      ];
      graph.add(node);
      return node;
    };

    const uploadImagesToNode = async (files, node, event) => {
      const imageFiles = files.filter(isImageFile);
      if (!imageFiles.length) return false;

      node ??= createLoadImageNode(event);
      node.isUploading = true;
      try {
        node.imgs = undefined;
        node.graph?.setDirtyCanvas?.(true, true);

        const paths = [];
        for (const file of imageFiles) {
          paths.push(await uploadImage(file));
        }

        setNodeImage(node, paths[0]);
        return true;
      } finally {
        node.isUploading = false;
        node.graph?.setDirtyCanvas?.(true, true);
      }
    };

    const clearPatchedDragOverNode = () => {
      if (patchedDragOverNode && app.dragOverNode === patchedDragOverNode) {
        app.dragOverNode = null;
      }
      patchedDragOverNode = null;
      lastPatchedDrag = null;
    };

    const getDropNode = (event) => {
      const node = getNodeUnderEvent(event);
      if (isLoadImageNode(node)) return node;

      if (!patchedDragOverNode || !lastPatchedDrag) return null;

      const elapsed = performance.now() - lastPatchedDrag.time;
      const distance = Math.hypot(
        event.clientX - lastPatchedDrag.clientX,
        event.clientY - lastPatchedDrag.clientY,
      );

      if (elapsed < 1000 && distance < 48) return patchedDragOverNode;
      return null;
    };

    const handleDragOver = (event) => {
      const node = getNodeUnderEvent(event);
      const canDrop =
        (isLoadImageNode(node) && isDraggingImages(event)) ||
        (!node && isDraggingFiles(event));

      if (isLoadImageNode(node)) {
        const key = `${node.id}:${canDrop}`;
        if (key !== lastDragOverLogKey) {
          lastDragOverLogKey = key;
          console.warn("[LoadImageDropFix] dragover LoadImage", {
            id: node.id,
            canDrop,
          });
        }
      }

      if (!canDrop) {
        clearPatchedDragOverNode();
        return;
      }

      patchedDragOverNode = node;
      lastPatchedDrag = {
        clientX: event.clientX,
        clientY: event.clientY,
        time: performance.now(),
      };
      app.dragOverNode = node;
      event.preventDefault();
      requestAnimationFrame(() => app.canvas?.setDirty?.(false, true));
    };

    const handleDrop = async (event) => {
      const node = getDropNode(event);
      const files = getFiles(event);

      if (!files.length) {
        clearPatchedDragOverNode();
        return;
      }

      if (isLoadImageNode(node)) {
        event.preventDefault();
        event.stopPropagation();

        try {
          console.warn("[LoadImageDropFix] upload image to node", node.id);
          await uploadImagesToNode(files, node, event);
        } finally {
          clearPatchedDragOverNode();
          app.canvas?.setDirty?.(false, true);
        }
        return;
      }

      if (node) {
        clearPatchedDragOverNode();
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      try {
        app.canvas?.adjustMouseEvent?.(event);
        if (app.canvas?.graph_mouse) {
          app.canvas.graph_mouse[0] = event.canvasX;
          app.canvas.graph_mouse[1] = event.canvasY;
        }

        for (const file of files) {
          const workflowData = isWorkflowMetadataCandidate(file)
            ? await getWorkflowData(file)
            : {};

          if (hasWorkflowData(workflowData)) {
            console.warn("[LoadImageDropFix] load workflow from drop", file.name);
            await handleWorkflowFile(file, workflowData);
          } else if (isImageFile(file)) {
            console.warn("[LoadImageDropFix] create LoadImage from drop", file.name);
            await uploadImagesToNode([file], createLoadImageNode(event), event);
          } else {
            await handleWorkflowFile(file, workflowData);
          }
        }
      } catch (error) {
        console.warn("[LoadImageDropFix] drop handling failed", error);
        for (const file of files.filter(isImageFile)) {
          try {
            await uploadImagesToNode([file], createLoadImageNode(event), event);
          } catch (uploadError) {
            console.warn("[LoadImageDropFix] image fallback failed", uploadError);
          }
        }
      } finally {
        clearPatchedDragOverNode();
        app.canvas?.setDirty?.(true, true);
      }
    };

    document.addEventListener("dragover", handleDragOver, true);
    document.addEventListener(
      "drop",
      (event) => {
        handleDrop(event).catch((error) => {
          console.warn("[LoadImageDropFix] unhandled drop failure", error);
          clearPatchedDragOverNode();
          app.canvas?.setDirty?.(true, true);
        });
      },
      true,
    );
  },
});
