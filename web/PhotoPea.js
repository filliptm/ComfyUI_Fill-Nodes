import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";

class LJPhotopeaEditorDialog {
  static instance = null;

  static getInstance() {
    if(!LJPhotopeaEditorDialog.instance) {
      LJPhotopeaEditorDialog.instance = new LJPhotopeaEditorDialog();
    }
    return LJPhotopeaEditorDialog.instance;
  }

  constructor() {
    this.element = $el("div.photopea-editor", {
      style: {
        position: "fixed",
        top: "0",
        left: "0",
        width: "100vw",
        height: "100vh",
        backgroundColor: "rgba(50, 0, 80, 0.8)",
        display: "none",
        zIndex: "9999"
      }
    }, [
      this.createCloseButton(),
      this.createUIContainer(),
    ]);

    document.body.appendChild(this.element);
    this.addKeyboardListener();
  }

  createCloseButton() {
    return $el("button.close-button", {
      textContent: "×",
      onclick: () => this.hide(),
      style: {
        position: "absolute",
        top: "10px",
        right: "10px",
        fontSize: "24px",
        background: "none",
        border: "none",
        color: "white",
        cursor: "pointer",
        zIndex: "10000"
      }
    });
  }

  createUIContainer() {
    this.uiContainer = $el("div.ui-container", {
      style: {
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden"
      }
    });
    return this.uiContainer;
  }

  addKeyboardListener() {
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.element.style.display !== "none") {
        this.hide();
      }
    });
  }

  show() {
    this.element.style.display = "block";
    this.createIframe();
  }

  hide() {
    this.element.style.display = "none";
    if (this.iframe) {
      this.uiContainer.removeChild(this.iframe);
      this.iframe = null;
    }
  }

  createIframe() {
    this.iframe = $el("iframe", {
      src: `https://www.photopea.com/`,
      style: {
        width: "100%",
        height: "100%",
        border: "none",
      },
    });

    this.uiContainer.appendChild(this.iframe);

    this.iframe.onload = () => {
      const target_image_path = ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src;
      this.imageToBase64(target_image_path, (dataURL) => {
        this.postMessageToPhotopea(`app.open("${dataURL}", null, false);`, "*");
      });
    };
  }

  async save() {
    const saveMessage = 'app.activeDocument.saveToOE("png");';
    const [payload, done] = await this.postMessageToPhotopea(saveMessage);
    const file = new Blob([payload], { type: "image/png" });
    const body = new FormData();

    const filename = "clipspace-photopea-" + performance.now() + ".png";

    if(ComfyApp.clipspace.widgets) {
      const index = ComfyApp.clipspace.widgets.findIndex(obj => obj.name === 'image');
      if(index >= 0)
        ComfyApp.clipspace.widgets[index].value = `photopea/${filename} [input]`;
    }

    body.append("image", file, filename);
    body.append("subfolder", "photopea");
    await this.uploadFile(body);

    ComfyApp.onClipspaceEditorSave();
    this.hide();
  }

  async postMessageToPhotopea(message) {
    var request = new Promise(function (resolve, reject) {
        var responses = [];
        var photopeaMessageHandle = function (response) {
            responses.push(response.data);
            if (response.data == "done") {
                window.removeEventListener("message", photopeaMessageHandle);
                resolve(responses)
            }
        };
        window.addEventListener("message", photopeaMessageHandle);
    });
    this.iframe.contentWindow.postMessage(message, "*");
    return await request;
  }

  imageToBase64(url, callback) {
    fetch(url)
      .then((response) => response.blob())
      .then((blob) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
          const base64String = reader.result;
          callback(base64String);
        };
      });
  }

  async uploadFile(formData) {
    try {
      const resp = await api.fetchApi('/upload/image', {
        method: 'POST',
        body: formData
      })
      if (resp.status === 200) {
        const data = await resp.json();
        ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']] = new Image();
        ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src = `view?filename=${data.name}&subfolder=${data.subfolder}&type=${data.type}`;
      } else {
        alert(resp.status + " - " + resp.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  }
}

function addMenuHandler(nodeType, cb) {
  const getOpts = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function () {
    const r = getOpts.apply(this, arguments);
    cb.apply(this, arguments);
    return r;
  };
}

app.registerExtension({
  name: "Comfy.LJ.PhotopeaEditor",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (Array.isArray(nodeData.output) && (nodeData.output.includes("MASK") || nodeData.output.includes("IMAGE"))) {
      addMenuHandler(nodeType, function (_, options) {
        options.unshift({
          content: "FL PhotoPea",
          callback: () => {
            ComfyApp.copyToClipspace(this);
            ComfyApp.clipspace_return_node = this;

            let dlg = LJPhotopeaEditorDialog.getInstance();
            dlg.show();
          },
        });
      });
    }
  },
  async setup() {
    const photopeaEditor = LJPhotopeaEditorDialog.getInstance();

    // Add save button
    const saveButton = $el("button", {
      textContent: "Save",
      onclick: () => photopeaEditor.save(),
      style: {
        position: "absolute",
        bottom: "10px",
        right: "10px",
        zIndex: "10001",
        padding: "5px 15px",
        fontSize: "14px",
        backgroundColor: "#8a2be2",
        color: "white",
        border: "none",
        borderRadius: "5px",
        cursor: "pointer",
      }
    });
    photopeaEditor.element.appendChild(saveButton);
  }
});