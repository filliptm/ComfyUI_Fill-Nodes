/**
    File: fl_code_node.js
    Project: ComfyUI_Fill-Nodes

    pythongossss to the rescue again
    original: https://github.com/pythongosssss/ComfyUI-Custom-Scripts/blob/main/web/js/betterCombos.js
*/

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { $el } from "../../../scripts/ui.js";

app.registerExtension({
    name: "fl.widget.combo+",
    init() {
        const splitBy = /\//;

        $el("style", {
            textContent: `
                .litemenu-entry:hover .pysssss-combo-image {
                    display: block;
                }
                .pysssss-combo-image {
                    display: none;
                    position: absolute;
                    left: 0;
                    top: 0;
                    transform: translate(-100%, 0);
                    width: 384px;
                    height: 384px;
                    background-size: contain;
                    background-position: top right;
                    background-repeat: no-repeat;
                    filter: brightness(65%);
                }
            `,
            parent: document.body,
        });

        function buildMenu(widget, values) {
            const lookup = {
                "": { options: [] },
            };

            // Split paths into menu structure
            for (let value of values) {
                value = String(value);
                const split = value.split(splitBy);
                let path = "";
                for (let i = 0; i < split.length; i++) {
                    const s = split[i];
                    const last = i === split.length - 1;
                    if (last) {
                        // Leaf node, manually add handler that sets the lora
                        lookup[path].options.push({
                            ...value,
                            title: s,
                            callback: () => {
                                widget.value = value;
                                widget.callback(value);
                                app.graph.setDirtyCanvas(true);
                            },
                        });
                    } else {
                        const prevPath = path;
                        path += s + splitBy;
                        if (!lookup[path]) {
                            const sub = {
                                title: s,
                                submenu: {
                                    options: [],
                                    title: s,
                                },
                            };

                            // Add to tree
                            lookup[path] = sub.submenu;
                            lookup[prevPath].options.push(sub);
                        }
                    }
                }
            }
            return lookup[""].options;
        }

        // Override COMBO widgets to patch their values
        const combo = ComfyWidgets["COMBO"];
        ComfyWidgets["COMBO"] = function (node) {
            const res = combo.apply(this, arguments);
            let value = res.widget.value;
            if (value !== 'combo+') {
                return res;
            }
            let values = res.widget.options.values;


            let menu = null;
            // Override the option values to check if we should render a menu structure
            Object.defineProperty(res.widget.options, "values", {
                get() {
                    let v = values;
                    if (!menu) {
                        // Only build the menu once
                        menu = buildMenu(res.widget, values);
                    }
                    v = menu;

                    const valuesIncludes = v.includes;
                    v.includes = function (searchElement) {
                        const includesFromMenuItem = function (item) {
                            return includesFromMenuItems(item.submenu.options)
                        }
                        const includesFromMenuItems = function (items) {
                            for (const item of items) {
                                if (includesFromMenuItem(item)) {
                                    return true;
                                }
                            }
                            return false;
                        }
                        const includes = valuesIncludes.apply(this, arguments) || includesFromMenuItems(this);
                        return includes;
                    }

                    return v;
                },
                set(v) {
                    // Options are changing (refresh) so reset the menu so it can be rebuilt if required
                    values = v;
                    menu = null;
                },
            });

            Object.defineProperty(res.widget, "value", {
                get() {
                    return value;
                },
                set(v) {
                    if (v?.submenu) {
                        // Dont allow selection of submenus
                        return;
                    }
                    value = v;
                },
            });
            const first = res.widget.options.values[0];
            const val = first?.submenu?.options?.[0] ?? undefined;
            if (val !== undefined) {
                res.widget.options.value = val.title;
            }
            return res;
        };
    },
});
