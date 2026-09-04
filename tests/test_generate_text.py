import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).parents[1]
MODULE_PATH = ROOT / "nodes" / "prompting" / "FL_GenerateText.py"
SPEC = importlib.util.spec_from_file_location("fl_generate_text", MODULE_PATH)
generate_text = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_text)


class FakeClip:
    def __init__(self):
        self.tokenize_call = None
        self.generate_call = None

    def tokenize(self, text, **kwargs):
        self.tokenize_call = (text, kwargs)
        return {"qwen3vl_8b": [[(1, 1.0)]]}

    def generate(self, tokens, **kwargs):
        self.generate_call = (tokens, kwargs)
        return [101, 102]

    def decode(self, token_ids):
        self.decoded_ids = token_ids
        return "generated answer"


class GenerateTextTests(unittest.TestCase):
    def test_formats_qwen_system_and_user_roles(self):
        formatted = generate_text.format_qwen_chat("System {rules}", "User request")

        self.assertEqual(
            formatted,
            "<|im_start|>system\nSystem {rules}<|im_end|>\n"
            "<|im_start|>user\nUser request<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_thinking_mode_does_not_prime_an_empty_thought(self):
        formatted = generate_text.format_qwen_chat("System", "User", thinking=True)

        self.assertTrue(formatted.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", formatted)

    def test_generates_with_serialized_chat_and_sampling_controls(self):
        clip = FakeClip()
        result = generate_text.FL_GenerateText().generate(
            clip=clip,
            system_prompt="Follow the system role.",
            prompt="Write a prompt.",
            max_length=384,
            sampling="on",
            temperature=0.6,
            top_k=48,
            top_p=0.9,
            min_p=0.04,
            repetition_penalty=1.1,
            presence_penalty=0.2,
            seed=17,
            thinking=False,
        )

        tokenized_text, tokenize_kwargs = clip.tokenize_call
        self.assertIn("<|im_start|>system\nFollow the system role.<|im_end|>", tokenized_text)
        self.assertIn("<|im_start|>user\nWrite a prompt.<|im_end|>", tokenized_text)
        self.assertEqual(tokenize_kwargs, {"skip_template": True, "min_length": 1})
        self.assertEqual(clip.generate_call[0], {"qwen3vl_8b": [[(1, 1.0)]]})
        self.assertEqual(
            clip.generate_call[1],
            {
                "do_sample": True,
                "max_length": 384,
                "temperature": 0.6,
                "top_k": 48,
                "top_p": 0.9,
                "min_p": 0.04,
                "repetition_penalty": 1.1,
                "presence_penalty": 0.2,
                "seed": 17,
            },
        )
        self.assertEqual(clip.decoded_ids, [101, 102])
        self.assertEqual(result, {"ui": {"generated_text": ["generated answer"]}, "result": ("generated answer",)})

    def test_sampling_can_be_disabled(self):
        clip = FakeClip()
        generate_text.FL_GenerateText().generate(
            clip, "System", "User", 32, "off", 0.7, 64, 0.95, 0.05, 1.05, 0.0, 0, False,
        )

        self.assertFalse(clip.generate_call[1]["do_sample"])

    def test_node_contract(self):
        inputs = generate_text.FL_GenerateText.INPUT_TYPES()["required"]

        self.assertEqual(inputs["clip"][0], "CLIP")
        self.assertEqual(inputs["system_prompt"][0], "STRING")
        self.assertEqual(inputs["prompt"][0], "STRING")
        self.assertEqual(generate_text.FL_GenerateText.RETURN_TYPES, ("STRING",))
        self.assertEqual(generate_text.FL_GenerateText.RETURN_NAMES, ("generated_text",))
        self.assertTrue(generate_text.FL_GenerateText.OUTPUT_NODE)


class GenerateTextFrontendTests(unittest.TestCase):
    def test_custom_editor_preserves_backend_widget_ownership(self):
        script = (ROOT / "web" / "nodes" / "prompting" / "FL_GenerateText.js").read_text(encoding="utf-8")

        for behavior in (
            'comfyClass !== NODE_CLASS',
            'data-field="system_prompt"',
            'data-field="prompt"',
            "setWidgetValue(this.node, this.widgets[name], value)",
            'serialize: false',
            'data-action="apply"',
            'data-action="apply-generate"',
            'data-action="cancel"',
            'data-action="copy"',
            'data-action="clear"',
            'data-action="generate"',
            'container-name: flgt-node',
            'class="flgt-workspace"',
            'grid-template-columns: minmax(210px, .8fr)',
            'DEFAULT_NODE_SIZE = [1080, 520]',
            'UI_FIELDS = [...BACKEND_FIELDS, "control_after_generate"]',
            'await app.queuePrompt(0, 1)',
            'this.resizeObserver = new ResizeObserver',
            "panel.showOutput(executionText(message))",
            'api.addEventListener("executing"',
            'api.addEventListener("execution_error"',
            "removeInstance(this)",
        ):
            with self.subTest(behavior=behavior):
                self.assertIn(behavior, script)

        self.assertNotIn("localStorage", script)
        self.assertNotIn("fetch(", script)
        self.assertNotIn('widget.type = "converted-widget"', script)


if __name__ == "__main__":
    unittest.main()
