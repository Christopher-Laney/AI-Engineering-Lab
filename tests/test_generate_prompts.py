import tempfile
import unittest
from pathlib import Path

from scripts import generate_prompts


class GeneratePromptsTests(unittest.TestCase):
    def test_load_templates_accepts_json_list(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "templates.json"
            config_path.write_text('["Explain {topic}."]', encoding="utf-8")

            templates = generate_prompts.load_templates(str(config_path))

        self.assertEqual(templates, ["Explain {topic}."])

    def test_fill_template_uses_defaults_for_optional_fields(self):
        prompt = generate_prompts.fill_template(
            "Explain {topic} to a {audience} in a {style} style for {domain}.",
            {"topic": "prompt engineering"},
        )

        self.assertEqual(
            prompt,
            "Explain prompt engineering to a general reader in a concise style for general.",
        )

    def test_build_permutations_skips_blank_topics(self):
        permutations = list(
            generate_prompts.build_permutations(
                {"topic": "   "},
                styles=["concise"],
                audiences=["engineer"],
                levels=["easy"],
            )
        )

        self.assertEqual(permutations, [])

    def test_dedupe_keep_order_preserves_first_seen_value(self):
        values = generate_prompts.dedupe_keep_order(["a", "b", "a", "c", "b"])

        self.assertEqual(values, ["a", "b", "c"])
