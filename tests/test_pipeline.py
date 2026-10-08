"""Counting pipeline regression tests; no model weights or API calls are used."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from agents import context_builder
from cli import main_cli
from core import config, llm_pipeline
from rules import finalize, rule_translator
from rules.primitives import evaluate_rules_structured
from storage import store

RULES = {"derived": {}, "rules": [{"id": "agent1_count", "type": "count", "target": "widget", "exact": 1}]}


def image_result(image_id, count=1):
    return {"image_id": image_id, "image_path": f"images/{image_id}.png",
            "detections": [{"tag": f"widget{i + 1}", "bbox": [0, 0, 8, 6]} for i in range(count)]}


class CountingPipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(patch.multiple(config, RESULT_DIR=str(self.root),
            TRAIN_RESULT_DIR=str(self.root / "train"), INFER_RESULT_DIR=str(self.root / "infer"),
            RULES_DIR=str(self.root / "train"),
            AGENT_DIR=str(self.root / "agent"), FACT_MEMORY_PATH=str(self.root / "agent" / "fact.json"),
            RULE_MEMORY_PATH=str(self.root / "agent" / "rules.jsonl")))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_json_training_to_inference(self):
        refs = self.write_json(self.root / "references.json", [image_result("017"), image_result("092"), image_result("301")])
        def census_reply(*, prompt_text, tag, llm_cfg):
            self.assertIn("image_id: 1", prompt_text)
            self.assertIn("image_id: 2", prompt_text)
            self.assertIn("image_id: 3", prompt_text)
            return ({"stable": ["widget"], "always_present": [], "sometimes_present": []},
                    [{"rule_line": "each sample must have 1 widget", "agent": tag}], "mock reply")
        with patch.object(llm_pipeline, "request_agent_output", side_effect=census_reply) as census, \
             patch.object(llm_pipeline, "_call_llm", return_value=json.dumps(RULES)) as compiler, \
             patch.object(main_cli, "_build_llm_cfg", return_value={}), \
             patch.object(sys, "argv", ["cli", "--mode", "train", "--train-detections-json", str(refs)]):
            main_cli.main()
        self.assertEqual((census.call_count, compiler.call_count), (1, 1))
        output = Path(config.TRAIN_RESULT_DIR)
        cfg = json.loads((output / "rules.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg, RULES)
        stored = json.loads((output / "store.json").read_text(encoding="utf-8"))
        self.assertEqual(len(stored["image_ids"]), 3)
        self.assertFalse((output / "train_sample_selection.json").exists())
        self.assertFalse((output / "train_timing.json").exists())
        self.assertEqual({p.stem for p in (output / "agent" / "prompt").glob("*.txt")}, {"agent1_b0", "translate_agent1"})
        infer_dir = Path(config.INFER_RESULT_DIR)
        queries = self.write_json(infer_dir / "detections.json", [image_result("003"), image_result("004", 0)])
        result = main_cli._run_infer(str(queries), cfg)
        self.assertEqual((result["normal"], result["abnormal"]), (1, 1))
        summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        self.assertEqual([i["status"] for i in summary["images"]], ["OK", "ABNORMAL"])
        self.assertIn("agent1_count", summary["images"][1]["issues"][0])

    def test_store_counts_renumbered_instances(self):
        data = store.build_store([{"image_id": "normal", "detections": [{"tag": "widget7"}, {"tag": "widget19"}]}])
        self.assertEqual([r["num"] for r in store.get_labels(data, "1")], ["1", "2"])
        self.assertIn("base_class: widget, num: 2", context_builder.build_context("agent1", data, {}, None))

    def test_exact_min_max_and_subtype(self):
        detections = [{"tag": "red part3"}, {"tag": "blue part9"}]
        rules = [{"id": "exact", "type": "count", "target": "part", "exact": 2},
                 {"id": "min", "type": "count", "target": "part", "min": 1},
                 {"id": "max", "type": "count", "target": "part", "max": 2},
                 {"id": "red", "type": "count", "target": "part", "filter": "subtype=='red'", "exact": 1}]
        self.assertTrue(evaluate_rules_structured(detections, {"rules": rules})["overall_pass"])
        failed = evaluate_rules_structured([], {"rules": rules})
        self.assertEqual({r["rule_id"] for r in failed["failed_rules"]}, {"exact", "min", "red"})

    def test_counting_or_implication(self):
        cfg = {"rules": [{"id": "or", "type": "imply",
               "if": [{"id": "if", "type": "count", "target": "widget", "max": 0}],
               "then": [{"id": "then", "type": "count", "target": "component", "exact": 1}]}]}
        for detections, expected in [([{"tag": "widget1"}], True), ([{"tag": "component1"}], True), ([], False)]:
            self.assertEqual(evaluate_rules_structured(detections, cfg)["overall_pass"], expected)

    def test_removed_types_rejected_including_nested_conditions(self):
        for rule_type in ["diag_range", "rel", "same_bucket"]:
            removed = {"id": "removed", "type": rule_type, "target": "widget"}
            for rules in [[removed], [{"id": "nested", "type": "imply", "if": [removed], "then": []}]]:
                with self.subTest(rule_type=rule_type):
                    result = evaluate_rules_structured([], {"rules": rules})
                    self.assertFalse(result["overall_pass"])
                    self.assertEqual(result["failed_rules"][0]["rule_id"], "config")
        invalid = {"rules": [{"id": "bad", "type": "imply", "if": [{"type": "count"}], "then": []}]}
        self.assertEqual(evaluate_rules_structured([], invalid)["failed_rules"][0]["rule_id"], "config")

    def test_removed_derived_features_rejected(self):
        result = evaluate_rules_structured([], {"rules": [], "derived": {"bucketize": [{"base_class": "widget", "edges": [15]}]}})
        self.assertFalse(result["overall_pass"])
        self.assertTrue(evaluate_rules_structured(image_result("1")["detections"], dict(RULES, derived={"bucketize": []}))["overall_pass"])

    def test_legacy_memory_only_compiles_counting_agent(self):
        path = self.root / "memory.jsonl"
        entries = [{"agent": "agent1_b0", "rule_line": "each sample must have 1 widget"},
                   {"agent": "agent2_split_b0", "rule_line": "widget diag in [9, 11]"},
                   {"agent": "agent3_map_b0", "rule_line": "mapping rule"},
                   {"agent": "agent4_inside_p0", "rule_line": "box contains widget"}]
        path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        with patch.object(llm_pipeline, "_call_llm", return_value=json.dumps(RULES)) as compiler:
            cfg, raw = rule_translator.translate_rule_memory(str(path), {})
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(set(raw), {"agent1"})
        self.assertEqual(cfg, RULES)
        self.assertEqual(finalize.finalize_constraints(str(path)), "agent=agent1_b0 | each sample must have 1 widget")
        prompt = compiler.call_args.kwargs["contents"][0]["text"]
        self.assertNotIn("diag_range", prompt)
        self.assertNotIn("box contains widget", prompt)


    def test_help_retains_modes_and_removes_size_arguments(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["cli", "--help"]), contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as status:
                main_cli.parse_args()
        self.assertEqual(status.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("train,infer,stats", help_text)
        self.assertIn("--infer-rules-json", help_text)
        self.assertNotIn("selfcheck", help_text)
        self.assertNotIn("--llm-", help_text)
        self.assertNotIn("--train-diag-subclasses", help_text)
        self.assertNotIn("--diag-subclass-margin", help_text)
        self.assertNotIn("sam3", help_text.lower())
        self.assertIn("--infer-detections-json", help_text)
        self.assertIn("--stats-detections-json", help_text)
        self.assertNotIn("--train-sample-ids", help_text)
        self.assertNotIn("--selfcheck-num-samples", help_text)
        self.assertNotIn("--vocab", help_text)
        self.assertNotIn("--split", help_text)
        self.assertIn("--output-dir", help_text)


    def test_explicit_json_inference_has_no_timing_outputs(self):
        path = self.write_json(self.root / "external.json", {"results": [image_result("001"), image_result("002", 0)]})
        result = main_cli._run_infer(str(path), RULES)
        self.assertEqual((result["normal"], result["abnormal"]), (1, 1))
        payload = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        self.assertNotIn("sam3", json.dumps(payload).lower())
        self.assertNotIn("timing_summary", payload)
        self.assertNotIn("timing_path", result)
        self.assertTrue(all("timing" not in item for item in payload["images"]))
        self.assertFalse((Path(result["summary_path"]).parent / "infer_timing_summary.json").exists())
        self.assertFalse((Path(result["summary_path"]).parent / "abnormal_masks").exists())

    def test_json_count_statistics(self):
        path = self.write_json(self.root / "objects.json", [image_result("1", 2), image_result("2", 0)])
        result = main_cli._run_stats(str(path))
        counts = json.loads(Path(result["counts_path"]).read_text(encoding="utf-8"))
        self.assertEqual(counts["total_images"], 2)
        self.assertEqual(counts["aggregate_base_class_counts"], {"widget": 2})


    def test_batch_inference_from_explicit_manifest(self):
        rules = self.write_json(self.root / "shared_rules.json", RULES)
        jobs = []
        for name, count in (("reference_set", 1), ("query_set", 0)):
            self.write_json(self.root / "inputs" / f"{name}.json", [image_result("1", count)])
            jobs.append({"detections_json": f"inputs/{name}.json", "rules_json": rules.name,
                         "output_dir": f"outputs/{name}"})
        manifest = self.write_json(self.root / "jobs.json", jobs)
        with patch.object(sys, "argv", ["cli", "--mode", "infer", "--batch-manifest", str(manifest)]):
            main_cli.main()
        data = json.loads((self.root / "infer" / "batch_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(sum(r["normal"] for r in data["results"]), 1)
        self.assertTrue((self.root / "outputs" / "query_set" / "infer_summary.json").exists())

    def test_batch_rejects_single_file_options(self):
        with patch.object(sys, "argv", ["cli", "--mode", "infer", "--batch-manifest", "jobs.json",
                                       "--infer-detections-json", "all.json"]):
            with self.assertRaisesRegex(ValueError, "cannot be used with --batch-manifest"):
                main_cli.main()

    def test_batch_rejects_duplicate_outputs_before_writing(self):
        manifest = self.write_json(self.root / "jobs.json", [
            {"detections_json": "a.json", "output_dir": "same"},
            {"detections_json": "b.json", "output_dir": "same"}])
        with self.assertRaisesRegex(ValueError, "distinct output directories"):
            main_cli._load_batch_jobs(str(manifest), "infer")
        self.assertFalse((self.root / "same").exists())

    def test_batch_statistics_from_manifest(self):
        self.write_json(self.root / "input.json", [image_result("1", 3)])
        manifest = self.write_json(self.root / "jobs.json", [
            {"detections_json": "input.json", "output_dir": "stats_output"}])
        with patch.object(sys, "argv", ["cli", "--mode", "stats", "--batch-manifest", str(manifest)]):
            main_cli.main()
        counts = json.loads((self.root / "stats_output" / "detection_counts.json").read_text(encoding="utf-8"))
        self.assertEqual(counts["aggregate_base_class_counts"], {"widget": 3})

    def test_inference_without_model_and_image_dependencies(self):
        import subprocess
        path = self.write_json(self.root / "objects.json", [image_result("1")])
        rules = self.write_json(self.root / "external_rules.json", RULES)
        code = (
            "import sys,runpy; "
            "sys.modules.update({n:None for n in ['sam3','vision','torch','transformers','peft','numpy','PIL','openai','socket']}); "
            "from core import config; "
            f"config.INFER_RESULT_DIR={str(self.root / 'subprocess_infer')!r}; "
            "runpy.run_module('cli',run_name='__main__')"
        )
        subprocess.check_output([sys.executable, "-B", "-c", code, "--mode", "infer",
                                 "--infer-detections-json", str(path), "--infer-rules-json", str(rules)],
                                 cwd=Path(__file__).resolve().parents[1], text=True)
        payload = json.loads((self.root / "subprocess_infer" / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["images"][0]["status"], "OK")


    def test_remote_configuration_uses_environment(self):
        with patch.dict(os.environ, {"MARAD_LLM_API_KEY": "example-test-key"}), \
             patch.object(config, "LLM_MODEL_DEFAULT", "example-model"), \
             patch.object(config, "LLM_BASE_URL_DEFAULT", "https://example.invalid/v1"):
            cfg = main_cli._build_llm_cfg()
        self.assertEqual(cfg, {"api_key": "example-test-key", "model": "example-model",
                              "base_url": "https://example.invalid/v1", "temperature": 0})
        with patch.object(llm_pipeline, "_call_remote_api", return_value="reply") as remote:
            self.assertEqual(llm_pipeline._call_llm([{"type": "text", "text": "evidence"}], cfg), "reply")
        self.assertEqual(remote.call_args.args[1:], ("example-model", "https://example.invalid/v1", "example-test-key", 0.0))

    def test_missing_api_configuration_errors_do_not_expose_credentials(self):
        with patch.dict(os.environ, {"MARAD_LLM_API_KEY": "example-test-key"}), \
             patch.object(config, "LLM_MODEL_DEFAULT", ""), \
             patch.object(config, "LLM_BASE_URL_DEFAULT", ""):
            with self.assertRaises(ValueError) as error:
                main_cli._build_llm_cfg()
        message = str(error.exception)
        self.assertIn("MARAD_LLM_MODEL", message)
        self.assertIn("MARAD_LLM_BASE_URL", message)
        self.assertNotIn("example-test-key", message)

    def test_remote_streaming_response(self):
        from types import SimpleNamespace
        client = unittest.mock.Mock()
        client.chat.completions.create.return_value = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello "))]),
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]),
        ]
        sdk = SimpleNamespace(OpenAI=unittest.mock.Mock(return_value=client))
        with patch.dict(sys.modules, {"openai": sdk}):
            reply = llm_pipeline._call_remote_api([{"type": "text", "text": "input"}],
                        "example-model", "https://example.invalid/v1", "example-test-key")
        self.assertEqual(reply, "hello world")
        options = client.chat.completions.create.call_args.kwargs
        self.assertTrue(options["stream"])
        self.assertEqual(options["messages"][0]["content"], [{"type": "text", "text": "input"}])

    def test_path_configuration_reads_environment(self):
        import subprocess
        env = dict(os.environ, MARAD_RESULT_DIR=str(self.root / "outputs"), MARAD_RULES_DIR=str(self.root / "rule_files"))
        output = subprocess.check_output([sys.executable, "-B", "-c",
                    "import json; from core import config; print(json.dumps([config.INFER_RESULT_DIR,config.get_rules_path()]))"],
                    cwd=Path(__file__).resolve().parents[1], env=env, text=True)
        self.assertEqual(json.loads(output), [str(self.root / "outputs" / "infer"),
                                              str(self.root / "rule_files" / "rules.json")])

    def test_arbitrary_labels_and_output_directory(self):
        path = self.write_json(self.root / "custom.json", [{"image_id": "1", "detections": [
            {"tag": "custom_object1"}, {"tag": "custom_object2"}, {"tag": "custom_other1"}]}])
        destination = self.root / "my_results"
        with patch.object(sys, "argv", ["cli", "--mode", "stats", "--stats-detections-json", str(path),
                                       "--output-dir", str(destination)]):
            main_cli.main()
        counts = json.loads((destination / "detection_counts.json").read_text(encoding="utf-8"))
        self.assertEqual(counts["aggregate_base_class_counts"], {"object": 2, "other": 1})
        self.assertEqual(len(counts["observed_labels"]), 3)
        self.assertNotIn("vocab", counts)
        self.assertNotIn("split", counts)
        self.assertNotIn("target_labels", counts)

    def test_missing_detection_path_does_not_guess_dataset_layout(self):
        with patch.object(sys, "argv", ["cli", "--mode", "infer"]):
            with self.assertRaisesRegex(ValueError, "--infer-detections-json is required"):
                main_cli.main()

    def test_auroc_from_explicit_summaries(self):
        from scripts import compute_infer_auroc as auroc
        normal = self.write_json(self.root / "accepted.json", {"images": [
            {"status": "OK"}, {"status": "ABNORMAL"}, {"status": "unknown"}]})
        abnormal = self.write_json(self.root / "rejected.json", {"images": [
            {"status": "ABNORMAL"}, {"status": "ABNORMAL"}]})
        output = self.root / "metrics.json"
        with patch.object(sys, "argv", ["auroc", "--normal-summary", str(normal),
                                       "--anomaly-summary", str(abnormal), "--output", str(output)]):
            auroc.main()
        result = json.loads(output.read_text(encoding="utf-8"))["results"][0]
        self.assertEqual(result["auc"], 0.75)
        self.assertEqual((result["n"], result["unknown_status"]), (4, 1))
        self.assertNotIn("vocab", result)
        self.assertIsNone(auroc.compute_summary([str(normal)], [])["auc"])


if __name__ == "__main__":
    unittest.main()
