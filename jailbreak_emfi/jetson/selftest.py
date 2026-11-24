"""Self-tests of the Jetson code that need no GPU, no board and no network (run anywhere):

    PYTHONPATH=<jailbreak_emfi> python3 -m jetson.selftest

A fake host drives the worker side of the protocol over a socketpair, a fake vLLM / Ollama server runs on
the loopback interface, and the sysfs/systemd/GPIO parts are checked on temporary files or command lines.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import unittest
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from common import config, protocol
from jetson.common import prompts, results, systemd
from jetson.common.thermal import ThermalGuard, max_temp_c

PAPER_CODE = Path(__file__).resolve().parents[1]
COLLECTION = uuid.uuid4().hex


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def fill(text: str, values: dict[str, str]) -> str:
    """The shipped TOML with every empty value replaced (by key name, else a dummy)."""
    out, section = [], ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.split("]")[0].strip("[")
        if '= ""' in line:
            key = line.split("=")[0].strip()
            value = values.get(f"{section}.{key}", values.get(key, "1"))
            line = line.replace('""', value if value.startswith(("[", "true", "false")) or is_number(value)
                                else json.dumps(value), 1)
        out.append(line)
    return "\n".join(out) + "\n"


@contextmanager
def fake_http(handler_fn):
    """A loopback HTTP server; handler_fn(method, path, body) -> (status, JSON-able or bytes)."""
    calls: list = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"null") if length else None
            calls.append((method, self.path, body))
            status, payload = handler_fn(method, self.path, body)
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply("GET")

        def do_POST(self):
            self._reply("POST")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], calls
    finally:
        server.shutdown()
        server.server_close()


class ConfigTests(unittest.TestCase):
    def test_shipped_values_are_empty(self):
        for name, key in (("thor.toml", "network.listen_port"), ("thor.toml", "vllm.temperature"),
                          ("thor.toml", "models.gemma3_1b.served_name"), ("orin.toml", "ollama.model"),
                          ("orin.toml", "memfault.gib")):
            with self.assertRaises(config.ConfigError) as ctx:
                config.load(name).get(key)
            self.assertIn(key, str(ctx.exception))


class ProtocolTests(unittest.TestCase):
    def test_prompt_trial_and_weight_report(self):
        host, worker = socket.socketpair()
        with host, worker:
            def fake_host():
                protocol.wait_for_collect(host)
                host.sendall(protocol.acknowledgement(7, COLLECTION, prompt_slot=12, prompt_repeat=3,
                                                      check_weights=protocol.CHECK_WEIGHTS_REPORT))
                self.report = protocol.recv_weight_report(host)

            t = threading.Thread(target=fake_host)
            t.start()
            hs = protocol.request_prompt_trial(worker)
            self.assertEqual((hs.attempt_index, hs.collection_id, hs.prompt_slot, hs.prompt_repeat, hs.check_weights),
                             (7, COLLECTION, 12, 3, protocol.CHECK_WEIGHTS_REPORT))
            protocol.send_weight_report(worker, 7, "CHANGED", 1, b'{"tensors": []}')
            t.join(5)
            self.assertEqual(self.report, (7, "CHANGED", 1, '{"tensors": []}'))

    def test_request_trial_any_both_forms(self):
        host, worker = socket.socketpair()
        with host, worker:
            def fake_host():
                protocol.wait_for_collect(host)
                host.sendall(protocol.acknowledgement(1, COLLECTION))
                protocol.wait_for_collect(host)
                ack = protocol.acknowledgement(2, COLLECTION) + protocol.u64(5) + protocol.u64(1)
                host.sendall(ack)

            t = threading.Thread(target=fake_host)
            t.start()
            short = protocol.request_trial_any(worker)
            long = protocol.request_trial_any(worker)
            t.join(5)
            self.assertEqual((short.attempt_index, short.prompt_slot), (1, None))
            self.assertEqual((long.attempt_index, long.prompt_slot, long.prompt_repeat), (2, 5, 1))


class CommonTests(unittest.TestCase):
    def test_thermal_guard(self):
        with tempfile.TemporaryDirectory() as d:
            for i, millideg in enumerate((45000, 72500)):
                zone = Path(d) / f"thermal_zone{i}"
                zone.mkdir()
                (zone / "temp").write_text(f"{millideg}\n")
            self.assertEqual(max_temp_c(Path(d)), 72.5)
            ThermalGuard(85, 70, 0.01, Path(d)).wait()          # below pause: returns at once
            hot = ThermalGuard(70, 60, 0.01, Path(d))
            readings = iter([72.5, 72.5, 65.0, 59.0])
            with mock.patch.object(hot, "temp", lambda: next(readings)):
                hot.wait()                                      # waits until <= resume
            with self.assertRaises(ValueError):
                ThermalGuard(70, 80, 1)

    def test_prompts_and_results(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.txt"
            p.write_text("first\n\n  second  \nthird\n")
            self.assertEqual(prompts.load_lines(p), ["first", "second", "third"])
            self.assertEqual(prompts.load_lines(p, 2), ["first", "second"])
            csv_path = Path(d) / "b.csv"
            csv_path.write_text('Index,Goal,Target,Behavior,Category,Source\n1,"Do X, now",t,b,c,s\n')
            self.assertEqual(prompts.load_jbb_csv(csv_path)[0]["Goal"], "Do X, now")
            out = Path(d) / "sub" / "r.jsonl"
            results.append_jsonl(out, {"a": "é"})
            with out.open("a") as f:
                f.write('{"torn": ')
            self.assertEqual(results.read_jsonl(out), [{"a": "é"}])
            tsv = Path(d) / "log.tsv"
            results.append_tsv(tsv, "h1\th2", ["1\t2"])
            results.append_tsv(tsv, "h1\th2", ["3\t4"])
            self.assertEqual(tsv.read_text(), "h1\th2\n1\t2\n3\t4\n")

    def test_systemctl_commands(self):
        self.assertEqual(systemd.systemctl_command("restart", ["vllm-x"], with_password=False),
                         ["sudo", "-n", "/usr/bin/systemctl", "restart", "vllm-x.service"])
        self.assertEqual(systemd.systemctl_command("stop", ["a.service", "b"], with_password=True)[:4],
                         ["sudo", "-S", "-p", ""])


class ThorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def thor_cfg(self, port: int) -> config.Config:
        text = fill((PAPER_CODE / "config" / "thor.toml").read_text(), {
            "vllm.client_host": "127.0.0.1", "vllm.port": str(port), "vllm.request_timeout_s": "5",
            "vllm.startup_retry_s": "0.01", "vllm.temperature": "0", "vllm.quantization": "none",
            "models.gemma3_1b.served_name": "test/gemma", "models.gemma3_1b.unit": "vllm-gemma",
            "models.gemma3_1b.family": "gemma", "models.llama32_1b.served_name": "test/llama",
            "models.llama32_1b.unit": "vllm-llama", "models.llama32_1b.family": "llama",
            "query_trigger.reload_token": "our", "query_trigger.reload_threshold": "5",
            "query_trigger.output": str(self.dir / "q.jsonl"), "query_trigger.max_tokens": "150",
            "prompts.file": str(self.dir / "prompts.txt"), "system.user": "tester",
            "system.paper_code": "/opt/paper_code", "system.home": "/home/tester", "system.python": "/usr/bin/python3",
            "vllm.prefix_caching": "false", "vllm.enforce_eager": "true", "vllm.weight_check": "true",
            "reload_trigger.cpu_performance_governor": "true", "memfault.pattern": "index", "memfault.mem": "host",
            "memfault.ecc_polling": "false", "vllm.venv": "/opt/venv", "vllm.cuda_dir": "/usr/local/cuda",
            "vllm.host": "0.0.0.0", "models.gemma3_1b.dir": "/models/gemma", "models.llama32_1b.dir": "/models/llama",
            "memfault.binary": "/opt/gpu_memfault", "memfault.log": "/tmp/flips.tsv"})
        path = self.dir / "thor.toml"
        path.write_text(text)
        return config.load(path)

    def test_query_worker_over_socketpair(self):
        from jetson.thor import vllm_query_trigger as qt
        from jetson.thor.vllm_client import VllmClient

        replies = iter(["Sure, our our our our our plan", "I can't help with that."])

        def vllm(method, path, body):
            if path == "/v1/models":
                return 200, {"data": [{"id": "test/llama", "root": "/models/llama"}]}
            assert path == "/v1/chat/completions" and body["return_token_ids"] and body["temperature"] == 0
            return 200, {"id": "r", "model": body["model"], "usage": {"prompt_tokens": 9},
                         "choices": [{"message": {"content": next(replies)}, "token_ids": [1, 2],
                                      "finish_reason": "length"}]}

        (self.dir / "prompts.txt").write_text("p0\np1\np2\n")
        with fake_http(vllm) as (port, calls):
            cfg = self.thor_cfg(port)
            client = VllmClient.from_config(cfg)
            model = client.wait_for_model(["test/gemma", "test/llama"])
            settings = qt.settings_from_config(cfg, "query_trigger", model)
            host, worker = socket.socketpair()
            restarts = []
            with host, worker, mock.patch.object(qt, "reload_by_restart", lambda *a: restarts.append(a[3])):
                def fake_host():
                    for attempt, slot in ((10, 4), (11, 1)):
                        protocol.wait_for_collect(host)
                        host.sendall(protocol.acknowledgement(attempt, COLLECTION, slot, 1, 0))

                t = threading.Thread(target=fake_host)
                t.start()
                qt.serve_trials(worker, client, settings, prompts.load_lines(self.dir / "prompts.txt"), 2,
                                ["vllm-gemma", "vllm-llama"], ["test/llama"])
                t.join(5)
        rows = results.read_jsonl(self.dir / "q.jsonl")
        self.assertEqual([r["prompt"] for r in rows], ["p1", "p1"])      # slot 4 % 3 prompts, then slot 1
        self.assertEqual([r["sentinel_count"] for r in rows], [5, 0])
        self.assertEqual(rows[0]["attempt_index"], 10)
        self.assertEqual(rows[0]["collection_id"], COLLECTION)
        self.assertEqual(restarts, [1])                                  # watchdog tripped on trial 1 only
        chat = [c for c in calls if c[1] == "/v1/chat/completions"][0][2]
        self.assertEqual(chat["max_tokens"], 150)

    def test_reload_worker_reports_weights(self):
        from jetson.thor import vllm_reload_trigger as rt
        from jetson.thor.vllm_client import VllmClient

        checks = iter([{"changed": [], "baselined": False},
                       {"changed": [{"name": "model.layers.8.mlp.down_proj.weight", "changed_elems": 12,
                                     "zero_elems": 0}], "baselined": False}])

        def vllm(method, path, body):
            if path == "/collective_rpc":
                if body["method"] == "reload_weights":
                    return 200, {"results": [None]}
                if body["method"] == "wc_check":
                    return 200, {"results": [next(checks)]}
            return 404, {}

        with fake_http(vllm) as (port, calls):
            cfg = self.thor_cfg(port)
            s = rt.ReloadSettings(reload_every=1, reload_timeout_s=5, reload_only=True, drop_cache=False,
                                  baseline_from_reload=True, zero_before_reload=False, stop_on_change=True,
                                  baseline_verify_tries=1, weight_report_log=self.dir / "w.jsonl")
            reloader = rt.Reloader(VllmClient.from_config(cfg), s, [])
            host, worker = socket.socketpair()
            reports = []
            with host, worker:
                def fake_host():
                    for attempt in (1, 2):
                        protocol.wait_for_collect(host)
                        host.sendall(protocol.acknowledgement(attempt, COLLECTION, 0, 1,
                                                              protocol.CHECK_WEIGHTS_REPORT))
                        reports.append(protocol.recv_weight_report(host))

                t = threading.Thread(target=fake_host)
                t.start()
                status = rt.run_trials(reloader, reloader.client, None, ["p"], 5, worker)
                t.join(5)
        self.assertEqual(status, rt.EXIT_WEIGHTS_CHANGED)               # stop_on_change after attempt 2
        self.assertEqual([r[1] for r in reports], ["MATCH", "CHANGED"])
        self.assertEqual(reports[1][2], 1)
        self.assertEqual(results.read_jsonl(self.dir / "w.jsonl")[0]["status"], "CHANGED")
        self.assertEqual([c[2]["method"] for c in calls].count("reload_weights"), 2)

    def test_serve_command_and_units(self):
        from jetson import install
        from jetson.thor.gpu_memfault.launch import memfault_command
        from jetson.thor.serve_vllm import WORKER_EXTENSION, vllm_command

        cfg = self.thor_cfg(8000)
        command, env = vllm_command(cfg, "gemma3_1b", ["--extra"])
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--enforce-eager", command)
        self.assertNotIn("--quantization", command)
        self.assertEqual(command[command.index("--worker-extension-cls") + 1], WORKER_EXTENSION)
        self.assertEqual(env["VLLM_SERVER_DEV_MODE"], "1")
        self.assertEqual(command[-1], "--extra")
        self.assertIn("--no-ecc", memfault_command(cfg))
        files = install.thor_files(cfg)
        self.assertIn("vllm-gemma.service", files)
        self.assertIn("Conflicts=gpu-memfault.service vllm-llama.service", files["vllm-gemma.service"])
        self.assertIn("performance", files["vllm-reload-trigger.service"])
        self.assertIn("tester ALL=(root) NOPASSWD: /usr/bin/systemctl restart vllm-llama.service",
                      files["vllm-restart.sudoers"])
        for name, text in files.items():
            self.assertNotRegex(text, install.PLACEHOLDER, name)


class OrinTests(unittest.TestCase):
    def test_ollama_runner_and_reload_reasons(self):
        from jetson.orin.ollama_client import OllamaClient, normalize_runner_result
        from jetson.orin.ollama_query_trigger import reload_reason

        r = normalize_runner_result("m", {"content": "hi", "stop": True,
                                          "timings": {"prompt_ms": 1.5, "predicted_ms": 2}, "tokens_predicted": 3})
        self.assertEqual((r["done_reason"], r["total_duration"], r["eval_count"]), ("stop", 3_500_000, 3))
        self.assertIsNone(normalize_runner_result("m", {"no": "content"}))
        self.assertIsNotNone(reload_reason({"response": " ??? "}, 1, 0, "<tok>"))
        self.assertIsNotNone(reload_reason({"response": "x<tok>"}, 1, 0, "<tok>"))
        self.assertIsNone(reload_reason({"response": "fine?"}, 1, 0, "<tok>"))
        self.assertIsNotNone(reload_reason({"response": "fine"}, 4, 2, "<tok>"))

        def runner(method, path, body):
            if path == "/completion":
                assert body["prompt"] == "<u>hello</u>" and body["n_predict"] == 10
                return 200, {"content": "answer", "stop": False}
            return 200, b"ok"

        with fake_http(runner) as (port, _), tempfile.TemporaryDirectory() as d:
            text = fill((PAPER_CODE / "config" / "orin.toml").read_text(), {
                "ollama.runner_host": "127.0.0.1", "ollama.runner_port": str(port), "ollama.max_tokens": "10",
                "ollama.prompt_template": "<u>{prompt}</u>", "ollama.cache_prompt": "true",
                "ollama.host": "127.0.0.1", "ollama.model": "m:1b", "ollama.keep_alive": "-1",
                "ollama.startup_retry_s": "0.01"})
            (Path(d) / "orin.toml").write_text(text)
            client = OllamaClient(config.load(Path(d) / "orin.toml"))
            self.assertEqual(client.wait_for_runner(), port)
            self.assertEqual(client.complete("hello")["response"], "answer")

    def test_memcheck_rows(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        from jetson.orin import memcheck

        w = memcheck.Workload(nbytes=1 << 20, blocks=4, threads=8, seed=1, passes=1, data_mismatch_capacity=4,
                              log_dir=Path("/nonexistent"))
        row = memcheck.format_pass_result(timestamp="T", run_id="R", pass_index=1, attempt_index=None,
                                          total_passes=1, compute_elapsed=0.5, mismatch_count=0)
        self.assertEqual(row.split("\t")[3:6], ["NA", "1", "NO_MISMATCH"])
        records = np.zeros(1, dtype=memcheck.DATA_MISMATCH_DTYPE)
        records[0] = (33, 0xFF, 0xF0)
        line = next(memcheck.data_mismatch_rows("T", "R", 1, 5, 0.1, records, 0x1000, w, 3)).split("\t")
        # word 33: grid stride 32 -> iteration 1, block 0, thread 1; truncated (3 found, 1 captured)
        self.assertEqual(line[7:15], ["132", "0", "1", "1", "0x000000ff", "0x000000f0", "0x0000000f",
                                      "0x0000000000001084"])
        self.assertEqual(line[-1], "1")
        self.assertEqual(memcheck.scan_status(3, 1), "TRUNCATED")
        with self.assertRaises(ValueError):
            memcheck.Workload(nbytes=1 << 20, blocks=4, threads=6, seed=1, passes=1, data_mismatch_capacity=1,
                              log_dir=Path("."))

    def test_orin_units(self):
        from jetson import install

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "orin.toml").write_text(fill((PAPER_CODE / "config" / "orin.toml").read_text(), {
                "system.user": "tester", "memfault.pa_module": "/opt/pa/orin_gpu_pa.ko"}))
            files = install.orin_files(config.load(Path(d) / "orin.toml"))
        self.assertIn("ExecStart=/usr/local/sbin/load-orin-gpu-pa start /opt/pa/orin_gpu_pa.ko",
                      files["orin-gpu-pa.service"])
        for name, text in files.items():
            self.assertNotRegex(text, install.PLACEHOLDER, name)


if __name__ == "__main__":
    os.environ.pop("EMFI_CONFIG_DIR", None)
    unittest.main(verbosity=2)
