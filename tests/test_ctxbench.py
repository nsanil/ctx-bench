#!/usr/bin/env python3
"""Tests for the parts that read a CSV and decide what it means.

    python3 tests/test_ctxbench.py

No server, no network, no fixtures beyond hand-written rows. The measurement
code needs a GPU and a model; the code that turns measurements into a verdict
does not, and it is the part a reader trusts without being able to check.

The case that matters most is a cell with one repeat. A single sample has no
spread. Treating that as a spread of zero would make the noise floor zero, so
every difference would clear it and print "real" in the same format as a
properly replicated result.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

# ctxbench.py is one level up, and both `python3 tests/test_ctxbench.py` and a
# bare `pytest` leave the repo root off sys.path. Inserting it here covers them
# without a package marker or a conftest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctxbench  # noqa: E402


def row(label="a", tps=60.0, depth=0, suite="code", effort="off",
        max_tokens=200, accept=70.0, answered=1, wall=10.0):
    """One loaded row, shaped as `load` would leave it."""
    return {"label": label, "decode_tps": tps, "depth_target": depth,
            "suite": suite, "effort": effort, "max_tokens": max_tokens,
            "accept_pct": accept, "answered": answered, "wall_s": wall,
            "prefill_tps": None, "gen_tok": 100, "pass": 1}


class Agg(unittest.TestCase):
    def test_sd_is_none_for_a_single_sample(self):
        self.assertIsNone(ctxbench.agg([row()])["sd"])

    def test_sd_is_a_number_once_replicated(self):
        # Sample standard deviation, not population: repeats are a sample of
        # run-to-run variation. pstdev would give 1.0 here and runs about 13%
        # smaller at four repeats, which would make the noise floor less
        # conservative than the README claims it is.
        g = ctxbench.agg([row(tps=60.0), row(tps=62.0)])
        self.assertEqual(g["n"], 2)
        self.assertAlmostEqual(g["tps"], 61.0)
        self.assertAlmostEqual(g["sd"], 2 ** 0.5)

    def test_acceptance_absent_when_there_is_no_drafter(self):
        # A server without speculative decoding reports no draft counters at
        # all, which has to read as "unknown" rather than as zero acceptance.
        g = ctxbench.agg([row(accept=None), row(accept=None)])
        self.assertIsNone(g["acc"])

    def test_answered_counts_replies_that_actually_produced_text(self):
        g = ctxbench.agg([row(answered=1), row(answered=0), row(answered=0)])
        self.assertEqual(g["answered"], 1)
        self.assertEqual(g["n"], 3)


class Report(unittest.TestCase):
    def render(self, rows):
        buf = io.StringIO()
        args = type("A", (), {"csv": None, "label": None})()
        # cmd_report reads from load(); feed it rows directly instead.
        orig, ctxbench.load = ctxbench.load, lambda _p: rows
        try:
            with redirect_stdout(buf):
                ctxbench.cmd_report(args)
        finally:
            ctxbench.load = orig
        return buf.getvalue()

    def test_single_repeat_refuses_to_call_a_difference_real(self):
        out = self.render([row(label="a", tps=50.0), row(label="b", tps=70.0)])
        self.assertIn("no floor", out)
        self.assertNotIn("real", out)

    def test_difference_inside_the_floor_reads_as_noise(self):
        rows = [row(label="a", tps=60.0), row(label="a", tps=64.0),
                row(label="b", tps=61.0), row(label="b", tps=65.0)]
        self.assertIn("noise", self.render(rows))

    def test_difference_clearing_the_floor_reads_as_real(self):
        rows = [row(label="a", tps=60.0), row(label="a", tps=60.2),
                row(label="b", tps=80.0), row(label="b", tps=80.2)]
        self.assertIn("real", self.render(rows))

    def test_labels_with_no_shared_cell_are_not_compared(self):
        # Two different experiments in one CSV must not be lined up by depth
        # alone; that produces a confident-looking table of nonsense.
        rows = [row(label="a", suite="code", max_tokens=200),
                row(label="b", suite="prose", max_tokens=8000)]
        self.assertIn("no cells in common", self.render(rows))

    def test_depth_column_is_relative_to_the_shallowest_row(self):
        rows = [row(depth=0, tps=100.0), row(depth=0, tps=100.0),
                row(depth=8192, tps=75.0), row(depth=8192, tps=75.0)]
        self.assertIn("-25.0%", self.render(rows))


class Prompts(unittest.TestCase):
    def test_filler_varies_so_it_does_not_inflate_draft_acceptance(self):
        # Repeated filler is highly draftable, which raises acceptance and
        # flatters every throughput number measured on top of it.
        self.assertNotEqual(ctxbench.block(0), ctxbench.block(1))
        self.assertEqual(len({ctxbench.block(i) for i in range(50)}), 50)

    def test_depth_zero_needs_no_server(self):
        self.assertEqual(ctxbench.build_prompt(None, 0, "ask"), "ask")

    def test_a_depth_too_small_to_hit_fails_instead_of_guessing(self):
        class FakeServer:
            def ntokens(self, text):
                return len(text.split())

        with self.assertRaises(ctxbench.BenchError) as e:
            ctxbench.build_prompt(FakeServer(), 3, "ask")
        self.assertIn("depth", str(e.exception))

    def test_a_server_whose_token_count_never_rises_stops_the_loop(self):
        # Without this guard the filler list grows forever and each iteration
        # POSTs a larger body, so a lying /tokenize turns into unbounded memory
        # and unbounded outbound traffic to whoever is lying.
        class LyingServer:
            calls = 0

            def ntokens(self, text):
                LyingServer.calls += 1
                return 5

        with self.assertRaises(ctxbench.BenchError) as e:
            ctxbench.build_prompt(LyingServer(), 100_000, "ask")
        self.assertIn("stopped rising", str(e.exception))
        self.assertLess(LyingServer.calls, 10)


class ResponseLimits(unittest.TestCase):
    def test_an_endless_body_is_refused_rather_than_buffered(self):
        import unittest.mock as mock

        class Endless:
            def read(self, n=-1):
                # More than asked for would be a broken file object; return
                # exactly the cap plus one, which is what the guard checks.
                return b"x" * n

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        srv = ctxbench.Server("http://example.invalid", "m")
        with mock.patch("urllib.request.urlopen", return_value=Endless()):
            with self.assertRaises(ctxbench.BenchError) as e:
                srv.post("/v1/chat/completions", {})
        self.assertIn("refusing to buffer", str(e.exception))


class UntrustedServerOutput(unittest.TestCase):
    """--url can point anywhere, so the server's strings are attacker input."""

    def test_formula_leads_are_defused_before_reaching_a_spreadsheet(self):
        for lead in ("=", "+", "-", "@"):
            self.assertTrue(
                ctxbench.safe_cell(lead + "HYPERLINK(...)").startswith("'"))

    def test_ordinary_values_are_left_alone(self):
        self.assertEqual(ctxbench.safe_cell("stop"), "stop")
        self.assertEqual(ctxbench.safe_cell(42), 42)
        self.assertIsNone(ctxbench.safe_cell(None))

    def test_control_characters_are_stripped_from_cells(self):
        self.assertEqual(ctxbench.safe_cell("st\x1b[2Jop"), "st[2Jop")

    def test_non_finite_numbers_do_not_bypass_the_escaping(self):
        # json.loads accepts Infinity and NaN, and -inf leads with a `-`, so
        # these are the one value class that is neither a string to escape nor
        # a number safe to pass through.
        for v in (float("inf"), float("-inf"), float("nan")):
            self.assertEqual(ctxbench.safe_cell(v), "")
        self.assertEqual(ctxbench.safe_cell(42.5), 42.5)

    def test_html_sniffing_has_one_definition(self):
        self.assertTrue(ctxbench.looks_html(b"<!DOCTYPE html><html>"))
        self.assertTrue(ctxbench.looks_html(b"\n  <html lang=\"en\">"))
        self.assertFalse(ctxbench.looks_html(b'{"choices": []}'))

    def test_a_foreign_csv_is_sanitised_on_the_way_in(self):
        # Every Report test patches load() out, so without this the real
        # function is untested -- and it is the one that sees a file this
        # tool did not write.
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "foreign.csv")
        with open(path, "w", newline="") as fh:
            fh.write(",".join(ctxbench.COLUMNS) + "\n")
            row = {c: "" for c in ctxbench.COLUMNS}
            row.update(label="\x1b[2Jevil", suite="co\x07de", effort="off",
                       decode_tps="50.0", depth_target="0", max_tokens="200",
                       answered="1", wall_s="1.0", accept_pct="70.0")
            fh.write(",".join(row[c] for c in ctxbench.COLUMNS) + "\n")
        r = ctxbench.load(path)[0]
        self.assertNotIn("\x1b", r["label"])
        self.assertNotIn("\x07", r["suite"])
        self.assertEqual(r["effort"], "off")

    def test_an_nvidia_smi_in_the_working_directory_is_refused(self):
        import unittest.mock as mock
        planted = os.path.join(os.getcwd(), "nvidia-smi")
        with mock.patch("shutil.which", return_value=planted):
            with mock.patch("subprocess.run") as run:
                self.assertEqual(ctxbench.gpu(), [""] * 5)
                run.assert_not_called()

    def test_escape_sequences_never_reach_the_terminal(self):
        # OSC 52 writes the clipboard on terminals that honour it.
        out = ctxbench.printable("\x1b]52;c;cGF5bG9hZA==\x07ok")
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x07", out)

    def test_newlines_and_tabs_survive_so_errors_stay_readable(self):
        self.assertEqual(ctxbench.printable("a\nb\tc"), "a\nb\tc")


class SharedArtifact(unittest.TestCase):
    """The manifest is written to be sent to other people."""

    def test_a_windows_server_path_is_reduced_on_a_posix_client(self):
        # os.path binds to posixpath here and does not split on backslashes, so
        # a Windows server's model_path -- account name and all -- would pass
        # through untouched.
        import ntpath
        win = "C:" + chr(92) + "Users" + chr(92) + "someone" + chr(92) + "Q.gguf"
        self.assertEqual(ntpath.basename(win), "Q.gguf")
        self.assertEqual(ntpath.basename("/home/u/models/Q.gguf"), "Q.gguf")

    def test_credentials_in_the_url_never_reach_the_manifest(self):
        out = ctxbench.safe_url("http://user:secrettoken@gpu.internal:8080")
        self.assertNotIn("secrettoken", out)
        self.assertNotIn("user", out)
        self.assertNotIn("gpu.internal", out)

    def test_a_private_hostname_is_not_recorded(self):
        self.assertEqual(
            ctxbench.safe_url("http://workstation.tail1234.ts.net:8080"),
            "http://<host>:8080")

    def test_loopback_is_kept_because_it_identifies_nobody(self):
        self.assertEqual(ctxbench.safe_url("http://127.0.0.1:8080"),
                         "http://127.0.0.1:8080")
        self.assertEqual(ctxbench.safe_url("http://[::1]:8080"),
                         "http://[::1]:8080")

    def test_a_junk_port_does_not_abort_the_run(self):
        # urlsplit accepts these and only raises when .port is read, and this
        # runs before the first request, so an escape here kills the run.
        for url in ("http://h:99999", "http://h:abc", "http://h:-1",
                    "", "http://", "not a url at all"):
            self.assertIsInstance(ctxbench.safe_url(url), str)
        self.assertEqual(ctxbench.safe_url("http://h:99999"), "unparseable")


class Telemetry(unittest.TestCase):
    """nvidia-smi prints one row per GPU, and the caller unpacks exactly five."""

    def _gpu_with(self, stdout):
        import unittest.mock as mock

        class R:
            def __init__(self, o):
                self.stdout = o

        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             mock.patch("subprocess.run", return_value=R(stdout)):
            return ctxbench.gpu()

    def test_a_second_gpu_does_not_crash_the_run(self):
        # A multi-GPU host emits one row per card. Splitting all of stdout
        # on commas would hand the caller ten fields for a five-value unpack,
        # and that raises outside gpu()'s own except.
        got = self._gpu_with("250.1, 65, 1800, 92, 23000\n"
                             "180.0, 55, 1600, 40, 8000\n")
        self.assertEqual(len(got), 5)
        self.assertEqual(got, [""] * 5)

    def test_one_gpu_is_read_normally(self):
        got = self._gpu_with("250.1, 65, 1800, 92, 23000\n")
        self.assertEqual(got, ["250.1", "65", "1800", "92", "23000"])

    def test_no_output_or_junk_still_yields_five_fields(self):
        for out in ("", "\n", "oops\n", "1, 2, 3\n"):
            self.assertEqual(len(self._gpu_with(out)), 5)

    def test_an_nvidia_smi_in_the_working_directory_is_refused_once(self):
        import unittest.mock as mock
        planted = os.path.join(os.getcwd(), "nvidia-smi")
        with mock.patch("shutil.which", return_value=planted):
            self.assertIsNone(ctxbench.safe_nvidia_smi())


class MixedLabels(unittest.TestCase):
    def test_a_label_holding_two_experiments_refuses_a_depth_table(self):
        # Averaging a 200-token code run with an 8,000-token prose one under
        # one label is the silent apples-to-oranges this tool exists to avoid.
        rows = [row(depth=0, suite="code", max_tokens=200),
                row(depth=8192, suite="prose", max_tokens=8000)]
        buf = io.StringIO()
        orig, ctxbench.load = ctxbench.load, lambda _p: rows
        try:
            with redirect_stdout(buf):
                ctxbench.cmd_report(type("A", (), {"csv": None, "label": None})())
        finally:
            ctxbench.load = orig
        out = buf.getvalue()
        self.assertIn("more than one kind of run", out)
        self.assertNotIn("vs ", out)


class Writer(unittest.TestCase):
    def test_a_file_with_a_foreign_header_is_not_appended_to(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "old.csv")
        with open(path, "w") as fh:
            fh.write("ts,label,something_else\n1,a,2\n")
        with self.assertRaises(ctxbench.BenchError) as e:
            ctxbench.Writer(path)
        self.assertIn("different column set", str(e.exception))

    def test_an_empty_file_gets_a_header(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "empty.csv")
        open(path, "w").close()          # zero bytes, but it exists
        w = ctxbench.Writer(path)
        self.addCleanup(w.fh.close)
        w.close()
        with open(path) as fh:
            self.assertEqual(fh.readline().strip().split(","), ctxbench.COLUMNS)

    def test_a_field_missing_from_COLUMNS_fails_at_write_time(self):
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "r.csv")
        w = ctxbench.Writer(path)
        self.addCleanup(w.fh.close)
        with self.assertRaises(ctxbench.BenchError):
            w.row(label="a", not_a_real_column=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
