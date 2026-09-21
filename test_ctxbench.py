#!/usr/bin/env python3
"""Tests for the parts that read a CSV and decide what it means.

    python3 test_ctxbench.py

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

import ctxbench


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
        g = ctxbench.agg([row(tps=60.0), row(tps=62.0)])
        self.assertEqual(g["n"], 2)
        self.assertAlmostEqual(g["tps"], 61.0)
        self.assertAlmostEqual(g["sd"], 1.0)

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


class Writer(unittest.TestCase):
    def test_a_field_missing_from_COLUMNS_fails_at_write_time(self):
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "r.csv")
        w = ctxbench.Writer(path)
        self.addCleanup(w.fh.close)
        with self.assertRaises(ctxbench.BenchError):
            w.row(label="a", not_a_real_column=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
