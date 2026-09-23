#!/usr/bin/env python3
"""Measure what a llama.cpp server actually does at the operating point you run it at.

Published tok/s figures are almost always taken on an empty cache with an
unstated prompt. This measures the things that move that number and are usually
left out: how full the cache is, what you asked for, how long the reply runs,
and what reasoning effort the template picked when you sent nothing.

    ctxbench.py depth    --url ... --model ...   # decode rate vs context depth
    ctxbench.py grid     --url ... --model ...   # reasoning effort x workload
    ctxbench.py ab       --url ... --model ...   # content and reply length
    ctxbench.py report   results.csv             # tables, with a noise floor

Stdlib only. Bring your own server; this never starts, stops or configures one,
because server lifecycle is the part that differs most between machines.

Three design choices worth knowing about before trusting any number it prints:

Depth order rotates every pass. A ladder that always ascends measures its
deepest point when the card is hottest, then reports heat as depth.

The server configuration, the requested output cap, the prompt and the
filler-generation procedure are held constant; context depth is the controlled
variable. Note the cap is a cap: if the model stops early at one depth and not
another, reply length has varied too, and `depth` says so when it happens.

Filler is sized by the server's own tokenizer, and varies as it goes. Repeating
one paragraph inflates speculative-draft acceptance, and acceptance strongly
affects throughput on a speculative setup -- so lazy filler quietly
manufactures the result.
"""
import argparse
import csv
import hashlib
import json
import ntpath
import math
import os
import shutil
import statistics as st
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request

VERSION = "0.1.0"

COLUMNS = [
    "ts", "version", "run_id", "label", "pass", "suite", "effort", "depth_target", "max_tokens",
    "ctx_tokens", "prompt_n", "cache_n", "gen_tok", "decode_tps", "prefill_tps",
    "draft_n", "draft_acc", "accept_pct", "reason_chars", "content_chars",
    "answered", "finish", "wall_s", "power_w", "temp_c", "clock_mhz",
    "util_pct", "vram_used_mib",
]

# Two workloads that sit at opposite ends of how predictable the output is.
# On a speculative-decoding setup this can materially change throughput, and it
# is the variable a benchmark prompt silently chooses for you.
PROMPTS = {
    "code": ("Write a complete Python implementation of a red-black tree with "
             "insert, delete and search. Include full docstrings and type hints."),
    "prose": ("Explain, in detail and step by step, how a memory-bandwidth-bound "
              "workload differs from a compute-bound one, and what that implies "
              "for choosing hardware. Be thorough."),
}

_WORDS = ("system latency throughput cache coherence pipeline scheduler kernel "
          "tensor gradient checkpoint allocator fragmentation residency batch "
          "quantization calibration outlier saturation bandwidth occupancy")

# --url can point at anything, so everything the server says is untrusted input.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_FILLER_BLOCKS = 200_000
# Leading characters a spreadsheet reads as the start of a formula. Tab and CR
# belong to this set too, but safe_cell strips non-printables before it checks,
# so a value hiding an `=` behind a leading tab arrives here with the `=`
# already exposed. Listing them as well would be unreachable.
_FORMULA_LEAD = ("=", "+", "-", "@")


def safe_cell(v):
    """Neutralise a value the server chose before it reaches a CSV.

    Results get opened in Excel or Sheets, where a cell beginning `=` is code.
    Most columns here are numbers, but `finish_reason` is a free string and the
    numeric fields are only numeric if the server felt like it.
    """
    if isinstance(v, float) and not math.isfinite(v):
        # json.loads accepts Infinity and NaN, and -inf leads with a `-`, so a
        # non-finite number is the one value that would slip past the check
        # below by not being a string at all.
        return ""
    if v is None or isinstance(v, (int, float)):
        return v
    s = str(v)
    # Strip control characters first; they would otherwise ride into a terminal
    # or a cell regardless of the quoting.
    s = "".join(c for c in s if c.isprintable())
    return "'" + s if s[:1] in _FORMULA_LEAD else s


def printable(s, limit=400):
    """Server text on its way to a terminal, with escape sequences removed."""
    return "".join(c for c in str(s)[:limit]
                   if c.isprintable() or c in "\n\t")


def looks_html(raw):
    """Is this body a web page rather than an API reply?

    One definition with two callers. Sniffing it separately at each is how the
    two drift apart the day a third prefix is worth recognising.
    """
    return raw[:200].lstrip().lower().startswith(
        (b"<!doctype", b"<html", b"<?xml"))


def not_a_server(url, path, raw):
    """What to say when the reply is not JSON.

    Printing the body here is worse than useless: the usual cause is a URL
    aimed at a web server, and the reader gets a screenful of HTML instead of
    the one sentence that would fix it.
    """
    what = "an HTML page" if looks_html(raw) else "something that is not JSON"
    return (f"{url}{path} returned {what}. That URL is probably not a "
            f"llama.cpp server -- this tool needs /v1/chat/completions and "
            f"/tokenize on the same host, and a generic OpenAI-compatible "
            f"proxy will not have the second one.")


# --------------------------------------------------------------------------
# server


class Server:
    def __init__(self, url, model, timeout=1800, sampling=None):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Greedy by default so repeats are comparable, but overridable: a model
        # whose published preset differs between thinking and non-thinking
        # modes cannot be benchmarked honestly on one fixed setting.
        self.sampling = sampling or {"temperature": 0}

    def _fetch(self, req, path, timeout):
        """One place where a response becomes JSON, so GET and POST agree.

        Keep it that way: a second copy is a second chance to lose the "this is
        not a llama.cpp server" diagnosis, which is the message here most worth
        having.
        """
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # Capped: an endpoint that streams forever would otherwise take the
            # client's memory with it, and --url can point anywhere.
            raw = r.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise BenchError(
                f"{self.url}{path} returned more than "
                f"{MAX_RESPONSE_BYTES // 1024 // 1024} MiB; refusing to buffer it")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Almost always a URL pointing at something that is not a
            # llama.cpp server. Say that, rather than printing the HTML.
            raise BenchError(not_a_server(self.url, path, raw))

    def post(self, path, payload, timeout=None):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        return self._fetch(req, path, timeout or self.timeout)

    def get(self, path, timeout=20):
        return self._fetch(urllib.request.Request(self.url + path), path, timeout)

    def ntokens(self, text):
        """Token count from the server, never an estimate.

        A 4-chars-per-token guess is wrong by enough at 100K to put a row at a
        different depth than its label claims.
        """
        return len(self.post("/tokenize", {"content": text}, timeout=120)["tokens"])

    def chat(self, prompt, max_tokens, effort=None, thinking=None):
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "stream": False}
        payload.update(self.sampling)
        if effort:
            payload["reasoning_effort"] = effort
        if thinking is not None:
            # enable_thinking is a chat-template variable, so it travels in
            # chat_template_kwargs. Sent at the top level it is accepted and
            # ignored, which looks exactly like the model refusing to comply.
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        t0 = time.perf_counter()
        resp = self.post("/v1/chat/completions", payload)
        return resp, time.perf_counter() - t0


def safe_url(url):
    """A server address fit to appear in a file meant to be shared.

    Credentials first: the tool cannot send an auth header, so an operator
    whose server wants one has only http://user:token@host left, and that
    would otherwise land in the manifest in clear. Then the host itself, which
    is a live machine on someone's network rather than a property of the
    software -- scheme and port are what make a result interpretable, the
    hostname is not.
    """
    # urlsplit is lazy: it accepts a junk port and only raises when .port is
    # read, so both have to sit inside the guard. This runs before any request
    # is made, and an unhandled error here would abort the run.
    try:
        u = urllib.parse.urlsplit(url)
        host = (u.hostname or "").lower()
        port = u.port
        scheme = u.scheme
    except ValueError:
        return "unparseable"
    if not host:
        return "unparseable"
    if host in ("localhost", "127.0.0.1", "::1"):
        # Brackets are stripped by urlsplit and have to go back, or an IPv6
        # loopback address comes out unparseable.
        shown = f"[{host}]" if ":" in host else host
    else:
        shown = "<host>"
    return f"{scheme}://{shown}" + (f":{port}" if port else "")


def new_run_id():
    """Ties every row to the manifest describing the run that produced it.

    Without it the documented compare-two-arms workflow writes both arms to one
    CSV and overwrites the manifest, leaving rows from two server
    configurations described by whichever ran last.
    """
    return (time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" +
            uuid.uuid4().hex[:4])


def sampling_from(a):
    """Only what the user actually set, so the manifest records real choices."""
    out = {"temperature": a.temperature}
    for flag, key in (("top_p", "top_p"), ("top_k", "top_k"),
                      ("presence_penalty", "presence_penalty")):
        v = getattr(a, flag, None)
        if v is not None:
            out[key] = v
    return out


def manifest(a, srv, run_id):
    """What this run was measured on.

    The argument of the write-up behind this tool is that an unlabelled tok/s
    figure describes nobody's machine. A results file that cannot answer "what
    produced this" has the same problem, so every run writes one of these
    beside the CSV.
    """
    m = {"ctxbench_version": VERSION,
         "run_id": run_id,
         "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "command": a.cmd,
         "label": a.label,
         "url": safe_url(a.url),
         "model": a.model,
         "passes": a.passes,
         "max_tokens": getattr(a, "max_tokens", None),
         "sampling": sampling_from(a)}
    for k in ("depths", "suite", "suites", "effort", "efforts", "depth", "lengths"):
        if hasattr(a, k):
            m[k] = getattr(a, k)
    # Ask the server what it is rather than trusting the alias on the command
    # line, which is whatever the operator typed. /props is a GET.
    try:
        props = srv.get("/props")
        if isinstance(props, dict):
            # Only the filename. The full model_path is an absolute path on
            # the operator's machine, and this file is meant to be shared
            # alongside results.
            mp = props.get("model_path")
            if isinstance(mp, str):
                m["server_model_file"] = ntpath.basename(mp)
            for k in ("build_info", "total_slots", "n_ctx"):
                if props.get(k) is not None:
                    m["server_" + k] = props[k]
            tmpl = props.get("chat_template")
            if isinstance(tmpl, str):
                # The template decides what reasoning_effort means, so its
                # identity matters; its 10 KB of Jinja does not.
                m["server_chat_template_sha256"] = hashlib.sha256(
                    tmpl.encode()).hexdigest()[:16]
    except BenchError as e:
        # A URL that is not a llama.cpp server at all is worth saying out loud,
        # rather than recording it as though the server merely lacks /props.
        m["server_props"] = f"error: {str(e).replace(srv.url, safe_url(srv.url))}"
    except Exception:
        m["server_props"] = "unavailable"
    try:
        exe = safe_nvidia_smi()
        if exe:
            r = subprocess.run(
                [exe, "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
            if lines:
                m["gpus"] = lines
    except Exception:
        pass
    return m


def start_run(a, srv):
    """Everything a command does before its first measurement.

    Extracted because all three commands did it identically, and a fourth that
    forgot a line would ship rows with no manifest describing them -- which is
    the failure the run id exists to prevent.
    """
    run_id = new_run_id()
    # Writer first: it creates the output directory, and it refuses a CSV whose
    # columns do not match. Both should happen before a manifest exists, so a
    # rejected run does not leave one describing rows that were never written.
    out = Writer(a.out)
    write_manifest(a, srv, run_id)
    return run_id, out


def write_manifest(a, srv, run_id):
    # Keyed by run id, not by the CSV name: several arms routinely share one
    # CSV, and each needs its own record of what it ran against.
    path = f"{os.path.splitext(a.out)[0]}.{run_id}.manifest.json"
    with open(path, "w") as fh:
        json.dump(manifest(a, srv, run_id), fh, indent=2, default=str)
    print(f"  manifest -> {path}", flush=True)


def safe_nvidia_smi():
    """nvidia-smi, unless it resolved into the working directory.

    Windows searches the working directory before PATH, so a stray
    nvidia-smi.exe in a downloads folder would run instead. Defined once
    because a second copy of a security check is a second thing to remember to
    harden.
    """
    exe = shutil.which("nvidia-smi")
    if not exe or os.path.dirname(os.path.abspath(exe)) == os.getcwd():
        return None
    return exe


def gpu():
    """A snapshot taken just after the request returns, not during it.

    These columns are not an average over the generation: on a short request
    utilisation has often already collapsed by the time this samples. Treat
    them as "what the card looked like around then", not as workload telemetry.

    Absent on non-NVIDIA hosts, and that is fine.

    The working-directory check below is load-bearing, not a stray tidy-up:
    it is what makes this safe to call from a directory someone else can
    write to. Do not drop it while refactoring.
    """
    try:
        q = ("--query-gpu=power.draw,temperature.gpu,clocks.sm,"
             "utilization.gpu,memory.used")
        exe = safe_nvidia_smi()
        if not exe:
            return [""] * 5
        r = subprocess.run([exe, q, "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        # One row per GPU. On a multi-card host, splitting all of stdout on
        # commas yields a multiple of five fields and the caller's unpack
        # raises -- so return blanks rather than a row that cannot be read.
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if len(lines) != 1:
            return [""] * 5
        vals = [x.strip() for x in lines[0].split(",")]
        return vals if len(vals) == 5 else [""] * 5
    except Exception:
        return [""] * 5


# --------------------------------------------------------------------------
# prompts


def block(i):
    """One paragraph of filler whose content shifts with i.

    Rotating the vocabulary keeps the drafter from learning the filler, which
    would raise acceptance and flatter every number downstream of it.
    """
    w = _WORDS.split()
    rot = w[i % len(w):] + w[:i % len(w)]
    return (f"Section {i}. " + " ".join(rot) +
            f". Observation {i * 7 % 101}: the {rot[0]} interacts with the "
            f"{rot[3]} whenever {rot[5]} exceeds {i % 89}. "
            f"Record {i * 13 % 211} notes a {rot[2]} of {i % 47} units.\n")


def build_prompt(srv, depth, ask, chunk=200):
    """Filler grown coarsely then trimmed, so this costs a few tokenize calls.

    `depth` sizes the filler alone. The task prompt and the chat template's own
    overhead sit on top of it, so the context the server actually sees is a
    little larger -- that number is recorded per row as `ctx_tokens`, and is
    the one to read when interpreting results.
    """
    if depth == 0:
        return ask
    # A small target can be overshot by a whole chunk before the first check,
    # and trimming cannot always claw that back. Grow in smaller steps rather
    # than quietly returning a prompt at the wrong depth.
    chunk = max(1, min(chunk, depth // 20 or 1))
    parts, i, prev = [], 0, -1
    while True:
        parts.extend(block(i + k) for k in range(chunk))
        i += chunk
        n = srv.ntokens("".join(parts))
        if n >= depth:
            break
        # A /tokenize that returns a constant would otherwise grow this list
        # forever, POSTing a larger body each time. Stop rather than trust it.
        if n <= prev or i > MAX_FILLER_BLOCKS:
            raise BenchError(
                f"the server's token count stopped rising ({prev} then {n}) "
                f"before reaching depth {depth}; it is not counting the filler "
                f"this tool is sending")
        prev = n
    while n > depth * 1.02 and len(parts) > 1:
        parts.pop()
        n = srv.ntokens("".join(parts))
    if n > depth * 1.02:
        # One paragraph already exceeds the target. Say so: a row labelled
        # with a depth it never reached is worse than a failed run.
        raise BenchError(
            f"cannot build a prompt at depth {depth}: the smallest filler "
            f"block is {n} tokens. Use a depth above roughly {n * 2}.")
    return "".join(parts) + "\n\n" + ask


# --------------------------------------------------------------------------
# measurement


class BenchError(RuntimeError):
    """A failure with enough context to say which cell was running."""


def measure(srv, prompt, max_tokens, effort=None, thinking=None):
    resp, wall = srv.chat(prompt, max_tokens, effort, thinking)

    # A 200 with an unexpected shape is the dangerous case: a proxy, a
    # different backend, or a build without timing support. Recording a zero
    # rate here would put a fabricated number into every mean downstream, so
    # this refuses instead.
    try:
        choice = resp["choices"][0]
        msg = choice.get("message", {})
        tm = resp["timings"]
        decode_tps = tm["predicted_per_second"]
    except (KeyError, IndexError, TypeError) as e:
        raise BenchError(
            f"server returned a body this tool cannot read ({e!r}). It needs "
            f"an OpenAI-shaped reply with a `timings` block; llama.cpp emits "
            f"one, most proxies in front of it do not. Got keys: "
            f"{printable(sorted(resp)[:8]) if isinstance(resp, dict) else type(resp).__name__}")

    think = msg.get("reasoning_content") or ""
    answer = msg.get("content") or ""
    dn, da = tm.get("draft_n"), tm.get("draft_n_accepted")
    power, temp, clock, util, vram = gpu()
    return {
        "ctx_tokens": (tm.get("prompt_n") or 0) + (tm.get("cache_n") or 0),
        "prompt_n": tm.get("prompt_n"),
        "cache_n": tm.get("cache_n"),
        "gen_tok": tm.get("predicted_n"),
        "decode_tps": round(decode_tps, 3),
        # Prefill rate is genuinely absent on a fully cached prompt, so unlike
        # the decode rate it is allowed to be missing.
        "prefill_tps": (round(tm["prompt_per_second"], 2)
                        if tm.get("prompt_per_second") is not None else ""),
        # Absent entirely when speculative decoding is off, so every consumer
        # of these has to treat them as optional rather than assume a number.
        "draft_n": dn,
        "draft_acc": da,
        "accept_pct": round(100 * da / dn, 1) if dn else "",
        "reason_chars": len(think),
        "content_chars": len(answer),
        # A reply that never starts is not a reply, whatever the rate says.
        "answered": 1 if answer.strip() else 0,
        "finish": resp["choices"][0].get("finish_reason"),
        "wall_s": round(wall, 2),
        "power_w": power, "temp_c": temp, "clock_mhz": clock,
        "util_pct": util, "vram_used_mib": vram,
    }


class Writer:
    def __init__(self, path):
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        if not new:
            with open(path, newline="") as fh:
                header = next(csv.reader(fh), [])
            if header != COLUMNS:
                raise BenchError(
                    f"{path} was written with a different column set, probably "
                    f"by another version of this tool. Appending would leave "
                    f"two schemas in one file. Use a new --out.")
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self.fh = open(path, "a", newline="")
        # No extrasaction="ignore": if a field is added to measure() and not to
        # COLUMNS, that should fail at the first write rather than drop the
        # column silently for the rest of the run.
        self.w = csv.DictWriter(self.fh, fieldnames=COLUMNS)
        if new:
            self.w.writeheader()

    def row(self, **kw):
        unknown = set(kw) - set(COLUMNS)
        if unknown:
            raise BenchError(f"columns missing from COLUMNS: {sorted(unknown)}")
        base = {c: "" for c in COLUMNS}
        base.update(ts=int(time.time()), version=VERSION, **kw)
        self.w.writerow({k: safe_cell(v) for k, v in base.items()})
        self.fh.flush()

    def close(self):
        self.fh.close()


def show(r, prefix):
    acc = f"acc={r['accept_pct']:>5}%" if r["accept_pct"] != "" else "acc=  n/a"
    print(f"    {prefix} ctx={r['ctx_tokens']:>7,} {r['decode_tps']:>6.2f} tok/s "
          f"{acc}  {'ANSWERED' if r['answered'] else 'NO ANSWER'}  "
          f"{r['wall_s']:>6.1f}s", flush=True)


# --------------------------------------------------------------------------
# experiments


def cmd_depth(a):
    srv = Server(a.url, a.model, sampling=sampling_from(a))
    depths = [int(d) for d in a.depths.split(",")]
    ask = PROMPTS.get(a.suite, a.suite)

    print(f"  building prompts for '{a.label}'", flush=True)
    prompts = {}
    for d in depths:
        prompts[d] = build_prompt(srv, d, ask)
        print(f"    depth {d:>7,} -> {srv.ntokens(prompts[d]):>7,} actual tokens",
              flush=True)

    run_id, out = start_run(a, srv)
    try:
        for p in range(a.passes):
            # Rotate, do not just reverse: with more than two depths a straight
            # reverse still puts the same rows at the hot end half the time.
            order = depths[p % len(depths):] + depths[:p % len(depths)]
            for d in order:
                r = measure(srv, prompts[d], a.max_tokens, a.effort or None)
                if r["gen_tok"] != a.max_tokens:
                    # The cap is the only thing holding reply length still. A
                    # model that stops early at one depth and not another has
                    # varied two things at once, and reply length moves tok/s
                    # on its own -- which is what `ab` exists to measure.
                    # Said mid-run, before the other depths exist, so it
                    # cannot yet claim the rows disagree with each other.
                    print(f"    warning: depth {d:,} stopped at "
                          f"{r['gen_tok']} tokens rather than the "
                          f"{a.max_tokens} requested; check gen_tok is "
                          f"comparable across depths before reading these "
                          f"rates", file=sys.stderr, flush=True)
                out.row(run_id=run_id, label=a.label, **{"pass": p + 1}, suite=a.suite,
                        effort=a.effort, depth_target=d,
                        max_tokens=a.max_tokens, **r)
                show(r, f"p{p+1} d={d:>7,}")
    finally:
        out.close()


def cmd_grid(a):
    """Every effort level against every workload, at one depth.

    Budget matters more than it looks. Too small a cap and the reasoning arms
    spend the whole budget thinking, so you measure the decode rate of
    deliberation and call it generation.
    """
    srv = Server(a.url, a.model, sampling=sampling_from(a))
    suites = a.suites.split(",")
    efforts = a.efforts.split(",")
    cells = [(s, e) for e in efforts for s in suites]
    # Built once per suite, not once per cell per pass: depth is fixed here, so
    # rebuilding costs a round of tokenize calls for an identical string.
    prompts = {s: build_prompt(srv, a.depth, PROMPTS.get(s, s)) for s in suites}

    run_id, out = start_run(a, srv)
    try:
        for p in range(a.passes):
            order = cells if p % 2 == 0 else cells[::-1]
            for suite, effort in order:
                # "off" is this tool's word for "no thinking", sent the way the
                # template actually reads it rather than as an effort level.
                thinking = False if effort == "off" else None
                r = measure(srv, prompts[suite], a.max_tokens,
                            None if effort == "off" else effort, thinking)
                out.row(run_id=run_id, label=a.label, **{"pass": p + 1}, suite=suite,
                        effort=effort, depth_target=a.depth,
                        max_tokens=a.max_tokens, **r)
                show(r, f"p{p+1} {suite:<5} {effort:<7}")
    finally:
        out.close()


def cmd_ab(a):
    """Content and reply length, each varied with the other held still.

    Kept separate on purpose. Reporting the spread between the fastest and
    slowest cell as one effect attributes to content what reply length did.
    """
    srv = Server(a.url, a.model, sampling=sampling_from(a))
    lengths = [int(x) for x in a.lengths.split(",")]
    suites = a.suites.split(",")
    cells = ([(s, lengths[0]) for s in suites] +
             [(suites[0], n) for n in lengths[1:]])
    prompts = {s: build_prompt(srv, a.depth, PROMPTS.get(s, s)) for s in suites}

    run_id, out = start_run(a, srv)
    try:
        for p in range(a.passes):
            order = cells if p % 2 == 0 else cells[::-1]
            for suite, n in order:
                r = measure(srv, prompts[suite], n, a.effort or None)
                out.row(run_id=run_id, label=a.label, **{"pass": p + 1}, suite=suite,
                        effort=a.effort, depth_target=a.depth,
                        max_tokens=n, **r)
                show(r, f"p{p+1} {suite:<5} {n:>5} tok")
    finally:
        out.close()


# --------------------------------------------------------------------------
# report


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("decode_tps", "accept_pct", "prefill_tps", "wall_s"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
        for k in ("depth_target", "gen_tok", "answered", "max_tokens"):
            r[k] = int(r[k]) if r[k] not in ("", None) else None
        # Writing a sanitised CSV does not make reading one safe. This file may
        # have come from a colleague, an older build, or a text editor, and
        # report prints these three straight to a terminal.
        for k in ("label", "suite", "effort"):
            r[k] = printable(r.get(k) or "", limit=60)
    return rows


def agg(rows):
    t = [r["decode_tps"] for r in rows]
    acc = [r["accept_pct"] for r in rows if r["accept_pct"] is not None]
    return {
        "n": len(rows),
        "tps": st.mean(t),
        # None, not 0.0. A single sample has no spread, and calling that zero
        # collapses the noise floor so that any difference at all reads as
        # real -- in the same confident format as a properly replicated one.
        "sd": st.stdev(t) if len(t) > 1 else None,
        "acc": st.mean(acc) if acc else None,
        "answered": sum(r["answered"] for r in rows),
        "wall": st.mean([r["wall_s"] for r in rows]),
    }


def group(rows, *keys):
    d = {}
    for r in rows:
        d.setdefault(tuple(r[k] for k in keys), []).append(r)
    return d


def cmd_report(a):
    rows = load(a.csv)
    if a.label:
        rows = [r for r in rows if r["label"] == a.label]
    if not rows:
        print("no rows")
        return

    by_label = group(rows, "label")
    for (label,), rs in sorted(by_label.items()):
        print(f"\n=== {label} ===")
        depths = sorted({r["depth_target"] for r in rs})
        cells = sorted({(r["suite"], r["effort"], r["max_tokens"]) for r in rs})

        if len(depths) > 1:
            if len(cells) > 1:
                print(f"  label '{label}' holds more than one kind of run "
                      f"({len(cells)} suite/effort/max-token combinations), so "
                      f"a depth table would average them together. Use a "
                      f"separate --label per experiment.")
            else:
                # Percentages are against the shallowest row present, which is
                # not always 0 once --depths is customised.
                base_depth = depths[0]
                print(f"  {'depth':>9}{'tok/s':>10}{'sd':>7}{'accept':>9}"
                      f"{('vs ' + format(base_depth, ',')):>14}")
                base = None
                for d in depths:
                    g = agg([r for r in rs if r["depth_target"] == d])
                    base = base or g["tps"]
                    acc = f"{g['acc']:.1f}%" if g["acc"] is not None else "n/a"
                    sd = f"{g['sd']:.2f}" if g["sd"] is not None else "-"
                    print(f"  {d:>9,}{g['tps']:>10.2f}{sd:>7}{acc:>9}"
                          f"{100 * (g['tps'] / base - 1):>13.1f}%")

        if len(cells) > 1:
            print(f"  {'suite':<8}{'effort':<9}{'maxtok':>8}{'tok/s':>10}"
                  f"{'sd':>7}{'accept':>9}{'answered':>10}{'wall':>8}")
            for suite, effort, mt in cells:
                g = agg([r for r in rs if (r["suite"], r["effort"],
                                           r["max_tokens"]) == (suite, effort, mt)])
                acc = f"{g['acc']:.1f}%" if g["acc"] is not None else "n/a"
                sd = f"{g['sd']:.2f}" if g["sd"] is not None else "-"
                print(f"  {suite:<8}{effort or '-':<9}{mt:>8}{g['tps']:>10.2f}"
                      f"{sd:>7}{acc:>9}"
                      f"{g['answered']:>7}/{g['n']}{g['wall']:>7.0f}s")

    labels = sorted({r["label"] for r in rows})
    if len(labels) == 2:
        print(f"\n=== {labels[1]} against {labels[0]} ===")
        print("  a difference counts only if it clears both standard deviations")
        print("  added together -- a deliberately conservative floor, not a")
        print("  significance test")
        a_rows, b_rows = ([r for r in rows if r["label"] == x] for x in labels)

        def cell(r):
            return (r["suite"], r["effort"], r["max_tokens"], r["depth_target"])

        # Only compare cells the two labels genuinely share. Two labels in one
        # CSV are often two different experiments rather than two arms of one,
        # and lining those up by depth alone produces a confident-looking table
        # of nonsense.
        shared = {cell(r) for r in a_rows} & {cell(r) for r in b_rows}
        if not shared:
            print("  no cells in common; nothing to compare")
            return
        for c in sorted(shared, key=lambda c: c[3]):
            ga = agg([r for r in a_rows if cell(r) == c])
            gb = agg([r for r in b_rows if cell(r) == c])
            delta = 100 * (gb["tps"] / ga["tps"] - 1)
            if ga["sd"] is None or gb["sd"] is None:
                # One repeat gives a mean and no spread, so there is no floor
                # to clear. Saying "real" here would be the tool's own headline
                # claim asserted from a single sample.
                verdict = f"no floor (n={min(ga['n'], gb['n'])}, need 2+)"
            else:
                verdict = ("real" if abs(gb["tps"] - ga["tps"]) >
                           (ga["sd"] + gb["sd"]) else "noise")
            suite, effort, mt, d = c
            print(f"  {suite:<6}{effort or '-':<8}{mt:>7}tok{d:>9,}"
                  f"{delta:>9.1f}%   {verdict}")


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"ctx-bench {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--url", default=os.environ.get("CTXBENCH_URL",
                                                       "http://127.0.0.1:8080"))
        p.add_argument("--model", default=os.environ.get("CTXBENCH_MODEL", "default"))
        p.add_argument("--out", default="results.csv")
        # Greedy by default so repeats are comparable. Overridable because a
        # model's published preset is part of the operating point, and some
        # publish different ones for thinking and non-thinking modes.
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--top-p", type=float, default=None, dest="top_p")
        p.add_argument("--top-k", type=int, default=None, dest="top_k")
        p.add_argument("--presence-penalty", type=float, default=None,
                       dest="presence_penalty")
        p.add_argument("--label", default="run",
                       help="names this arm in the CSV; compare two arms by "
                            "running twice with different labels")
        p.add_argument("--passes", type=int, default=2,
                       help="repeats of the whole cell set, order rotated each "
                            "time; drift on a single card can reach 10%%, so one "
                            "pass is not a measurement")

    p = sub.add_parser("depth", help="decode rate against context depth")
    common(p)
    p.add_argument("--depths", default="0,8192,32768,65536,110592")
    p.add_argument("--suite", default="code", help="code, prose, or literal text")
    p.add_argument("--effort", default="")
    p.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    p.set_defaults(func=cmd_depth)

    p = sub.add_parser("grid", help="reasoning effort against workload")
    common(p)
    p.add_argument("--suites", default="code,prose")
    p.add_argument("--efforts", default="off,low,medium,xhigh",
                   help="'off' disables thinking via chat_template_kwargs; the "
                        "rest are sent as reasoning_effort and must be levels "
                        "your chat template accepts")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=8000, dest="max_tokens")
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("ab", help="content type and reply length")
    common(p)
    p.add_argument("--suites", default="code,prose")
    p.add_argument("--lengths", default="200,500")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--effort", default="")
    p.set_defaults(func=cmd_ab)

    p = sub.add_parser("report", help="summarise a results CSV")
    p.add_argument("csv")
    p.add_argument("--label", default=None)
    p.set_defaults(func=cmd_report)

    a = ap.parse_args()
    try:
        a.func(a)
    except urllib.error.HTTPError as e:
        # Only ever displayed, and printable() truncates to 400 characters, so
        # there is nothing to gain from reading more than a few KiB of it.
        raw = e.read(8192)
        if looks_html(raw):
            # An HTML error page means the URL is wrong, and its contents tell
            # the reader nothing they can act on.
            print(f"\n{not_a_server(getattr(a, 'url', ''), '', raw)}",
                  file=sys.stderr)
            return 1
        body = printable(raw.decode("utf-8", "replace"))
        print(f"\nserver returned HTTP {e.code}\n{body}", file=sys.stderr)
        if "reasoning effort" in body:
            print("\nThe chat template rejected that effort level. Templates "
                  "accept different sets, and llama.cpp's --help advertises "
                  "levels your template may not take.", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"\ncannot reach {getattr(a, 'url', 'the server')}: {e.reason}",
              file=sys.stderr)
        return 1
    except BenchError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; rows already written are still in the CSV")
        return 130
    except Exception as e:
        # A run can die hours in. A bare traceback does not say which cell was
        # in flight, which is the one thing worth knowing at that point.
        print(f"\nunexpected failure during '{a.cmd}'"
              f"{' (label ' + a.label + ')' if getattr(a, 'label', None) else ''}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
