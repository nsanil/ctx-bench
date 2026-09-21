# ctx-bench

`ctx-bench` measures llama.cpp decode throughput at realistic context depths and
workloads, rather than only on a fresh prompt. It sweeps context depth, compares
reasoning effort levels, and isolates content-type and reply-length effects,
recording llama.cpp's own timings, speculative-draft acceptance, and optional
GPU telemetry.

Python 3.8+, standard library only. No dependencies, one file.

## Quick start

You need a llama.cpp server already running. The tool does not start or
reconfigure one: launch flags vary too much between machines, so you manage the
server separately.

```bash
llama-server -m model.gguf -c 131072 --host 127.0.0.1 --port 8080
```

```bash
export CTXBENCH_URL=http://127.0.0.1:8080
export CTXBENCH_MODEL=your-model-alias

./ctxbench.py depth --label mine
./ctxbench.py report results.csv
```

Real output, from `--depths 0,16384,65536,110592 --max-tokens 500 --passes 3`
rather than the defaults, to keep the example short:

```text
=== mine ===
      depth     tok/s     sd   accept  vs depth 0
          0     62.88   0.20    68.9%        0.0%
     16,384     63.61   1.65    77.0%        1.2%
     65,536     52.75   0.72    75.1%      -16.1%
    110,592     44.47   0.16    71.5%      -29.3%
```

The size of that drop depends on the model, the card, and what you asked it to
generate. The point is that it exists and a fresh-prompt benchmark cannot see
it. Note the 16,384 row: the apparent `+1.2%` is smaller than the
run-to-run spread, so it is not a speed-up at depth. That is what the `sd`
column is there for.

**Your server's context must exceed your deepest `--depths` value.** The default
ladder reaches 110,592 tokens, so a server started on a smaller context will
not be able to run the deeper rows as requested.

## Commands

| command | purpose |
|---|---|
| `depth` | sweep decode throughput across increasing context depth |
| `grid` | compare reasoning effort levels across workloads |
| `ab` | isolate content-type and reply-length effects |
| `report` | aggregate a results CSV and compare labels |

```bash
./ctxbench.py grid --label mine-grid --max-tokens 8000
./ctxbench.py ab   --label mine-ab
```

Give `grid` a generous `--max-tokens`. At a small cap the reasoning arms spend
the whole budget thinking, and you measure the decode rate of deliberation
while believing you measured generation.

`ab` varies content and reply length one at a time and reports them separately,
because folding them into a single spread attributes to content what reply
length did.

## Requirements

Two endpoints, both needed:

| endpoint | why |
|---|---|
| `/v1/chat/completions` | the measurement, and the `timings` block it returns |
| `/tokenize` | sizing filler by the server's own tokenizer instead of guessing |

`/tokenize` is not part of the OpenAI API, so a generic OpenAI-compatible proxy
will fail there, and one that strips `timings` is refused rather than measured.

Acceptance columns stay blank without speculative decoding, and GPU telemetry
stays blank without `nvidia-smi` on PATH. Neither is required; both are absent
rather than wrong.

## Comparing configurations

Run the same command twice with different `--label` values, restarting the
server in between:

```bash
./ctxbench.py depth --label n4 --out compare.csv
# restart the server with the setting changed
./ctxbench.py depth --label n2 --out compare.csv
./ctxbench.py report compare.csv
```

`report` marks a difference as real only if it clears both labels' standard
deviations added together — a deliberately conservative floor, not a
significance test. A cell with one repeat has no spread, so it reports no floor
rather than calling every difference real.

**That floor does not protect a comparison that crosses a restart.** It is
computed from the spread within each arm, and two arms run back to back are
separated by drift neither arm can see. Run the first configuration again
afterwards as a control:

```bash
./ctxbench.py depth --label n4-before --out compare.csv
# restart with the setting changed
./ctxbench.py depth --label n2        --out compare.csv
# restart back
./ctxbench.py depth --label n4-after  --out compare.csv
```

If the before/after control differs materially, treat the middle arm as
suspect. On a single consumer card these short runs have differed by 1–2% with
no configuration change at all, which is a reasonable warning threshold to
start from.

## Methodology notes

**Sampling parameters are part of the measurement.** Several models publish
different presets for thinking and non-thinking modes. Benchmarking one mode
with the other's preset is easy to do and hard to notice.

**Effort level names are not portable.** llama.cpp's `--help` advertises
`minimal` through `max`, but your chat template decides which of those it
honours, which it silently rewrites, and which it rejects. Check the template
you downloaded, not the runtime's help text. The tool prints a hint when the
server rejects a level.

**Turning thinking off may be a template variable rather than a request
field.** On llama.cpp servers using Qwen-family chat templates, `enable_thinking`
travels in `chat_template_kwargs` rather than as a top-level request field; sent
at the top level it is accepted and ignored, which looks exactly like a model
that will not comply. `grid`'s `off` arm sends it the working way. Newer builds
may differ.

**Run more than one pass.** Order rotates between passes, because a ladder that
always ascends measures its deepest point when the card is hottest and then
reports heat as depth.

**Filler is generated, not repeated.** Repeating one paragraph to reach depth
raises draft acceptance and flatters every number downstream, so the filler
rotates its vocabulary.

## Reading the output

`report` prints a mean and standard deviation per cell.

`accept` is speculative-draft acceptance, blank when the server runs without a
drafter. On speculative-decoding runs it often explains a large share of any
throughput difference, so it is worth checking first when tok/s moves.

`finished` counts replies that produced an answer. A row can report a healthy
tok/s having generated nothing but reasoning.

## Failure behavior

`--url` can point anywhere, so the server's responses are treated as untrusted.
The tool aborts rather than silently continuing when:

- a response exceeds 64 MiB
- `/tokenize` stops counting the filler being sent to it
- a single filler block already overshoots a shallow requested depth
- the reply is not JSON, or has no `timings` block

Rows already written stay in the CSV, and the message says which case it was.

## Output

One CSV, one row per request. Columns cover the request (`suite`, `effort`,
`depth_target`, `max_tokens`), what the server reported (`decode_tps`,
`prefill_tps`, `prompt_n`, `cache_n`, draft counters), what came back
(`reason_chars`, `content_chars`, `answered`, `finish`), and host telemetry.

Results append to the same file, so several runs can share one CSV. Use
`--label` to separate configurations. Be cautious comparing runs from different
sessions: machine drift can exceed the effects you are trying to measure.

## How this differs from `llama-bench`

llama.cpp ships `llama-bench`, and it does more than this tool for hardware
tuning: it sweeps `-ngl`, cache types, batch sizes, tensor splits, and it has a
`-d/--n-depth` flag, so depth benchmarking is not new here. Use it for
comparing builds, quantizations and offload settings. It is faster and more
thorough at that job.

Three things it does not do, which is why this exists:

**It benchmarks a model it loads itself, not the server your application is
using.** `llama-bench -m model.gguf` creates its own benchmark process. Many
server settings can be reproduced there, but you have to do so explicitly, and
it is still not exercising the runtime instance your application talks to. This
tool measures the server that is already up, through the same API.

It also states that its measurements exclude tokenization and sampling time.
This reads the timings the server reports for a real request, which is a
different quantity rather than a better one.

**It has no application workload or chat template.** `-p 512 -n 128`
benchmarks synthetic token processing and generation, not a particular prompt.
There is no code-versus-prose workload, no `reasoning_effort`, and no
distinction between a completed reply and a model that spent its generation
budget reasoning.

**It currently does not expose speculative decoding.** There is no
draft-model or `--spec-type` option, so it cannot report draft acceptance. On a
speculative setup that is usually the term that explains why one configuration
is faster than another. llama.cpp moves quickly, so check before relying on
this one.

So: `llama-bench` for what the hardware and the build can do, this for what
your running server does on the traffic you actually send it.

## Limitations

Measures one llama.cpp server over HTTP. It manages no servers, downloads no
models, and configures no GPUs — power limits and quantization are yours to set
and yours to record. Nothing here is specific to a model family except the
default prompts, which are `--suite` arguments and can be any text.

MIT.
