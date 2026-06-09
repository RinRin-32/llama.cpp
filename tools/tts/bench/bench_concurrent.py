#!/usr/bin/env python3
"""Benchmark #3: concurrency scaling for llama-qwen3tts.

Launches N identical synthesis jobs at once (separate processes) and reports, per
concurrency level: aggregate throughput, per-request latency percentiles, generation
real-time factor, time-to-first-byte (streaming only), and success rate. A single
process underutilizes the GPU, so concurrent requests should raise total throughput
for only a modest per-request slowdown.

Each process reloads the model, so the per-request *total wall* includes a fixed
model-load cost; the CLI's own "Real-time factor" / "TTFB" are generation-only
(post-load) and are the numbers to trust for steady-state serving.

Run inside the container, e.g.:
  python tools/tts/bench/bench_concurrent.py \
    --bin llama-qwen3tts \
    --talker  /models/qwen3tts-talker-bf16.gguf \
    --cp      /models/qwen3tts-cp-bf16.gguf \
    --vocoder /models/qwen3tts-tokenizer-f16.gguf \
    --concurrency 1,2,3,4 --reps 1 --n-gpu-layers 99 \
    --stream-chunk 3 --out /output/bench_concurrent
"""
import argparse, subprocess, time, re, csv, os, statistics, tempfile

TEXT = ("The quick brown fox jumps over the lazy dog, and then it runs back again "
        "across the wide green field under a bright afternoon sky.")


def parse(out: str) -> dict:
    d = {}
    m = re.search(r"Decode:\s+(\d+)\s+frames in\s+([\d.]+)\s*ms", out)
    if m:
        d["frames"] = int(m.group(1)); d["decode_s"] = float(m.group(2)) / 1000.0
    m = re.search(r"Real-time factor:\s+([\d.]+)x\s+\(([\d.]+)s audio", out)
    if m:
        d["rtf"] = float(m.group(1)); d["audio_s"] = float(m.group(2))
    m = re.search(r"Streaming: TTFB\s+([\d.]+)\s*ms", out)
    if not m:
        m = re.search(r"first audio chunk ready at\s+([\d.]+)\s*ms", out)
    if m:
        d["ttfb_ms"] = float(m.group(1))
    return d


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def run_level(args, concurrency: int) -> dict:
    base = [args.bin, "--model-talker", args.talker, "--model-cp", args.cp,
            "--model-vocoder", args.vocoder, "--text", TEXT,
            "--n-gpu-layers", str(args.n_gpu_layers),
            "--cp-n-gpu-layers", str(args.n_gpu_layers),
            "--seed", "42", "--max-tokens", str(args.max_tokens)]
    if args.stream_chunk > 0:
        base += ["--stream-chunk", str(args.stream_chunk),
                 "--stream-left-ctx", str(args.stream_left_ctx)]

    walls, decodes, rtfs, audios, ttfbs = [], [], [], [], []
    ok = 0
    n_jobs = concurrency * args.reps
    batch_t0 = time.time()
    # Fire all jobs, each in its own process, then collect. The CLI logs one line per frame,
    # so route each child's output to its own temp file rather than a pipe -- with many
    # concurrent children, sequential pipe reads would deadlock on full pipe buffers.
    procs = []
    for j in range(n_jobs):
        cmd = base + ["--output", f"{args.out}_c{concurrency}_j{j}.wav"]
        log = tempfile.NamedTemporaryFile("w+", suffix=f"_c{concurrency}_j{j}.log", delete=False)
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        procs.append((p, t0, log))
    for p, t0, log in procs:
        p.wait()
        wall = time.time() - t0
        log.seek(0)
        out = log.read()
        log.close()
        os.unlink(log.name)
        d = parse(out)
        if p.returncode == 0 and d.get("frames", 0) > 0:
            ok += 1
            walls.append(wall)
            if "decode_s" in d: decodes.append(d["decode_s"])
            if "rtf" in d: rtfs.append(d["rtf"])
            if "audio_s" in d: audios.append(d["audio_s"])
            if "ttfb_ms" in d: ttfbs.append(d["ttfb_ms"])
    batch_wall = time.time() - batch_t0

    med = lambda xs: statistics.median(xs) if xs else float("nan")
    row = {
        "concurrency": concurrency,
        "jobs": n_jobs,
        "success": ok,
        "batch_wall_s": round(batch_wall, 2),
        # Aggregate audio produced per wall-second (incl. per-process model load).
        "agg_throughput_rtf": round(sum(audios) / batch_wall, 2) if audios else float("nan"),
        "gen_rtf_med": round(med(rtfs), 2),          # generation-only, per request
        "lat_p50_s": round(pct(walls, 50), 2),
        "lat_p90_s": round(pct(walls, 90), 2),
        "lat_p99_s": round(pct(walls, 99), 2),
        "lat_max_s": round(max(walls), 2) if walls else float("nan"),
        "ttfb_med_ms": round(med(ttfbs), 0) if ttfbs else "",
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--talker", required=True)
    ap.add_argument("--cp", required=True)
    ap.add_argument("--vocoder", required=True)
    ap.add_argument("--out", default="bench_concurrent")
    ap.add_argument("--concurrency", default="1,2,3,4")
    ap.add_argument("--reps", type=int, default=1, help="jobs per client at each level")
    ap.add_argument("--n-gpu-layers", default="99")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--stream-chunk", type=int, default=0, help=">0 enables streaming + TTFB")
    ap.add_argument("--stream-left-ctx", type=int, default=32)
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    rows = []
    for c in levels:
        row = run_level(args, c)
        rows.append(row)
        print(f"C={row['concurrency']:2d} ok={row['success']}/{row['jobs']} "
              f"agg_thru={row['agg_throughput_rtf']}x  gen_rtf(med)={row['gen_rtf_med']}x  "
              f"lat p50/p90/p99={row['lat_p50_s']}/{row['lat_p90_s']}/{row['lat_p99_s']}s  "
              f"ttfb={row['ttfb_med_ms']}ms", flush=True)

    csv_path = args.out + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", csv_path)

    # Speedup vs single-request baseline (aggregate throughput).
    base = next((r["agg_throughput_rtf"] for r in rows if r["concurrency"] == 1), None)
    if base:
        for r in rows:
            r["_speedup"] = r["agg_throughput_rtf"] / base
        print("throughput speedup vs C=1: " +
              ", ".join(f"C{r['concurrency']}={r['_speedup']:.2f}x" for r in rows))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [r["concurrency"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(x, [r["agg_throughput_rtf"] for r in rows], "o-", color="tab:green")
        ax[0].set_xlabel("concurrent requests"); ax[0].set_ylabel("aggregate throughput (xRT)")
        ax[0].set_title("Throughput vs concurrency"); ax[0].grid(True, alpha=0.3)
        ax[1].plot(x, [r["lat_p50_s"] for r in rows], "o-", label="p50")
        ax[1].plot(x, [r["lat_p90_s"] for r in rows], "s-", label="p90")
        ax[1].plot(x, [r["lat_p99_s"] for r in rows], "^-", label="p99")
        ax[1].set_xlabel("concurrent requests"); ax[1].set_ylabel("per-request wall (s)")
        ax[1].set_title("Latency vs concurrency"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        png = args.out + ".png"; fig.savefig(png, dpi=120)
        print("wrote", png)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
