#!/usr/bin/env python3
"""Benchmark #1: end-to-end latency vs input text length for llama-qwen3tts.

Runs the CLI over a range of text lengths, measures wall-clock time to completion,
and plots latency (total wall + generation-only) and real-time factor vs token count.

Run inside the container, e.g.:
  python tools/tts/bench/bench_latency.py \
    --bin ./build/bin/llama-qwen3tts \
    --talker /workspace/models/qwen3tts-talker-f16.gguf \
    --cp /workspace/models/qwen3tts-cp-f16.gguf \
    --vocoder /workspace/models/qwen3tts-tokenizer-f16.gguf \
    --out /workspace/models/bench_latency
"""
import argparse, subprocess, time, re, csv, os

BASE = ("The quick brown fox jumps over the lazy dog. ")


def make_text(n_sentences: int) -> str:
    return (BASE * n_sentences).strip()


def parse(out: str):
    d = {}
    m = re.search(r"Decode:\s+(\d+)\s+frames in\s+([\d.]+)\s*ms", out)
    if m:
        d["frames"] = int(m.group(1)); d["decode_ms"] = float(m.group(2))
    m = re.search(r"CP:\s+([\d.]+)\s*ms total", out);  d["cp_ms"] = float(m.group(1)) if m else None
    m = re.search(r"Real-time factor:\s+([\d.]+)x\s+\(([\d.]+)s audio", out)
    if m:
        d["rtf"] = float(m.group(1)); d["audio_s"] = float(m.group(2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--talker", required=True)
    ap.add_argument("--cp", required=True)
    ap.add_argument("--vocoder", required=True)
    ap.add_argument("--out", default="bench_latency")
    ap.add_argument("--n-gpu-layers", default="99")
    ap.add_argument("--sentences", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--reps", type=int, default=2, help="repeats per length (median reported)")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sentences.split(",")]
    rows = []
    for n in sizes:
        text = make_text(n)
        ntok_est = len(text.split())
        totals, decodes, rtfs, audios, frames = [], [], [], [], []
        for r in range(args.reps):
            cmd = [args.bin, "--model-talker", args.talker, "--model-cp", args.cp,
                   "--model-vocoder", args.vocoder, "--text", text,
                   "--output", f"{args.out}_n{n}.wav", "--n-gpu-layers", args.n_gpu_layers,
                   "--seed", "42", "--max-tokens", "1024"]
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.time() - t0
            d = parse(p.stdout + p.stderr)
            totals.append(wall)
            if "decode_ms" in d: decodes.append(d["decode_ms"] / 1000.0)
            if "rtf" in d: rtfs.append(d["rtf"]); audios.append(d["audio_s"])
            if "frames" in d: frames.append(d["frames"])
        med = lambda xs: sorted(xs)[len(xs)//2] if xs else float("nan")
        row = {"sentences": n, "words": ntok_est, "frames": med(frames),
               "audio_s": med(audios), "total_wall_s": med(totals),
               "decode_s": med(decodes), "rtf": med(rtfs)}
        rows.append(row)
        print(f"n={n:2d} words={ntok_est:3d} frames={row['frames']} "
              f"audio={row['audio_s']:.2f}s total={row['total_wall_s']:.2f}s "
              f"decode={row['decode_s']:.2f}s rtf={row['rtf']:.2f}x")

    csv_path = args.out + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", csv_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [r["words"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(x, [r["total_wall_s"] for r in rows], "o-", label="total wall (incl. model load)")
        ax[0].plot(x, [r["decode_s"] for r in rows], "s-", label="generation only (decode)")
        ax[0].plot(x, [r["audio_s"] for r in rows], "^--", label="audio duration", alpha=0.6)
        ax[0].set_xlabel("input length (words)"); ax[0].set_ylabel("seconds")
        ax[0].set_title("Qwen3-TTS latency vs text length (0.6B, Vulkan, CP on GPU)")
        ax[0].legend(); ax[0].grid(True, alpha=0.3)
        ax[1].plot(x, [r["rtf"] for r in rows], "o-", color="tab:red")
        ax[1].axhline(1.0, ls="--", color="gray", label="real-time")
        ax[1].set_xlabel("input length (words)"); ax[1].set_ylabel("real-time factor (x)")
        ax[1].set_title("Real-time factor vs text length"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        png = args.out + ".png"; fig.savefig(png, dpi=120)
        print("wrote", png)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
