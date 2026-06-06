#!/usr/bin/env python3
"""Plot TTFB-vs-chunk and concurrency results (data captured from bench runs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "bench_ttfb_concurrency.png"

# TTFB vs streaming chunk size (5.44s clip, 0.6B, Vulkan, CP on GPU)
chunk = [2, 4, 8, 16, 32]
ttfb_ms = [1274, 1293, 1270, 2453, 4870]
full_ms = 5440 * 1.43  # ~full generation wall (RTF ~0.70x)

# Concurrency (parallel synthesis processes, single 9070 XT)
N = [1, 2, 3]
wall = [9.03, 9.59, 10.28]
per_req = [5.57, 5.80, 5.82]
throughput = [n / (w / wall[0]) for n, w in zip(N, wall)]  # rel. to N=1 wall

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

ax[0].plot(chunk, ttfb_ms, "o-", color="tab:blue")
ax[0].axhline(full_ms, ls="--", color="gray", label=f"full clip (~{full_ms/1000:.1f}s)")
ax[0].set_xlabel("streaming chunk size (frames)"); ax[0].set_ylabel("time to first audio (ms)")
ax[0].set_title("TTFB vs chunk size\n(floor ~1.27s = 6-frame quality margin)")
ax[0].legend(); ax[0].grid(True, alpha=0.3)

ax[1].plot(N, wall, "o-", label="batch wall time", color="tab:red")
ax[1].plot(N, per_req, "s-", label="per-request latency", color="tab:orange")
ax[1].set_xlabel("concurrent requests"); ax[1].set_ylabel("seconds")
ax[1].set_title("Wall time vs concurrency (single 9070 XT)")
ax[1].set_xticks(N); ax[1].legend(); ax[1].grid(True, alpha=0.3); ax[1].set_ylim(0, 12)

ax[2].plot(N, throughput, "o-", color="tab:green", label="actual")
ax[2].plot(N, N, "--", color="gray", label="ideal linear")
ax[2].set_xlabel("concurrent requests"); ax[2].set_ylabel("throughput (x vs N=1)")
ax[2].set_title("Throughput scaling\n(dispatch-bound CP overlaps well)")
ax[2].set_xticks(N); ax[2].legend(); ax[2].grid(True, alpha=0.3)

fig.suptitle("Qwen3-TTS-0.6B on llama.cpp / RX 9070 XT (Vulkan): streaming TTFB + concurrency", y=1.02)
fig.tight_layout()
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
