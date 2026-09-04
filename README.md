# llama.cpp A100 DFlash2 speculative decoding

Measured comparison of z-lab's DFlash2 sidecar drafter against the embedded
Qwen MTP head plus N-gram speculation on an NVIDIA A100 (SM80), using a
recent upstream llama.cpp tree. DFlash2 is a block-diffusion drafter: it
predicts a whole block of tokens in one pass, keeps top candidates at every
position, and a selector traces one coherent path. Verification is lossless.

## Test platform

- NVIDIA A100 PCIe 40 GB, SM80, 250 W, one slot
- CUDA 12.6
- llama.cpp commit `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`
- 27B dense hybrid-attention Q6_K GGUF, batch 1
- Context 172032, Q8_0 K/V (main and draft), Flash Attention, CUDA graphs
- Draft: `z-lab/Qwen3.8-27B-DFlash2-GGUF` `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
  (1.09 GB; `dflash.block_size=8`, `n_extract=5`, effective `n_max=7`)
- The SM80 Q6_K kernel patches from
  [llama-cpp-a100-q6-kernels](https://github.com/t-sm0/llama-cpp-a100-q6-kernels)
  were applied and apply cleanly on this commit as well.

Every arm used a fresh container, the same six fixed coding/agent prompts,
384 output tokens, seed 42, and the same warm ordering for the long-context
runs (six-prompt, then 32k, then 16k retrieval).

## Six-prompt decode (cold, tok/s)

| Arm | Speculation | Mean tok/s |
|---|---|---:|
| Recorded previous image (older pinned tree) | FastMTP-patched MTP2 + ngram 24/48/64 | 50.31 / 49.87 |
| A | native embedded MTP2 + ngram 24/48/64 | 51.37 |
| B | DFlash2 Q4_K_M sidecar, n_max 7 | 52.80 |
| B2 | DFlash2 Q8_0 sidecar | 52.75 |
| Live service verify | DFlash2 Q4_K_M sidecar | 53.17 |

DFlash2 wins two coding cases decisively (66.2/68.3 vs 53.0/58.9) and loses
three slightly; acceptance is lower (0.16-0.40 vs 0.46-0.78) but accepted runs
are longer (block drafting; mean draft length 2.1-3.8).

## Long-context retrieval (repeated keys at 15/50/95% of the prompt)

| Workload | Metric | MTP2 + ngram | DFlash2 | Change |
|---|---|---:|---:|---:|
| 32,084-token prompt | prompt tok/s | 1022.50 | 1015.19 | -0.7% |
| 32,084-token prompt | decode tok/s | 52.99 | 71.20 | **+34.4%** |
| 16,164-token prompt | prompt tok/s | 1068.48 | 1060.60 | -0.7% |
| 16,164-token prompt | decode tok/s | 73.94 | 74.21 | +0.4% |
| both | retrieval accuracy | 3/3 | 3/3 | unchanged |

At 32k tokens, embedded MTP2 + N-gram drafting degrades (mean draft 2.83)
while DFlash2 keeps long accepted runs (mean draft 4.94). At 16k, the N-gram
pool reproduces key names from the prompt (mean draft 6.04 for the MTP2 arm),
which levels the field.

Prefill speed is unchanged between arms (within 0.7%). VRAM with the sidecar
was 33.2 GB of 40 GB at full context (about +0.8 GB vs the embedded head).
No CUDA errors or OOMs in any arm.

## Findings

1. DFlash2 won the throughput comparison (+2.8% cold coding mean, +34%
   decode at 32k, identical prefill and retrieval accuracy) and ran in
   production for one afternoon - then a vision regression forced a
   rollback to embedded MTP2 + N-gram (see "Vision limitation").
2. Draft quantization is irrelevant here: Q8_0 scored within noise of Q4_K_M.
   Keep the 1.09 GB Q4_K_M.
3. Upstream master does not register `ngram-mod` beside a sidecar draft
   model; requesting both produced bit-identical results to pure DFlash2.
4. `SPEC_DRAFT_N_MAX` above the trained block size is clamped with a warning:
   with `block_size=8`, n_max 8 becomes 7. Set `n_max = block_size - 1`.
5. Upstream master now covers the full server feature set used here
   (embedded `draft-mtp`, `ngram-mod`, `--cache-type-*-draft`, reasoning
   flags) without any downstream patch, at parity with the patched older

## Vision limitation (production rollback)

On upstream `8b4b3558f`, `llama-server` routes image-embedding ubatches from
`mtmd` chunks through `common_speculative_process`. The `draft-dflash` impl
injects them into the sidecar draft KV cache at target positions. In a long
conversation served with prompt caching, those positions are fragmented by
drafting noise from earlier turns, and the 256-token-wide placement fails
the KV slot search:

```
find_slot: non-consecutive token position <p> after <p> ... 256 new tokens
decode: failed to find a memory slot for batch of size 256
process: llama_decode(ctx_dft) failed rc=1 (n_tokens=256, offset=256)
slot: failed to decode mtmd chunk ... failed to process mtmd chunk
-> HTTP 500 "failed to process mtmd chunk"
```

Every retry of the affected request failed the same way. Short or fresh
image requests succeed; the trigger is a wide image chunk landing over a
fragmented draft cache (observed at a ~23.7k-token cached prefix). Upstream
is aware speculative decoding lacks vision support - the `draft-mtp` and
`draft-eagle3` impls skip embedding batches behind a TODO ("how to make it
work with vision tokens?") - and only `draft-dflash` attempts them. The
upstream MTP mode skips image batches cleanly and survived the same
structure (52,980-token cached prefix, 1024x1024 image injected over stale
response cells) without errors.

Consequence: if your workload sends images to a long-lived slot, keep
speculation on `draft-mtp`/n-gram until this is fixed upstream, or restrict
DFlash2 to text-only endpoints. The A100 Q6_K kernel patches are not
involved: the failure is KV bookkeeping before any math kernel runs.


## Reproduce

Serve the 27B Q6_K target with the DFlash2 sidecar:

```bash
llama-server \
    --model <27B-Q6_K.gguf> \
    --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
    --spec-type draft-dflash \
    --spec-draft-n-max 7 \
    --spec-draft-p-min 0.0 \
    --spec-draft-ngl 999 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
    --ctx-size 172032 --parallel 1 --n-gpu-layers 999 \
    --flash-attn on --jinja
```

Then run `scripts/bench-six-prompts.sh` (cold) and
`scripts/bench-long-context.sh` (retrieval). The scripts time via the server
log lines and require only curl, jq, and docker.

## Caveats

- Single A100 PCIe 40 GB, single-slot, one-at-a-time runs; absolute numbers
  are specific to this card and cooling.
- Six fixed coding prompts do not represent all workloads; the DFlash2
  advantage concentrated in structured/code-heavy cases.
- The 32k/16k arms ran in the same warm container order on both sides, so
  the comparison is fair arm-to-arm, but absolute decode values are warmer
  than a first-request measurement.
- `draft-dspark` (Markov-head variant) was not tested; no DSpark sidecar for
  this target model was published at test time.

## Results

- `results/six-prompt-cold.tsv` - per-case tok/s, acceptance, draft length
- `results/long-context.tsv` - prompt/decode throughput, retrieval score

## License

MIT. See [LICENSE](LICENSE). The DFlash2 drafts are from
`z-lab/Qwen3.8-27B-DFlash2-GGUF` (Apache-2.0); llama.cpp is MIT.
