# PSA: llama.cpp draft-dflash + vision (mtmd) breaks on long cached conversations — llama-server, master @ 8b4b3558f

Setup: A100 40GB, 27B Q6_K @ 172k ctx, Q8_0 KV, DFlash2 Q4_K_M sidecar
(z-lab), mmproj loaded, --cache-prompt on.

Symptom: image request from an agent session with ~23k tokens of cached
history → HTTP 500 "failed to process mtmd chunk", every retry identical.
Short conversations with the SAME image: 200, correct answer. So it looks
intermittent. It isn't.

Cause (traced in server logs + common/speculative.cpp):

```
find_slot: non-consecutive token position 23694 after 23694 ... 256 new tokens
decode: failed to find a memory slot for batch of size 256
process: llama_decode(ctx_dft) failed rc=1 (n_tokens=256, offset=256)
```

llama-server routes mtmd image-embedding ubatches through the speculative
callback; draft-dflash injects them into the sidecar draft KV cache at
target positions. Over a long cached conversation those positions hold
stale cells from earlier drafting, and the 256-wide placement fails the
KV slot search. MTP/EAGLE3 skip image batches entirely (there's a TODO in
the code about vision tokens), so draft-mtp/ngram keeps working — that's
also why short DFlash2 image requests work: small chunks fit one ubatch.

Fix: skip mtmd embedding batches in dflash process() and zero-fill the
positional hole they leave (ported from z-lab/llama.cpp-fork#1, their
Metal-validated two commits). Drafted tokens are still verified by the
target, so output stays distribution-exact. Patch + full A100 numbers:

https://github.com/t-sm0/llama-cpp-a100-dflash2

Validation after the patch: 5/5 vision matrix including the exact failure
structure (52,980-token cached prefix, 1024x1024 image over stale cells);
text throughput unchanged (53.15 tok/s coding mean, 3/3 retrieval at
16k/32k).

Bonus finding for the long-context crowd: ngram/MTP drafting decays hard
with context (mean accepted draft 6.0 at 16k → 2.8 at 32k) while DFlash2
holds ~4.9 — decode at 32k was 53 → 71+ tok/s (+34%). If you run long
agent sessions, DFlash2 is the one speculation that doesn't fade.
