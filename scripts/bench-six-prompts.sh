#!/usr/bin/env bash
set -euo pipefail

# Cold six-prompt decode benchmark for a llama.cpp OpenAI-compatible server.
# Times via the server log lines; requires curl, docker, jq, awk, sed.
base_url="${BENCH_BASE_URL:-http://127.0.0.1:8080}"
container="${BENCH_CONTAINER:-llama-server}"
model="${BENCH_MODEL:-model}"
max_tokens="${BENCH_TOKENS:-384}"
seed="${BENCH_SEED:-42}"
label="${1:-six-prompt-cold}"

for dependency in curl docker jq awk sed; do
    command -v "$dependency" >/dev/null || {
        printf 'missing dependency: %s\n' "$dependency" >&2
        exit 1
    }
done

curl -fsS "$base_url/health" >/dev/null

prompts=(
    'Implement a production-ready JavaScript LRU cache with tests. Explain edge cases briefly, then output the complete code.'
    'Design a safe async job scheduler in TypeScript with bounded concurrency, cancellation, retries, and deterministic tests.'
    'Review this requirement and return strict JSON only: build a mobile inventory app with offline sync, conflict handling, image uploads, and audit logs.'
    'Write a single-file HTML canvas particle simulation optimized for smartphones. Include touch controls and adaptive performance.'
    'Explain how to diagnose a race condition in a Dockerized Node.js service, then provide a minimal reproducible test and fix.'
    'Create a robust Python parser for a streaming line protocol. Handle malformed input, partial frames, backpressure, and property-based tests.'
)

printf 'label\tcase\tcompletion_tokens\tdecode_tps\tdraft_acceptance\tmean_draft_len\n'

case_id=0
for prompt in "${prompts[@]}"; do
    case_id=$((case_id + 1))
    since_epoch="$(date +%s)"
    response_file="$(mktemp /tmp/bench-response-XXXXXX.json)"
    trap 'rm -f "$response_file"' EXIT

    jq -n \
        --arg model "$model" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$max_tokens" \
        --argjson seed "$seed" \
        '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,seed:$seed,stream:false}' |
        curl -fsS "$base_url/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            --data-binary @- >"$response_file"

    completion_tokens="$(jq -r '.usage.completion_tokens // 0' "$response_file")"
    logs="$(docker logs --since "$since_epoch" "$container" 2>&1)"
    decode_tps="$(printf '%s\n' "$logs" | sed -n 's/.*eval time =.*,[[:space:]]*\([0-9][0-9.]*\) tokens per second).*/\1/p' | tail -n 1)"
    acceptance="$(printf '%s\n' "$logs" | sed -n 's/.*draft acceptance = \([0-9][0-9.]*\).*/\1/p' | tail -n 1)"
    mean_len="$(printf '%s\n' "$logs" | sed -n 's/.*mean len = *\([0-9][0-9.]*\).*/\1/p' | tail -n 1)"

    printf '%s\t%d\t%s\t%s\t%s\t%s\n' \
        "$label" "$case_id" "$completion_tokens" \
        "${decode_tps:-n/a}" "${acceptance:-n/a}" "${mean_len:-n/a}"

    rm -f "$response_file"
    trap - EXIT
done
