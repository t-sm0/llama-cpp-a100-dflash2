#!/usr/bin/env bash
set -euo pipefail

# Long-context retrieval benchmark: hides three keys at 15/50/95% of a
# generated record list and scores their exact recall in a JSON answer.
# Times via the server log lines; requires curl, docker, jq, awk, grep, date.
record_count="${BENCH_RECORDS:?BENCH_RECORDS is required}"
label="${1:-long-context}"
base_url="${BENCH_BASE_URL:-http://127.0.0.1:8080}"
model="${BENCH_MODEL:-model}"
container="${BENCH_CONTAINER:-llama-server}"
max_tokens="${BENCH_TOKENS:-512}"
result_dir="${BENCH_RESULT_DIR:-results}"

for dependency in curl docker jq awk grep date; do
    command -v "$dependency" >/dev/null || {
        printf 'missing dependency: %s\n' "$dependency" >&2
        exit 1
    }
done

mkdir -p "$result_dir"
temp_dir="$(mktemp -d "$result_dir/tmp-XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT
prompt_file="$temp_dir/prompt.txt"
payload_file="$temp_dir/payload.json"
response_file="$result_dir/${label}.json"

awk -v count="$record_count" 'BEGIN {
    alpha = int(count * 0.15)
    bravo = int(count * 0.50)
    charlie = int(count * 0.95)
    for (i=1; i<=count; i++) {
        printf "Record %05d: routine telemetry is nominal; no retrieval key is present.\n", i;
        if (i==alpha) print "KEY ALPHA has exact value SABLE-4821.";
        if (i==bravo) print "KEY BRAVO has exact value ORBIT-7319.";
        if (i==charlie) print "KEY CHARLIE has exact value MINT-2654.";
    }
    print "Return only one compact JSON object with keys alpha, bravo, charlie and their exact values from the records. Do not add prose or markdown.";
}' >"$prompt_file"

jq -n \
    --arg model "$model" \
    --rawfile prompt "$prompt_file" \
    --argjson max_tokens "$max_tokens" \
    '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0.2,top_p:0.95,seed:42,max_tokens:$max_tokens}' \
    >"$payload_file"

curl -fsS "$base_url/health" >/dev/null
start_epoch="$(date +%s)"
curl -fsS "$base_url/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@$payload_file" >"$response_file"

if jq -e '.error' "$response_file" >/dev/null; then
    jq -c '.error' "$response_file" >&2
    exit 1
fi

score=0
grep -q 'SABLE-4821' "$response_file" && score=$((score + 1))
grep -q 'ORBIT-7319' "$response_file" && score=$((score + 1))
grep -q 'MINT-2654' "$response_file" && score=$((score + 1))

logs="$(docker logs --since "$start_epoch" "$container" 2>&1)"
prompt_tps="$(printf '%s\n' "$logs" | sed -n 's/.*prompt eval time =.*,[[:space:]]*\([0-9][0-9.]*\) tokens per second).*/\1/p' | tail -n 1)"
decode_tps="$(printf '%s\n' "$logs" | sed -n 's/.*eval time =.*,[[:space:]]*\([0-9][0-9.]*\) tokens per second).*/\1/p' | tail -n 1)"
acceptance="$(printf '%s\n' "$logs" | sed -n 's/.*draft acceptance = \([0-9][0-9.]*\).*/\1/p' | tail -n 1)"
mean_len="$(printf '%s\n' "$logs" | sed -n 's/.*mean len = *\([0-9][0-9.]*\).*/\1/p' | tail -n 1)"

prompt_tokens="$(jq -r '.usage.prompt_tokens // 0' "$response_file")"
completion_tokens="$(jq -r '.usage.completion_tokens // 0' "$response_file")"

printf 'label\trecords\tprompt_tokens\tcompletion_tokens\tprompt_tps\tdecode_tps\tacceptance\tmean_draft_len\tretrieval_score\n'
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s/3\n' \
    "$label" "$record_count" "$prompt_tokens" "$completion_tokens" \
    "${prompt_tps:-n/a}" "${decode_tps:-n/a}" "${acceptance:-n/a}" "${mean_len:-n/a}" "$score"
