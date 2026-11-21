# Serving: thinking budgets, sampling and parallel requests

The defaults come from a study of 9B Qwen3.5 models served on TensorFold on a DGX Spark (stock Qwen3.5-9B q4 unless
stated, greedy for benchmarks): HumanEval+ and MBPP+ with thinking on, BFCL v4 tool calls, and OpenCode agentic
workflows graded by running what the model built.

## Thinking budget

Thinking is on by default (the checkpoint's chat template). The server's `--thinking-budget` caps the reasoning; a
request can send its own `thinking_budget`.

| work | budget | why |
| --- | ---: | --- |
| agentic coding (OpenCode and similar), the serve default | **2048** | run-graded scores were flat from 2048 to 8192 (6/10, 10/20, 5/10); at 2048 no first turn hit the 30-minute cap and turns were 30% shorter than at 4096 (326 vs 463 s); 1024 dropped to 1/10 |
| single-turn coding | 1024 | HumanEval+ 78.7 with thinking off, 89.6 at 1024 |
| tool calls | 512 | BFCL held-out 77.8 with thinking off, 82.4 at 512; refusals hold up |

4096 is a reasonable alternative for agentic work (the most data: 10/20 cells, 3 of 8 browser games ran); 8192 only
made turns longer. Per request:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "q9",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
    "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "thinking_budget": 512, "max_tokens": 1024}'
```

Send `"chat_template_kwargs": {"enable_thinking": false}` to turn thinking off for one request. A budget at or above
`max_tokens` never closes the thinking block; keep it under `max_tokens`, or add `--thinking-reserve N` to give the
answer room after the reasoning.

## Sampling

The checkpoint ships no `generation_config.json`, so an engine without one samples greedily. `scripts/convert.sh`
writes one with Qwen's thinking-mode defaults (temperature 1.0, top_p 0.95); the banner shows
`sampling: temperature 1.0, top_p 0.95`. The study's OpenCode runs used exactly these values; its benchmark rows
sent `temperature: 0`.

## Parallel requests and drafts

Copy drafts are on by default: a chat drafts from its own context (a file it is editing, a passage it quotes) and the
engine verifies every drafted token, so replies are unchanged. They pay most on re-quotes and code edits.

On the DGX Spark the default profile also drafts with the checkpoint's MTP head when the tree holds
`mtp.safetensors` (`scripts/mtp-head.sh`): one chat decodes at 73-75 tok/s against 46-50, and 4 chats at 229 against
168 aggregate. A stream with a copy ahead takes the copy first, so re-quotes and code edits stay at copy-draft speed or
better (1,674 and 1,171 tok/s). `MTP=0 scripts/serve.sh` serves copy drafts only; `PROFILE=batch` never adds MTP,
because many short chats at once are where it stops paying ([CUDA](CUDA.md#mtp-drafting)).

For batch work (benchmarks, many independent coding requests) serve 8 to 16 requests at once (`PROFILE=batch` on the
Spark): replies are identical at every width. In the study (on SenseNova-Loop-9B, the same engine), 16 in flight
brought thinking-on MBPP+ back within 16% of the thinking-off wall time.

## Tool calls

`tool_choice: "required"`, a named function and `response_format` need `libtfgrammar` beside the server
(`XGRAMMAR=DIR scripts/build-engine.sh ...`, from xgrammar v0.2.8); without it those requests get HTTP 400.
`tool_choice: "auto"` works without it.

Don't add a "call the tool" system prompt with thinking off: it raises simple-call scores but turns refusals into
forced, wrong calls (BFCL irrelevance 71 to 39 on SenseNova-Loop-9B in the study). Thinking at 512 gets the calls
without that cost.
