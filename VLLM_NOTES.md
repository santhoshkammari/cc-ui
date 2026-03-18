# vLLM Model Support Notes

## What it does
Runs Claude Code CLI pointed at a local vLLM server instead of Anthropic's API.
This is essentially `claude` CLI with env vars overriding the base URL.

## How it works
- Sets `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY=dummy` in the environment
- Passes `model=VLLM_MODEL` to `ClaudeAgentOptions`
- Uses the same `query()` streaming loop as `_run_claude`
- Restores env vars in `finally` block to avoid leaking

## Config
```python
VLLM_BASE_URL = "http://192.168.170.76:8000"
VLLM_MODEL = "/home/ng6309/datascience/santhosh/models/qwen3.5-9b"
```

## Effectively equivalent to running:
```bash
ANTHROPIC_BASE_URL=http://192.168.170.76:8000 \
ANTHROPIC_API_KEY=dummy \
claude --model /home/ng6309/datascience/santhosh/models/qwen3.5-9b
```

## UI changes needed
- Add `<option value="vllm">vllm (local)</option>` to model selector in index.html
- Route `model == "vllm"` to `_run_vllm()` in both `create_task` and `send_message`

## Key difference from _run_claude
- Only difference: injects vllm env vars + sets model path
- All message handling (AssistantMessage, SystemMessage, ResultMessage, etc.) is identical
