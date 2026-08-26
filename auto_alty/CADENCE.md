# Cadence

Which model to think with at which time of day, and how often to take an autonomous
turn. Rules are matched against server time; the first matching window wins and the
`default` line is both the off-hours rule and the fallback when a provider is down.
Example (backtick-quoted so it stays inert):

`- 22:00-08:00: lm_studio qwen/qwen3-32b, tick every 15 minutes`

- default: ollama gemma4:e4b, tick every 15 minutes
