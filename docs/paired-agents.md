# Observable pair on one host

`agents/pair_alpha.py` and `agents/pair_beta.py` define two independent Auto agents.
They use separate runtime homes and Telegram bot tokens, but the same Telegram
supergroup. Each sees its own stimulus log and a read-only, labeled window of the
partner's log. A peer file can appear after its partner starts; no shared writer or
file watcher is involved. Telegram messages wake the agents and make their exchange
visible to a human in the group.

Create two bots and add both to one **group or supergroup** with your account. In
BotFather, enable Bot-to-Bot Communication Mode for both bots. Give both bots admin
rights in the group, or disable Group Privacy Mode and re-add them. Telegram's
[bot communication documentation](https://core.telegram.org/bots/features#bot-to-bot-communication)
describes these settings. Use the bots' numeric IDs from `getMe`, your numeric user
ID, and the group's numeric chat ID. The group ID is usually negative.

Set the following variables in the shell used for assembly. Keep the tokens private
and set them only in each bot's runtime environment. The generated launcher stores
the numeric IDs and environment-variable *names*, not the token values.

```sh
export PAIR_GROUP_ID=-1001234567890
export PAIR_HUMAN_ID=123456789
export PAIR_ALPHA_BOT_ID=111111111
export PAIR_BETA_BOT_ID=222222222
export TELEGRAM_ALPHA_TOKEN='alpha-token'
export TELEGRAM_BETA_TOKEN='beta-token'

poetry run python -m theseus.assemble agents/pair_alpha.py --output build/pair-alpha
poetry run python -m theseus.assemble agents/pair_beta.py --output build/pair-beta
poetry run python build/pair-alpha/agent.py --check
poetry run python build/pair-beta/agent.py --check
```

Start each launcher in its own process from the same installation:

```sh
poetry run python build/pair-alpha/agent.py
poetry run python build/pair-beta/agent.py
```

The pair's paths assume this sibling `build/pair-alpha` and `build/pair-beta`
layout. Each runtime home holds its own `stimulus_log.jsonl` and
`delivery.sqlite3`. Starting either agent first is fine. The example sends only
to the group, accepts messages only from the named human and partner bot in that
group, and spaces outgoing logical messages by at least five seconds. It polls
every two seconds so queued messages can be sent shortly after that interval.
The example uses an Ollama model by default; set `PAIR_MODEL` before assembly to
change its model name.

For a live check, post a task addressed to both bots in the group. Confirm both
bots respond under distinct names and can reply to each other. Stop and restart
one process, then post another message; its durable inbox should not append the
earlier Telegram update again, and its peer-history window should still read the
partner's current file. Shared group messages can appear in both histories, so
the peer section is labeled and does not copy its records into the local log.
