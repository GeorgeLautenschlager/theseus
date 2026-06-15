import sqlite3
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog
from openai import OpenAI

client = OpenAI(
    base_url="http://100.126.84.49:1234/v1",   # e.g. http://192.168.1.50:1234/v1
    api_key="lm-studio",                      # required by the client, ignored by LM Studio
)

class NaiveContextAssembler:
    def __init__(self, db_path="context.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add_message(self, role, content):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute(
            "INSERT INTO messages (timestamp, role, content) VALUES (?, ?, ?)",
            (timestamp, role, content),
        )
        self.conn.commit()

    def get_context(self):
        rows = self.conn.execute(
            "SELECT timestamp, role, content FROM messages ORDER BY timestamp"
        ).fetchall()
        return "\n".join(
            f"(Current System Time: {timestamp}) {role}: {content}"
            for timestamp, role, content in rows
        )

class Prompter:
    def __init__(self, context_assembler):
        self.context_assembler = context_assembler

    def generate_prompt(self, user_input):
        context = self.context_assembler.get_context()
        prompt = f"{context}\nUser: {user_input}\nAgent:"
        return prompt

    def prompt_model(self, user_input):
        prompt = self.generate_prompt(user_input)

        resp = client.chat.completions.create(
            model="local-model",                      # LM Studio ignores/auto-routes; or use the loaded model id
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        return resp.choices[0].message.content

class AgentApp(App):
    def __init__(self):
        super().__init__()
        self.prompter = Prompter(NaiveContextAssembler())

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True)
        yield Input(placeholder="words go here")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        log = self.query_one(RichLog)
        user_input = event.value
        log.write(f"[bold]you:[/bold] {user_input}")

        agent_reply = self.prompter.prompt_model(user_input)
        log.write(f"[bold]agent:[/bold] {agent_reply}")

        assembler = self.prompter.context_assembler
        assembler.add_message("user", user_input)
        assembler.add_message("agent", agent_reply)

        event.input.clear()

if __name__ == "__main__":
    AgentApp().run()
