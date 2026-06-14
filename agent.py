from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog
from openai import OpenAI

client = OpenAI(
    base_url="http://100.126.84.49:1234/v1",   # e.g. http://192.168.1.50:1234/v1
    api_key="lm-studio",                      # required by the client, ignored by LM Studio
)

class NaiveContextAssembler:
    def __init__(self):
        self.context = ""

    def get_context(self):
        return self.context

    def add_to_context(self, new_info):
        self.context += f"\n(Current System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {new_info})"

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
        log.write(f"[bold]you:[/bold] {event.value}")
        log.write(f"[bold]agent:[/bold] {self.prompter.prompt_model(event.value)}")
        self.prompter.context_assembler.add_to_context(event.value)

        event.input.clear()

if __name__ == "__main__":
    AgentApp().run()
