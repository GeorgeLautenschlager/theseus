from openai import OpenAI


class LmStudioProvider:
    def __init__(
        self,
        base_url: str = "http://100.126.84.49:1234/v1",
        model: str = "local-model",
        api_key: str = "lm-studio",
    ):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content