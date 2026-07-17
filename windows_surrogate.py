from __future__ import annotations

import base64
import io
import time

from theseus.model_providers.openrouter_provider import OpenRouterProvider


class WindowsSurrogate:
    """Watches the local desktop: periodically grabs a screenshot and asks a
    vision model to describe what it sees.

    Requires the optional `surrogates` dependency group (Pillow):
    `poetry install --with surrogates`.
    """

    def __init__(
        self,
        model: str = "google/gemma-4-31b-it",
        interval_seconds: float = 5.0,
    ):
        self.model_provider = OpenRouterProvider(model=model)
        self.interval_seconds = interval_seconds

    def _grab_screen(self) -> str:
        """Returns the current screen as a JPEG data URI."""
        from PIL import ImageGrab

        image = ImageGrab.grab().convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def run(self):
        if self.model_provider.is_available():
            print("Model provider is available.")
        else:
            print("Model provider is not available.")
            return

        # Grab the screen every `interval_seconds` and ask the model what it
        # sees, until interrupted.
        while True:
            started = time.monotonic()
            description = self.model_provider.chat(
                "This is a screenshot of a Windows desktop. "
                "Briefly describe what you see.",
                images=[self._grab_screen()],
            )
            print(description)
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, self.interval_seconds - elapsed))


def main():
    WindowsSurrogate().run()


if __name__ == "__main__":
    main()
