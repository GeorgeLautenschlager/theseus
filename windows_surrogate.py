from __future__ import annotations

import base64
import io
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.openrouter_provider import OpenRouterProvider


class WindowsSurrogate:
    """Watches the local desktop: grabs screenshots at `target_fps` and asks a
    vision model to name the current activity.

    Single-request latency to the model is ~200 ms, so hitting the target rate
    relies on keeping several requests in flight; when the model can't keep up,
    frames are dropped rather than queued so descriptions stay fresh.

    Requires the optional `surrogates` dependency group (Pillow):
    `poetry install --with surrogates`.
    """

    SYSTEM_PROMPT = (
        "You watch screenshots of George's PC. Reply with one short phrase "
        "(under 8 words) naming his current activity. No punctuation, no "
        "extra words."
    )

    def __init__(
        self,
        target_fps: float = 10.0,
        max_in_flight: int = 4,
        image_long_side: int = 448,
    ):
        # self.model_provider = OpenRouterProvider(model="google/gemma-4-31b-it")
        self.model_provider = LmStudioProvider(model="gemma-4-e2b-it-qat-nvfp4")
        self.target_fps = target_fps
        self.max_in_flight = max_in_flight
        self.image_long_side = image_long_side

    def _grab_screen(self) -> str:
        """Returns the current screen, downscaled, as a JPEG data URI."""
        from PIL import Image, ImageGrab

        image = ImageGrab.grab().convert("RGB")
        scale = self.image_long_side / max(image.size)
        if scale < 1.0:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.BILINEAR,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _describe(self, screen: str) -> str:
        return self.model_provider.chat(
            "Current activity:",
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=16,
            temperature=0.0,
            images=[screen],
        )

    def run(self, duration_seconds: float | None = None):
        if self.model_provider.is_available():
            print("Model provider is available.")
        else:
            print("Model provider is not available.")
            return

        period = 1.0 / self.target_fps
        slots = threading.Semaphore(self.max_in_flight)
        print_lock = threading.Lock()
        completions: deque[float] = deque(maxlen=20)
        dropped = 0

        def report(future: Future, frame: int, captured_at: float) -> None:
            slots.release()
            latency = (time.monotonic() - captured_at) * 1000
            error = future.exception()
            if error is not None:
                line = f"[{frame:5d}] {latency:4.0f} ms  ERROR: {error}"
            else:
                now = time.monotonic()
                completions.append(now)
                fps = (
                    (len(completions) - 1) / (now - completions[0])
                    if len(completions) > 1
                    else 0.0
                )
                line = (
                    f"[{frame:5d}] {latency:4.0f} ms  {fps:4.1f}/s  "
                    f"{dropped} dropped  {future.result().strip()}"
                )
            with print_lock:
                print(line)

        started = time.monotonic()
        next_capture = started
        frame = 0
        with ThreadPoolExecutor(max_workers=self.max_in_flight) as pool:
            while duration_seconds is None or time.monotonic() - started < duration_seconds:
                delay = next_capture - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                next_capture += period
                frame += 1
                if not slots.acquire(blocking=False):
                    dropped += 1
                    continue
                screen = self._grab_screen()
                captured_at = time.monotonic()
                future = pool.submit(self._describe, screen)
                future.add_done_callback(
                    lambda f, n=frame, t=captured_at: report(f, n, t)
                )


def main():
    WindowsSurrogate().run()


if __name__ == "__main__":
    main()
