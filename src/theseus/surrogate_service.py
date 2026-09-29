"""Surrogate transport owned by an existing managed agent's lifecycle."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from theseus.command_feed import CommandFeed
from theseus.high_water import HighWaterMarks
from theseus.replication_ingress import ReplicationIngress
from theseus.tools.surrogate_voice import SaySurrogate, NotifySurrogate
from theseus.tools.desktop_state import DesktopStateTool


class SurrogateService:
    def __init__(self, core, spec):
        self.ingress = ReplicationIngress(
            core.stimulus_log, HighWaterMarks(core.stimulus_log), on_arrival=core.wake
        )
        self.feed = CommandFeed(core.stimulus_log)

        @asynccontextmanager
        async def lifespan(app):
            self.ingress.start()
            try:
                yield
            finally:
                self.ingress.stop()

        self.app = FastAPI(lifespan=lifespan)
        self.ingress.add_routes(self.app)
        self.feed.add_routes(self.app)

        @self.app.get('/health')
        def health():
            return {'ready': True, 'target': spec.target}

        for tool in (SaySurrogate(spec.target, core.stimulus_log),
                     NotifySurrogate(spec.target, core.stimulus_log),
                     DesktopStateTool(spec.target, core.stimulus_log)):
            core.tools[tool.name] = tool
        core.tools['say_to_surrogate'].description += (
            f' Reply here to messages originating from {spec.target}; '
            'use Telegram for Telegram messages.'
        )
        self.server = uvicorn.Server(uvicorn.Config(
            self.app, host=spec.host, port=spec.port,
            timeout_graceful_shutdown=3, log_level='warning',
        ))

    def run(self):
        self.server.run()

    def stop(self):
        # Safe in the main-thread signal handler; cleanup belongs to run().
        self.server.should_exit = True
