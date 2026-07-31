from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from web.engine import get_pack
from web.game import shutdown_benchmark_pool
from web.routers import benchmark as benchmark_router
from web.routers import board as board_router
from web.routers import game as game_router
from web.routers import scan as scan_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the default language so the common case pays no first-request
    # latency. Other languages load lazily, on the first game that asks.
    get_pack()
    yield
    shutdown_benchmark_pool()  # don't leave ~300 MB/worker benchmark processes behind


app = FastAPI(title="Scrablozaur", lifespan=lifespan)

app.include_router(game_router.router, prefix="/api")
app.include_router(board_router.router, prefix="/api")
app.include_router(scan_router.router, prefix="/api")
app.include_router(benchmark_router.router, prefix="/api")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
