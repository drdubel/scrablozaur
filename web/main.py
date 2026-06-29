from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from web.engine import get_dawg
from web.routers import board as board_router
from web.routers import game as game_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_dawg()  # load the 44 MB DAWG binary eagerly to avoid first-request latency
    yield


app = FastAPI(title="Scrablozaur", lifespan=lifespan)

app.include_router(game_router.router, prefix="/api")
app.include_router(board_router.router, prefix="/api")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
