from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from web.engine import Dawg, get_dawg
from web.game import BenchmarkResult, Difficulty, run_benchmark
from web.models import (BenchmarkBestGame, BenchmarkJobStartResponse, BenchmarkJobStatusResponse,
                        BenchmarkMoveRecord, BenchmarkPlayerStats, BenchmarkRequest,
                        BenchmarkResultResponse, PlayerState)

router = APIRouter(prefix="/benchmark")

# In-memory job store -- benchmarks run in a background thread (not
# request-scoped) so the client can poll progress instead of blocking on one
# long request. Bounded to the most recent jobs since this is a single-process
# dev-scale tool, not a multi-tenant service.
_MAX_JOBS = 20
_jobs: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_jobs_lock = threading.Lock()


def _run_job(job_id: str, player_specs: list[tuple[str, Difficulty]], games: int, dawg: Dawg) -> None:
    def on_game_done(done: int) -> None:
        with _jobs_lock:
            _jobs[job_id]["games_done"] = done

    try:
        result = run_benchmark(player_specs, games, dawg, on_game_done=on_game_done)
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


@router.post("/start", response_model=BenchmarkJobStartResponse)
async def start_benchmark(
    body: BenchmarkRequest, dawg: Dawg = Depends(get_dawg)
) -> BenchmarkJobStartResponse:
    player_specs = [(p.name, Difficulty(p.difficulty)) for p in body.players]
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "games_done": 0,
            "games_total": body.games,
            "result": None,
            "error": None,
        }
        while len(_jobs) > _MAX_JOBS:
            _jobs.popitem(last=False)

    threading.Thread(
        target=_run_job, args=(job_id, player_specs, body.games, dawg), daemon=True
    ).start()
    return BenchmarkJobStartResponse(job_id=job_id)


def _to_response(result: BenchmarkResult) -> BenchmarkResultResponse:
    best_game = None
    if result.best_game:
        bg = result.best_game
        best_game = BenchmarkBestGame(
            winner_name=bg.winner_name,
            winner_score=bg.winner_score,
            final_scores=[
                PlayerState(
                    name=p.name,
                    is_computer=p.is_computer,
                    score=p.score,
                    letters=p.letters,
                    difficulty=p.difficulty.value,
                )
                for p in bg.final_players
            ],
            moves=[
                BenchmarkMoveRecord(
                    player_idx=m.player_idx,
                    word=m.word,
                    score=m.score,
                    row=m.row,
                    col=m.col,
                    horizontal=m.horizontal,
                    passed=m.passed,
                    board=m.board,
                    scores_after=m.scores_after,
                    letters_after=m.letters_after,
                    tile_owners=m.tile_owners,
                )
                for m in bg.moves
            ],
        )

    return BenchmarkResultResponse(
        games_played=result.games_played,
        duration_ms=result.duration_ms,
        player_stats=[
            BenchmarkPlayerStats(
                name=s.name,
                difficulty=s.difficulty,
                games_played=s.games_played,
                wins=s.wins,
                ties=s.ties,
                avg_score=s.avg_score,
                high_score=s.high_score,
                low_score=s.low_score,
                words_played=s.words_played,
                avg_word_score=s.avg_word_score,
            )
            for s in result.player_stats
        ],
        best_game=best_game,
        avg_game_length=result.avg_game_length,
        longest_word=result.longest_word,
        longest_word_score=result.longest_word_score,
        highest_single_move_score=result.highest_single_move_score,
    )


@router.get("/status/{job_id}", response_model=BenchmarkJobStatusResponse)
async def benchmark_status(job_id: str) -> BenchmarkJobStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Nie znaleziono benchmarku.")
        status, games_done, games_total = job["status"], job["games_done"], job["games_total"]
        result, error = job["result"], job["error"]

    return BenchmarkJobStatusResponse(
        status=status,
        games_done=games_done,
        games_total=games_total,
        result=_to_response(result) if result is not None else None,
        error=error,
    )
