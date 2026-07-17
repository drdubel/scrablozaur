'use strict';

/** Runs N headless SANDBOX_AUTO games via /benchmark/run and shows the
 * aggregate stats plus a move-by-move replay of the single highest-scoring
 * game. A peer view of #game-view/#scan-view, same show/hide convention as
 * ScanController -- reachable from the setup dialog's "Automatyczny" sandbox
 * config, never needing a live game session to exist first. */
class BenchmarkController {
  constructor(api, gameController) {
    this._api = api;
    this._gameController = gameController;
    this._board = new BoardRenderer('benchmark-board');

    this._moves = [];
    this._currentIdx = -1;
    this._playIntervalId = null;
    this._pollTimer = null;
    this._pollGeneration = 0;
    this._lastPlayers = null;
    this._lastGames = 20;

    this._bindElements();
    this._bindEvents();
  }

  _bindElements() {
    this._dialog       = document.getElementById('dialog-setup');
    this._setupPlayers = document.getElementById('setup-players');
    this._btnRun       = document.getElementById('btn-run-benchmark');
    this._inGames      = document.getElementById('setup-benchmark-games');

    this._elView         = document.getElementById('benchmark-view');
    this._elLoading       = document.getElementById('benchmark-loading');
    this._elLoadingText   = document.getElementById('benchmark-loading-text');
    this._elProgressBar   = document.getElementById('benchmark-progress-bar');
    this._elResults       = document.getElementById('benchmark-results');
    this._elError         = document.getElementById('benchmark-error');
    this._elSummary       = document.getElementById('benchmark-summary');
    this._elStats         = document.getElementById('benchmark-stats');
    this._elBestSection   = document.getElementById('benchmark-best-game');
    this._elBestTitle      = document.getElementById('benchmark-best-title');
    this._elReplaySb      = document.getElementById('benchmark-replay-scoreboard');
    this._elMoveDetail    = document.getElementById('benchmark-move-detail');
    this._elMoveCounter   = document.getElementById('benchmark-move-counter');

    this._btnBack      = document.getElementById('btn-benchmark-back');
    this._btnRerun      = document.getElementById('btn-benchmark-rerun');
    this._btnToSandbox   = document.getElementById('btn-benchmark-to-sandbox');
    this._btnPrev       = document.getElementById('btn-replay-prev');
    this._btnNext       = document.getElementById('btn-replay-next');
    this._btnPlay       = document.getElementById('btn-replay-play');

    this._elWordDefPanel = document.getElementById('benchmark-word-def');
    this._elWordDefTitle = document.getElementById('benchmark-word-def-title');
    this._elWordDefText  = document.getElementById('benchmark-word-def-text');
    this._btnCloseDef    = document.getElementById('btn-benchmark-close-def');
  }

  _bindEvents() {
    this._btnRun.addEventListener('click', () => this._runFromSetup());
    this._btnBack.addEventListener('click', () => this._back());
    this._btnToSandbox.addEventListener('click', () => this._back());
    this._btnRerun.addEventListener('click', () => this._rerun());
    this._btnPrev.addEventListener('click', () => this._stepTo(this._currentIdx - 1));
    this._btnNext.addEventListener('click', () => this._stepTo(this._currentIdx + 1));
    this._btnCloseDef.addEventListener('click', () => this._hideWordDefinition());
    this._board.setOnCellClick((r, c) => this._onBoardCellClick(r, c));
    this._btnPlay.addEventListener('click', () => this._togglePlay());
  }

  // ── Launching a run ──────────────────────────────────────────────────────

  _readPlayersFromSetup() {
    const rows = [...this._setupPlayers.querySelectorAll('.setup-player-row')];
    return rows.map((row, i) => ({
      name: row.querySelector('input[type="text"]').value.trim() || `Gracz ${i + 1}`,
      difficulty: row.querySelector('.player-difficulty').value,
    }));
  }

  async _runFromSetup() {
    const players = this._readPlayersFromSetup();
    const games = Math.max(1, parseInt(this._inGames.value, 10) || 20);
    this._lastPlayers = players;
    this._lastGames = games;
    this._dialog.close();
    await this._run(players, games);
  }

  async _rerun() {
    if (!this._lastPlayers) return;
    await this._run(this._lastPlayers, this._lastGames);
  }

  async _run(players, games) {
    this._stopPlay();
    this._stopPolling();
    this._gameController.hideGameView();
    document.getElementById('scan-view').hidden = true;
    this._elView.hidden = false;
    this._elResults.hidden = true;
    this._hideError();
    this._setProgress(0, games);
    this._elLoading.hidden = false;

    const generation = ++this._pollGeneration;
    try {
      const { job_id } = await this._api.startBenchmark(players, games);
      await this._pollUntilDone(job_id, generation);
    } catch (err) {
      if (generation !== this._pollGeneration) return;
      this._elLoading.hidden = true;
      this._showError(err.detail ?? err.message ?? 'Nie udało się uruchomić benchmarku.');
    }
  }

  /** Poll /benchmark/status every 300ms until done/error. `generation` guards
   * against a stale poll loop (from an abandoned or superseded run) still
   * updating the UI after the user navigated away or started a new run. */
  async _pollUntilDone(jobId, generation) {
    while (generation === this._pollGeneration) {
      const status = await this._api.getBenchmarkStatus(jobId);
      if (generation !== this._pollGeneration) return;
      this._setProgress(status.games_done, status.games_total);

      if (status.status === 'done') {
        this._elLoading.hidden = true;
        this._renderResult(status.result);
        return;
      }
      if (status.status === 'error') {
        this._elLoading.hidden = true;
        this._showError(status.error ?? 'Benchmark zakończył się błędem.');
        return;
      }
      await new Promise(resolve => { this._pollTimer = setTimeout(resolve, 300); });
    }
  }

  _stopPolling() {
    this._pollGeneration++;
    clearTimeout(this._pollTimer);
    this._pollTimer = null;
  }

  _setProgress(done, total) {
    this._elLoadingText.textContent = `Rozgrywam gry… (${done}/${total})`;
    const pct = total ? Math.round((done / total) * 100) : 0;
    this._elProgressBar.style.width = `${pct}%`;
  }

  _back() {
    this._stopPlay();
    this._stopPolling();
    this._elView.hidden = true;
    this._gameController.showGameView();
  }

  _showError(msg) { this._elError.textContent = msg; this._elError.hidden = false; }
  _hideError()     { this._elError.hidden = true; }

  // ── Rendering results ────────────────────────────────────────────────────

  _renderResult(result) {
    this._elResults.hidden = false;
    const seconds = (result.duration_ms / 1000).toFixed(1);
    const bits = [
      `${result.games_played} gier`,
      `${seconds} s`,
      `śr. długość gry: ${result.avg_game_length.toFixed(1)} ruchów`,
    ];
    if (result.longest_word) {
      bits.push(`najdłuższe słowo: ${result.longest_word.toUpperCase()} (${result.longest_word_score} pkt)`);
    }
    if (result.highest_single_move_score) {
      bits.push(`najlepszy pojedynczy ruch: ${result.highest_single_move_score} pkt`);
    }
    this._elSummary.textContent = bits.join(' · ');

    this._renderStats(result.player_stats);
    this._renderBestGame(result.best_game);
  }

  _renderStats(stats) {
    this._elStats.innerHTML = '';
    stats.forEach((s, i) => {
      const winPct = s.games_played ? Math.round((s.wins / s.games_played) * 100) : 0;
      const card = document.createElement('div');
      card.className = 'benchmark-stat-card';
      card.innerHTML = `
        <div class="benchmark-stat-name">
          <span class="player-dot player-dot-${i}"></span>${escapeHtml(s.name)} ${DIFFICULTY_EMOJI[s.difficulty] ?? ''}
        </div>
        <div class="benchmark-stat-diff">${DIFFICULTY_LABEL[s.difficulty] ?? s.difficulty}</div>
        <dl class="benchmark-stat-grid">
          <dt>Gry</dt><dd>${s.games_played}</dd>
          <dt>Wygrane</dt><dd>${s.wins} (${winPct}%)</dd>
          <dt>Remisy</dt><dd>${s.ties}</dd>
          <dt>Śr. wynik</dt><dd>${s.avg_score.toFixed(1)}</dd>
          <dt>Najlepszy</dt><dd>${s.high_score}</dd>
          <dt>Najgorszy</dt><dd>${s.low_score}</dd>
          <dt>Śr. pkt/słowo</dt><dd>${s.avg_word_score.toFixed(1)}</dd>
        </dl>`;
      this._elStats.appendChild(card);
    });
  }

  _renderBestGame(bestGame) {
    this._bestGame = bestGame;
    if (!bestGame || !bestGame.moves.length) {
      this._elBestSection.hidden = true;
      return;
    }
    this._elBestSection.hidden = false;
    this._elBestTitle.textContent = `Najlepsza gra — ${bestGame.winner_name} (${bestGame.winner_score} pkt)`;
    this._moves = bestGame.moves;
    this._stepTo(-1);
  }

  // ── Replay stepping ──────────────────────────────────────────────────────

  _stepTo(idx) {
    if (!this._moves.length) return;
    idx = Math.max(-1, Math.min(this._moves.length - 1, idx));
    this._currentIdx = idx;

    this._hideWordDefinition();

    if (idx === -1) {
      const empty = Array.from({ length: 15 }, () => Array(15).fill('-'));
      this._board.render(empty, null);
      this._elMoveDetail.textContent = 'Początek gry.';
      this._renderReplayScoreboard(this._bestGame.final_scores.map(p => ({ ...p, score: 0, letters: '' })));
    } else {
      const move = this._moves[idx];
      this._board.render(move.board, move.tile_owners);
      const player = this._bestGame.final_scores[move.player_idx];
      this._elMoveDetail.textContent = move.passed
        ? `${player.name} spasował.`
        : `${player.name}: ${move.word.toUpperCase()} — ${move.score} pkt`;
      this._renderReplayScoreboard(
        this._bestGame.final_scores.map((p, i) => ({
          ...p, score: move.scores_after[i], letters: move.letters_after[i],
        })),
      );
    }
    this._elMoveCounter.textContent = `Ruch ${idx + 1}/${this._moves.length}`;
    this._btnPrev.disabled = idx <= -1;
    this._btnNext.disabled = idx >= this._moves.length - 1;
  }

  /** Shows each computer's current rack alongside its score -- useful during
   * best-game replay since, unlike a live game, there's no single "your"
   * rack to hide the rest of; every player's letters are fair to show. */
  _renderReplayScoreboard(players) {
    this._elReplaySb.innerHTML = '';
    players.forEach((p, i) => {
      const rack = [...(p.letters ?? '')].map(ch => (ch === '?' ? '★' : ch.toUpperCase())).join(' ');
      const block = document.createElement('div');
      block.className = 'score-block';
      block.innerHTML =
        `<div class="score-block-top">` +
          `<span class="score-label"><span class="player-dot player-dot-${i}"></span>` +
            `${escapeHtml(p.name)} ${DIFFICULTY_EMOJI[p.difficulty] ?? ''}</span>` +
          `<span class="score-value">${p.score}</span>` +
        `</div>` +
        (rack ? `<span class="score-block-letters">${rack}</span>` : '');
      this._elReplaySb.appendChild(block);
    });
  }

  // ── Word definition lookup (same feature as the live board) ──────────────

  _onBoardCellClick(r, c) {
    if (this._board._grid[r]?.[c] !== '-') this._showWordDefinition(r, c);
  }

  _showWordDefinition(r, c) {
    const { horizontal, vertical } = this._board.wordsAt(r, c);
    const word = (horizontal && vertical)
      ? (horizontal.length >= vertical.length ? horizontal : vertical)
      : (horizontal ?? vertical);
    if (!word) return;

    this._elWordDefTitle.textContent = word.toUpperCase();
    this._elWordDefText.textContent  = 'Szukam definicji…';
    this._elWordDefPanel.style.visibility = 'visible';

    this._api.getDefinition(word)
      .then(data => {
        this._elWordDefText.innerHTML = (data.found && data.definitions.length > 0)
          ? data.definitions.map(d => `<p>${d}</p>`).join('')
          : 'Brak definicji dla tego słowa.';
      })
      .catch(() => { this._elWordDefText.textContent = 'Nie udało się pobrać definicji.'; });
  }

  _hideWordDefinition() {
    this._elWordDefPanel.style.visibility = 'hidden';
  }

  _togglePlay() {
    if (this._playIntervalId) this._stopPlay();
    else this._startPlay();
  }

  _startPlay() {
    if (this._currentIdx >= this._moves.length - 1) this._currentIdx = -1;
    this._btnPlay.textContent = '⏸ Pauza';
    this._playIntervalId = setInterval(() => {
      if (this._currentIdx >= this._moves.length - 1) { this._stopPlay(); return; }
      this._stepTo(this._currentIdx + 1);
    }, 500);
  }

  _stopPlay() {
    if (this._playIntervalId) clearInterval(this._playIntervalId);
    this._playIntervalId = null;
    if (this._btnPlay) this._btnPlay.textContent = '⏩ Odtwórz';
  }
}
