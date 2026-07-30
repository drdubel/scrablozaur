'use strict';

const DIFFICULTY_EMOJI = { easy: '🌱', medium: '🎯', hard: '🔥', impossible: '💀', smart: '🧠', sim: '🔮' };
const DIFFICULTY_LABEL = { easy: 'Łatwy', medium: 'Średni', hard: 'Trudny', impossible: 'Niemożliwy', smart: 'Sprytny', sim: 'Wizjoner' };

class GameController {
  constructor(api, board) {
    this._api         = api;
    this._board       = board;
    this._suggestions = [];
    this._activeIndex = -1;
    this._players     = [];

    this._typingStartR = null;
    this._typingStartC = null;
    this._scorePreviewTimer = null;
    this._previewAbortCtrl  = null;

    // Indices (into the human rack string) of tiles selected for exchange.
    this._selectedExchangeIndices = new Set();

    this._lastMode      = 'competitive';
    this._lastHumanName = 'Gracz';
    this._playerConfig  = [
      { name: 'Gracz', is_computer: false },
      { name: 'Komputer', is_computer: true },
    ];

    // "Automatyczny" sandbox sub-mode: 2-4 computer players, each with its
    // own difficulty, no human -- a distinct row shape from manual sandbox's
    // name+radio rows, remembered separately across dialog reopens.
    this._sandboxSubMode  = 'manual';
    this._autoPlayerConfig = [
      { name: 'Gracz 1', difficulty: 'easy' },
      { name: 'Gracz 2', difficulty: 'hard' },
    ];

    // Live sandbox_auto play: move log + autoplay loop state.
    this._autoMoveLog    = [];
    this._autoplayActive = false;
    this._autoplayTimer  = null;

    this._bindElements();
    this._buildSetupRows(2);
    this._bindEvents();
  }

  // ── Element references ────────────────────────────────────────────────────

  _bindElements() {
    this._elScoreboard      = document.getElementById('scoreboard');
    this._panelHuman        = document.getElementById('panel-human');
    this._panelComputer     = document.getElementById('panel-computer');
    this._elHumanTitle      = document.getElementById('panel-human-title');
    this._elComputerTitle   = document.getElementById('panel-computer-title');

    this._selHumanDir       = document.getElementById('human-dir');
    this._elWordDisplay     = document.getElementById('human-word-display');
    this._elScorePreview    = document.getElementById('human-score-preview');
    this._btnPlaceHuman     = document.getElementById('btn-place-human');
    this._btnExchangeHuman  = document.getElementById('btn-exchange-human');
    this._btnSkipHuman      = document.getElementById('btn-skip-human');
    this._btnPassHuman      = document.getElementById('btn-pass-human');
    this._elHumanError      = document.getElementById('human-error');

    this._inComputerLetters = document.getElementById('computer-letters');
    this._btnSuggest        = document.getElementById('btn-suggest');
    this._btnSkipComputer   = document.getElementById('btn-skip-computer');
    this._elSuggestError    = document.getElementById('suggest-error');
    this._elSuggestionList  = document.getElementById('suggestion-list');
    this._elSuggestionSort  = document.getElementById('suggestion-sort');

    this._panelAuto         = document.getElementById('panel-auto');
    this._elAutoCurrent     = document.getElementById('panel-auto-current');
    this._btnAutoNext       = document.getElementById('btn-auto-next');
    this._btnAutoPlay       = document.getElementById('btn-auto-play');
    this._elAutoMoveLog     = document.getElementById('auto-move-log');

    this._btnNewGame        = document.getElementById('btn-new-game');
    this._btnUndo           = document.getElementById('btn-undo');
    this._tplSuggestion     = document.getElementById('tpl-suggestion');

    this._elGameView        = document.getElementById('game-view');
    this._elScanView        = document.getElementById('scan-view');

    this._elTileRackWrap    = document.getElementById('tile-rack-wrap');
    this._elTileRack        = document.getElementById('tile-rack');
    this._elCompMoveInfo    = document.getElementById('computer-move-info');
    this._elTypingInput     = document.getElementById('board-typing-input');

    this._elExchangeHint    = document.getElementById('exchange-hint');

    this._elWordDefPanel    = document.getElementById('word-def-panel');
    this._elWordDefTitle    = document.getElementById('word-def-title');
    this._elWordDefText     = document.getElementById('word-def-text');
    this._btnCloseDef       = document.getElementById('btn-close-def');

    this._btnHints              = document.getElementById('btn-hints');
    this._elHintList            = document.getElementById('hint-list');
    this._elHintSort            = document.getElementById('hint-sort');
    this._elRatingPanel         = document.getElementById('rating-panel');
    this._elRatingArc           = document.getElementById('rating-arc');
    this._elRatingValue         = document.getElementById('rating-value');
    this._elRatingDesc          = document.getElementById('rating-desc');
    this._elRatingHistoryWrap   = document.getElementById('rating-history-wrap');
    this._elRatingHistory       = document.getElementById('rating-history');
    this._ratingHistory         = [];

    this._dialog            = document.getElementById('dialog-setup');
    this._setupMode         = document.getElementById('setup-mode');
    this._setupSandboxCfg   = document.getElementById('setup-sandbox-config');
    this._setupCompCfg      = document.getElementById('setup-competitive-config');
    this._setupCount        = document.getElementById('setup-count');
    this._setupPlayers      = document.getElementById('setup-players');
    this._inPlayerName      = document.getElementById('setup-player-name');
    this._selDifficulty     = document.getElementById('setup-difficulty');
    this._btnStartGame      = document.getElementById('btn-start-game');
    this._elSetupError      = document.getElementById('setup-error');

    this._elSandboxSubDesc  = document.getElementById('sandbox-sub-desc');
    this._elBenchmarkRow    = document.getElementById('setup-benchmark-row');
  }

  // ── Setup dialog ──────────────────────────────────────────────────────────

  _buildSetupRows(count) {
    if (this._sandboxSubMode === 'auto') this._buildAutoSetupRows(count);
    else this._buildManualSetupRows(count);
  }

  _buildManualSetupRows(count) {
    this._setupPlayers.innerHTML = '';
    const defaults = this._playerConfig;
    for (let i = 0; i < count; i++) {
      const def = defaults[i] ?? { name: `Gracz ${i + 1}`, is_computer: false };
      const row = document.createElement('div');
      row.className = 'setup-player-row';
      const num = document.createElement('span');
      num.className = 'player-num'; num.textContent = `${i + 1}.`;
      const inp = document.createElement('input');
      inp.type = 'text'; inp.maxLength = 20; inp.value = def.name;
      inp.placeholder = `Gracz ${i + 1}`;
      const lbl = document.createElement('label');
      lbl.className = 'computer-label';
      const radio = document.createElement('input');
      radio.type = 'radio'; radio.name = 'computer-player'; radio.value = i;
      if (def.is_computer) radio.checked = true;
      lbl.appendChild(radio);
      lbl.appendChild(document.createTextNode('Komputer'));
      row.appendChild(num); row.appendChild(inp); row.appendChild(lbl);
      this._setupPlayers.appendChild(row);
    }
    const anyChecked = this._setupPlayers.querySelector('input[type="radio"]:checked');
    if (!anyChecked) {
      const radios = this._setupPlayers.querySelectorAll('input[type="radio"]');
      radios[radios.length - 1].checked = true;
    }
  }

  /** Automatyczny sandbox: every row is a computer with its own difficulty
   * (no radio -- there's no human to designate). Difficulty is a small
   * button group (matching the competitive mode's diff-cards) rather than
   * a dropdown -- the chosen value lives on the row's own dataset since
   * there's no single underlying <select> to read it back from. */
  _buildAutoSetupRows(count) {
    this._setupPlayers.innerHTML = '';
    const defaults = this._autoPlayerConfig;
    for (let i = 0; i < count; i++) {
      const def = defaults[i] ?? { name: `Gracz ${i + 1}`, difficulty: 'hard' };
      const row = document.createElement('div');
      row.className = 'setup-player-row';
      row.dataset.difficulty = def.difficulty;
      const num = document.createElement('span');
      num.className = 'player-num'; num.textContent = `${i + 1}.`;
      const inp = document.createElement('input');
      inp.type = 'text'; inp.maxLength = 20; inp.value = def.name;
      inp.placeholder = `Gracz ${i + 1}`;
      const diffGroup = document.createElement('div');
      diffGroup.className = 'player-diff-buttons';
      for (const diff of ['easy', 'medium', 'hard', 'impossible', 'smart', 'sim']) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'player-diff-btn' + (diff === def.difficulty ? ' player-diff-btn--active' : '');
        btn.dataset.diff = diff;
        btn.title = DIFFICULTY_LABEL[diff];
        btn.textContent = DIFFICULTY_EMOJI[diff];
        btn.addEventListener('click', () => {
          row.dataset.difficulty = diff;
          diffGroup.querySelectorAll('.player-diff-btn').forEach(b =>
            b.classList.toggle('player-diff-btn--active', b === btn)
          );
        });
        diffGroup.appendChild(btn);
      }
      row.appendChild(num); row.appendChild(inp); row.appendChild(diffGroup);
      this._setupPlayers.appendChild(row);
    }
  }

  _setSandboxSubMode(mode) {
    this._sandboxSubMode = mode;
    this._dialog.querySelectorAll('.sub-toggle-btn').forEach(btn => {
      btn.classList.toggle('sub-toggle-btn--active', btn.dataset.sandboxSub === mode);
    });
    this._elSandboxSubDesc.textContent = mode === 'auto'
      ? 'Komputery grają same, każdy z własnym poziomem trudności — obserwuj partię albo uruchom benchmark.'
      : 'Ty sterujesz każdym graczem — wpisujesz słowa albo litery komputera ręcznie.';
    this._elBenchmarkRow.hidden = mode !== 'auto';
    this._buildSetupRows(parseInt(this._setupCount.value, 10));
  }

  _syncModeUI() {
    const competitive = this._setupMode.value === 'competitive';
    this._setupSandboxCfg.hidden = competitive;
    this._setupCompCfg.hidden    = !competitive;
    // Sync mode card active state
    this._dialog.querySelectorAll('.mode-card').forEach(btn => {
      btn.classList.toggle('mode-card--active', btn.dataset.mode === this._setupMode.value);
    });
  }

  _openSetupDialog() {
    this._stopAutoplay();
    this._setupMode.value = this._lastMode;
    this._inPlayerName.value = this._lastHumanName;
    this._setSandboxSubMode(this._sandboxSubMode);
    this._syncModeUI();
    this._hideError(this._elSetupError);
    this._dialog.showModal();
  }

  _readSetupConfig() {
    const mode = this._setupMode.value;
    if (mode === 'competitive') {
      const name = this._inPlayerName.value.trim() || 'Gracz';
      const difficulty = this._selDifficulty.value;
      return { players: [{ name, is_computer: false }], game_mode: 'competitive', difficulty };
    }
    const rows = [...this._setupPlayers.querySelectorAll('.setup-player-row')];
    if (this._sandboxSubMode === 'auto') {
      return {
        players: rows.map((row, i) => ({
          name: row.querySelector('input[type="text"]').value.trim() || `Gracz ${i + 1}`,
          is_computer: true,
          difficulty: row.dataset.difficulty,
        })),
        game_mode: 'sandbox_auto',
      };
    }
    const radios = [...this._setupPlayers.querySelectorAll('input[type="radio"]')];
    const checked = radios.findIndex(r => r.checked);
    return {
      players: rows.map((row, i) => ({
        name: row.querySelector('input[type="text"]').value.trim() || `Gracz ${i + 1}`,
        is_computer: i === checked,
      })),
      game_mode: 'sandbox',
    };
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  _bindEvents() {
    this._btnNewGame.addEventListener('click', () => this._openSetupDialog());
    this._btnUndo.addEventListener('click', () => this._undoMove());
    this._setupMode.addEventListener('change', () => this._syncModeUI());
    this._setupCount.addEventListener('change', () =>
      this._buildSetupRows(parseInt(this._setupCount.value, 10))
    );

    this._dialog.querySelectorAll('.sub-toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => this._setSandboxSubMode(btn.dataset.sandboxSub));
    });

    this._btnAutoNext.addEventListener('click', () => this._doNextAutoMove());
    this._btnAutoPlay.addEventListener('click', () => this._toggleAutoplay());

    // Mode card clicks ("scan" isn't a real game_mode -- ScanController binds
    // its own click handler on that card to close this dialog and open the
    // scan dialog instead)
    this._dialog.querySelectorAll('.mode-card').forEach(btn => {
      if (btn.dataset.mode === 'scan') return;
      btn.addEventListener('click', () => {
        this._setupMode.value = btn.dataset.mode;
        this._syncModeUI();
      });
    });

    // Difficulty card clicks
    this._dialog.querySelectorAll('.diff-card').forEach(btn => {
      btn.addEventListener('click', () => {
        this._selDifficulty.value = btn.dataset.diff;
        this._dialog.querySelectorAll('.diff-card').forEach(b =>
          b.classList.toggle('diff-card--active', b === btn)
        );
      });
    });
    this._btnStartGame.addEventListener('click', () => this._startGame());

    this._btnPlaceHuman.addEventListener('click', () => this._submitHumanWord());
    this._btnExchangeHuman.addEventListener('click', () => this._exchangeTiles());
    this._btnSkipHuman.addEventListener('click', () => this._skipTurn());
    this._btnPassHuman.addEventListener('click', () => this._passTurn());
    this._btnSkipComputer.addEventListener('click', () => this._skipTurn());

    this._btnCloseDef.addEventListener('click', () => {
      this._elWordDefPanel.style.visibility = 'hidden';
    });

    // Direction change restarts typing at same cell
    this._selHumanDir.addEventListener('change', () => {
      if (this._typingStartR !== null && !this._panelHuman.hidden) {
        this._board.startTyping(
          this._typingStartR, this._typingStartC,
          this._selHumanDir.value === 'true',
        );
      }
    });

    // Board cell click
    this._board.setOnCellClick((r, c) => this._onBoardCellClick(r, c));

    // Mouse drag-and-drop from the rack (native HTML5 DnD) -- the touch
    // equivalent is wired per-tile in _bindRackTileDrag instead, since
    // touch input has no native drag event stream to hook here.
    this._board.setOnTileDrop((r, c, payload) => this._handleTileDrop(r, c, payload));

    // Sync word display + trigger live validation + score preview on typing change
    this._board.setOnTypingUpdate(data => {
      this._elWordDisplay.textContent  = data ? data.word.toUpperCase() : '—';
      this._elScorePreview.textContent = '—';
      this._board.clearWordHighlight();
      this._syncRackWithTyping();
      clearTimeout(this._scorePreviewTimer);
      this._previewAbortCtrl?.abort();
      this._previewAbortCtrl = null;
      if (data && data.word.length >= 2) {
        this._scorePreviewTimer = setTimeout(() => this._fetchScorePreview(data), 100);
      }
    });

    // Global keyboard handler (physical keyboard, page focus anywhere else)
    document.addEventListener('keydown', e => {
      // Don't hijack keystrokes meant for a focused input/textarea (e.g. the
      // scan-board cell editor, the exchange/typing inputs, or any dialog's
      // own text fields) -- this listener is document-wide and would
      // otherwise also drive the main board's typing mode underneath one.
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (this._panelHuman.hidden) return;
      if (this._typingStartR === null) return;
      if (this._handleTypingControlKey(e)) return;
      if (/^[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]$/.test(e.key)) {
        this._board.typeLetter(e.key.toLowerCase());
        e.preventDefault();
      }
    });

    // Phone/tablet keyboard: #board-typing-input is focused when typing mode
    // starts (see _onBoardCellClick) purely to summon the on-screen
    // keyboard. Its own keydown still fires reliably for control keys, but
    // mobile virtual keyboards often don't report usable e.key values for
    // letters, so those are read from the `input` event instead (see
    // below), diffing against a value kept cleared after every keystroke.
    this._elTypingInput.addEventListener('keydown', e => {
      if (this._typingStartR === null) return;
      if (this._handleTypingControlKey(e)) this._elTypingInput.value = '';
    });
    this._elTypingInput.addEventListener('input', () => {
      if (this._typingStartR !== null) {
        for (const ch of this._elTypingInput.value) {
          if (/^[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]$/.test(ch)) this._board.typeLetter(ch.toLowerCase());
        }
      }
      this._elTypingInput.value = '';
    });

    this._btnSuggest.addEventListener('click', () => this._getSuggestions());
    this._elSuggestionSort?.addEventListener('change', () => {
      if (!this._elSuggestionList.hidden) this._getSuggestions();
    });
    this._inComputerLetters.addEventListener('keydown', e => {
      if (e.key === 'Enter') this._getSuggestions();
    });

    this._btnHints.addEventListener('click', () => this._loadHints());
    this._elHintSort?.addEventListener('change', () => {
      if (!this._elHintList.hidden) this._loadHints();
    });
  }

  async init() {
    try {
      const state = await this._api.getState();
      this._applyState(state);
    } catch (err) {
      if (err.status === 401 || err.status === 404) this._openSetupDialog();
      else console.error('Init failed:', err);
    }
  }

  /** Whether any game session (fresh or in-progress) currently exists. */
  hasActiveSession() { return this._lastState != null; }

  /** Public hook so ScanController can bring the user back to the mode
   * picker if they leave scan mode without an active game session yet
   * (e.g. first-ever page load). */
  openSetupDialog() { this._openSetupDialog(); }

  /** Show the normal game (sandbox/competitive) view, hiding scan-view.
   * If no game has ever been started this session, there's nothing to
   * show -- open the mode picker instead rather than an empty board. */
  showGameView() {
    if (!this.hasActiveSession()) {
      this._openSetupDialog();
      return;
    }
    this._elScanView.hidden = true;
    this._elGameView.hidden = false;
    this._btnUndo.hidden = false;
  }

  /** Hide the normal game view so scan-view (a peer view, not an overlay)
   * can take over the page. The game session itself is untouched server-side
   * -- showGameView() brings it straight back, no re-fetch needed. */
  hideGameView() {
    this._elGameView.hidden = true;
    this._btnUndo.hidden = true;
  }

  async _startGame() {
    const config = this._readSetupConfig();
    if (config.game_mode === 'sandbox') {
      const computerCount = config.players.filter(p => p.is_computer).length;
      if (computerCount !== 1) {
        this._showError(this._elSetupError, 'Dokładnie jeden gracz musi być komputerem.');
        return;
      }
    }
    this._setLoading(this._btnStartGame, true);
    try {
      this._lastMode = config.game_mode === 'sandbox_auto' ? 'sandbox' : config.game_mode;
      if (config.game_mode === 'competitive') this._lastHumanName = config.players[0].name;
      else if (config.game_mode === 'sandbox_auto') this._autoPlayerConfig = config.players;
      else this._playerConfig = config.players;
      this._stopAutoplay();
      const state = await this._api.resetGame(config);
      this._dialog.close();
      this._applyState(state);
    } catch (err) {
      this._showError(this._elSetupError, err.detail ?? err.message);
    } finally {
      this._setLoading(this._btnStartGame, false);
    }
  }

  // ── State application ─────────────────────────────────────────────────────

  _applyState(state, opts = {}) {
    this._lastState = state;
    this._elScanView.hidden = true;
    this._elGameView.hidden = false;
    this._btnUndo.hidden = false;
    this._typingStartR = null;
    this._typingStartC = null;
    this._players = state.players;
    this._board.clearHint();
    if (this._elHintList) { this._elHintList.hidden = true; this._elHintList.innerHTML = ''; }
    if (this._btnHints) this._btnHints.textContent = 'Pokaż podpowiedzi';
    if (state.move_number === 0) {
      this._ratingHistory = [];
      if (this._elRatingPanel) this._elRatingPanel.style.visibility = 'hidden';
      if (this._elRatingHistoryWrap) this._elRatingHistoryWrap.hidden = true;
      this._autoMoveLog = [];
    } else if (state.game_mode === 'sandbox_auto') {
      if (opts.logMove) this._logAutoMove(state);
      if (opts.popLog) this._autoMoveLog.pop();
    }

    this._board.render(state.board, state.tile_owners ?? null, state.board_blanks ?? null);
    this._renderScoreboard(state);

    this._suggestions = [];
    this._activeIndex = -1;
    this._elSuggestionList.hidden    = true;
    this._elSuggestionList.innerHTML = '';
    this._hideError(this._elHumanError);
    this._hideError(this._elSuggestError);
    this._btnUndo.disabled = !state.can_undo;

    if (state.game_over) {
      this._panelHuman.hidden    = true;
      this._panelComputer.hidden = true;
      this._panelAuto.hidden     = true;
      this._elTileRackWrap.hidden = true;
      this._stopAutoplay();
      this._showGameOver(state);
      return;
    }

    this._elGameOver?.remove();

    // Hints (/board/hints) and tile exchange (/board/exchange) are both
    // rejected server-side outside COMPETITIVE (no rack there to hint from
    // or exchange against) -- hide rather than leave them clickable into a
    // guaranteed error.
    const isCompetitive = state.game_mode === 'competitive';
    if (this._btnHints) this._btnHints.hidden = !isCompetitive;
    if (this._btnExchangeHuman) this._btnExchangeHuman.hidden = !isCompetitive;
    if (this._elExchangeHint) this._elExchangeHint.hidden = !isCompetitive;

    if (state.game_mode === 'competitive') {
      this._panelAuto.hidden = true;
      this._renderTileRack(state);
      this._renderCompMoveInfo(state);
      this._panelComputer.hidden = true;
      this._panelHuman.hidden    = false;
      const humanPlayer = state.players.find(p => !p.is_computer);
      this._elHumanTitle.textContent = `Twój ruch — ${humanPlayer?.name ?? ''}`;
    } else if (state.game_mode === 'sandbox_auto') {
      this._elTileRackWrap.hidden = true;
      this._elCompMoveInfo.hidden = true;
      this._panelHuman.hidden    = true;
      this._panelComputer.hidden = true;
      this._panelAuto.hidden     = false;
      this._renderAutoPanel(state);
      this._renderAutoMoveLog();
    } else {
      this._panelAuto.hidden = true;
      this._elTileRackWrap.hidden = true;
      this._elCompMoveInfo.hidden = true;
      const current = state.players[state.current_player_idx];
      if (current.letters) this._inComputerLetters.value = current.letters;
      this._transitionToSandbox(current);
    }
  }

  _renderScoreboard(state) {
    this._elScoreboard.innerHTML = '';
    const moveEl = document.createElement('div');
    moveEl.className = 'score-move';
    moveEl.textContent = `Ruch ${state.move_number + 1}`;
    this._elScoreboard.appendChild(moveEl);

    if (state.game_mode === 'competitive' || state.game_mode === 'sandbox_auto') {
      const bagEl = document.createElement('div');
      bagEl.className = 'score-bag';
      bagEl.innerHTML = `Worek<span>${state.tiles_remaining}</span>`;
      this._elScoreboard.appendChild(bagEl);
    }

    const showActive = state.game_mode === 'sandbox' || state.game_mode === 'sandbox_auto';
    for (const [i, p] of state.players.entries()) {
      const block = document.createElement('div');
      block.className = 'score-block' + (showActive && i === state.current_player_idx ? ' active-player' : '');
      const lbl = document.createElement('span');
      lbl.className = 'score-label';
      const dot = document.createElement('span');
      dot.className = `player-dot player-dot-${i}`;
      const diffBadge = state.game_mode === 'sandbox_auto' ? ` ${DIFFICULTY_EMOJI[p.difficulty] ?? ''}` : '';
      lbl.appendChild(dot);
      lbl.appendChild(document.createTextNode(`${p.name}${p.is_computer ? ' 🤖' : ''}${diffBadge}`));
      const val = document.createElement('span');
      val.className = 'score-value'; val.textContent = p.score;
      block.appendChild(lbl); block.appendChild(val);
      this._elScoreboard.appendChild(block);
    }
  }

  _renderTileRack(state) {
    const human   = state.players.find(p => !p.is_computer);
    const letters = human?.letters ?? '';
    this._selectedExchangeIndices.clear();
    // Tile refs kept so _syncRackWithTyping can mark the ones consumed by
    // the word currently being placed as "left the rack" (see below).
    this._rackTiles = [];
    this._elTileRack.innerHTML = '';
    for (let i = 0; i < letters.length; i++) {
      const ch = letters[i];
      const tile = document.createElement('span');
      const isBlank = ch === '?';
      tile.className = 'rack-tile exchangeable' + (isBlank ? ' blank' : '');
      if (isBlank) {
        tile.textContent = '★';
      } else {
        const val = LETTER_VALUES[ch.toLowerCase()] ?? 0;
        tile.innerHTML =
          `<span class="tile-letter">${ch.toUpperCase()}</span>` +
          `<span class="tile-val">${val}</span>`;
      }
      tile.addEventListener('click', () => {
        // A tile already placed on the board (shown as an empty slot) is
        // out of play for exchange until it's taken back off the board.
        if (tile.classList.contains('rack-tile-used')) return;
        if (this._selectedExchangeIndices.has(i)) {
          this._selectedExchangeIndices.delete(i);
          tile.classList.remove('selected');
        } else {
          this._selectedExchangeIndices.add(i);
          tile.classList.add('selected');
        }
      });
      this._bindRackTileDrag(tile, isBlank ? 'BLANK' : ch);
      this._rackTiles.push({ el: tile, letter: ch, isBlank });
      this._elTileRack.appendChild(tile);
    }
    this._elTileRackWrap.hidden = letters.length === 0;
  }

  /** Mark the rack tiles consumed by the word currently being placed so
   * they visibly "leave" the rack (rendered as empty slots), and restore
   * any that a backspace/Escape/direction-change freed back up. Driven off
   * the board's typing state, so it stays in sync whether letters were
   * dragged onto the board or typed. Greedy assignment (a matching real
   * tile first, else a blank) mirrors the server's own tile deduction
   * (_leave_after_word), so what's shown as used is exactly what will be
   * deducted on submit. */
  _syncRackWithTyping() {
    if (!this._rackTiles) return;
    const typed = this._board.getTypedLetters();
    const used = new Set();
    for (const letter of typed) {
      let idx = this._rackTiles.findIndex((t, i) => !used.has(i) && !t.isBlank && t.letter === letter);
      if (idx === -1) idx = this._rackTiles.findIndex((t, i) => !used.has(i) && t.isBlank);
      if (idx !== -1) used.add(idx);
    }
    this._rackTiles.forEach((t, i) => {
      const isUsed = used.has(i);
      t.el.classList.toggle('rack-tile-used', isUsed);
      // A tile that just left the rack can't stay selected for exchange.
      if (isUsed && this._selectedExchangeIndices.delete(i)) t.el.classList.remove('selected');
    });
  }

  // ── Drag-and-drop tile placement (rack → board) ──────────────────────────
  // Two input paths feed the same _handleTileDrop: native HTML5 drag-and-drop
  // for mouse (dragstart on the tile, drop on a board cell -- see
  // board.js's setOnTileDrop), and a hand-rolled touch version below, since
  // touch input never fires HTML5 DnD events on phones/tablets.

  _bindRackTileDrag(tile, payload) {
    tile.draggable = true;
    tile.addEventListener('dragstart', e => {
      // Not during the computer's/other turn, and not a tile that already
      // left the rack onto the board (an empty slot -- take it back off the
      // board first, via Backspace, to redrag it).
      if (this._panelHuman.hidden || tile.classList.contains('rack-tile-used')) { e.preventDefault(); return; }
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', payload);
    });

    // Distinguish a tap (handled by the tile's own 'click' listener, e.g.
    // toggling exchange-selection) from a drag: only commit to dragging
    // once the finger has actually moved past a small threshold, so a
    // plain tap still reaches the click handler undisturbed.
    let touch = null;
    tile.addEventListener('touchstart', e => {
      if (this._panelHuman.hidden || tile.classList.contains('rack-tile-used') || e.touches.length !== 1) return;
      const t = e.touches[0];
      touch = { startX: t.clientX, startY: t.clientY, dragging: false, ghost: null, target: null };
    }, { passive: true });

    tile.addEventListener('touchmove', e => {
      if (!touch) return;
      const t = e.touches[0];
      if (!touch.dragging) {
        if (Math.hypot(t.clientX - touch.startX, t.clientY - touch.startY) < 10) return;
        touch.dragging = true;
        touch.ghost = this._createDragGhost(tile);
      }
      e.preventDefault();
      this._moveDragGhost(touch.ghost, t.clientX, t.clientY);
      const cell = this._board.cellAt(t.clientX, t.clientY);
      if (!cell || !touch.target || cell.row !== touch.target.row || cell.col !== touch.target.col) {
        this._board.clearDragTarget();
      }
      if (cell) this._board.setDragTarget(cell.row, cell.col);
      touch.target = cell;
    }, { passive: false });

    const endTouch = () => {
      if (!touch) return;
      if (touch.dragging) {
        touch.ghost.remove();
        this._board.clearDragTarget();
        if (touch.target) this._handleTileDrop(touch.target.row, touch.target.col, payload);
      }
      touch = null;
    };
    tile.addEventListener('touchend', endTouch);
    tile.addEventListener('touchcancel', endTouch);
  }

  _createDragGhost(tile) {
    const ghost = tile.cloneNode(true);
    ghost.className = tile.className + ' rack-tile-ghost';
    document.body.appendChild(ghost);
    return ghost;
  }

  _moveDragGhost(ghost, x, y) {
    const w = ghost.offsetWidth, h = ghost.offsetHeight;
    ghost.style.transform = `translate(${x - w / 2}px, ${y - h / 2}px)`;
  }

  /** Shared by the mouse (native drop event) and touch (manual hit-test)
   * paths -- payload is either a literal letter or the 'BLANK' sentinel for
   * a blank tile, which has no letter of its own until the player says
   * what it stands for. */
  _handleTileDrop(r, c, payload) {
    if (this._panelHuman.hidden) return;
    let letter = payload;
    if (payload === 'BLANK') {
      const chosen = (prompt('Jaką literę reprezentuje pusty kafelek?', '') ?? '').trim().toLowerCase();
      letter = chosen[0];
      if (!letter || !/^[a-ząćęłńóśźż]$/.test(letter)) return;
    }
    this._placeLetterAt(r, c, letter);
  }

  /** Place one letter at (r, c): continue the word in progress if the drop
   * is still on its line (typeLetter already auto-skips any existing
   * board tiles between the cursor and the next empty slot, same as it
   * does for keyboard typing -- so this only needs to know whether to
   * keep going, not exactly which cell the engine will land on),
   * otherwise (re)start typing there -- same as clicking that cell fresh
   * (see _onBoardCellClick). */
  _placeLetterAt(r, c, letter) {
    const horizontal = this._selHumanDir.value === 'true';
    const continuesLine = this._board.isTyping()
      && (horizontal ? r === this._typingStartR : c === this._typingStartC);
    if (!continuesLine) {
      this._typingStartR = r;
      this._typingStartC = c;
      this._board.startTyping(r, c, horizontal);
    }
    this._board.typeLetter(letter);
    this._elTypingInput.focus({ preventScroll: true });
  }

  _renderCompMoveInfo(state) {
    const m = state.last_computer_move;
    if (!m) { this._elCompMoveInfo.hidden = true; return; }
    this._elCompMoveInfo.textContent = m.passed
      ? '🤖 Komputer spasował (brak możliwych ruchów).'
      : `🤖 Komputer zagrał ${m.word.toUpperCase()} za ${m.score} pkt.`;
    this._elCompMoveInfo.hidden = false;
  }

  _showGameOver(state) {
    this._elGameOver?.remove();
    const el = document.createElement('div');
    el.id = 'game-over-panel';
    el.className = 'game-over-panel';
    const sorted = [...state.players].sort((a, b) => b.score - a.score);
    const winner = state.winner_name;
    const title = winner === 'Remis' ? 'Remis!' : `Wygrał: ${escapeHtml(winner)}!`;
    el.innerHTML = `
      <h2 class="game-over-title">${title}</h2>
      <ul class="game-over-scores">
        ${sorted.map(p => `<li><span>${escapeHtml(p.name)}</span><span>${p.score} pkt</span></li>`).join('')}
      </ul>
      <button id="btn-game-over-new" class="btn btn-primary">Nowa gra</button>
    `;
    this._elScoreboard.parentNode.insertBefore(el, this._elScoreboard.nextSibling);
    this._elGameOver = el;
    el.querySelector('#btn-game-over-new').addEventListener('click', () => this._openSetupDialog());
  }

  _transitionToSandbox(current) {
    this._panelHuman.hidden    = current.is_computer;
    this._panelComputer.hidden = !current.is_computer;
    this._board.clearHighlights();
    if (!current.is_computer) {
      this._elHumanTitle.textContent = `Ruch gracza — ${current.name}`;
    } else {
      this._elComputerTitle.textContent = `Ruch komputera — ${current.name}`;
      this._inComputerLetters.focus();
    }
  }

  // ── Sandbox auto-play (SANDBOX_AUTO: every player is a computer) ─────────

  _renderAutoPanel(state) {
    const current  = state.players[state.current_player_idx];
    const diffEmoji = DIFFICULTY_EMOJI[current.difficulty] ?? '';
    const diffLabel = DIFFICULTY_LABEL[current.difficulty] ?? current.difficulty;
    this._elAutoCurrent.textContent = `Na ruchu: ${current.name} (${diffEmoji} ${diffLabel})`;
  }

  /** Append the move that was just made to the log. Reads the mover off
   * state.current_player_idx *before* advance_turn moved it on -- unless the
   * game just ended, in which case advance_turn never ran and the index
   * still points at whoever made the final move. */
  _logAutoMove(state) {
    const move = state.last_computer_move;
    if (!move) return;
    const n = state.players.length;
    const moverIdx = state.game_over
      ? state.current_player_idx
      : (state.current_player_idx - 1 + n) % n;
    const mover = state.players[moverIdx];
    this._autoMoveLog.unshift({
      playerName: mover.name,
      difficulty: mover.difficulty,
      word: move.word,
      score: move.score,
      passed: move.passed,
    });
  }

  _renderAutoMoveLog() {
    this._elAutoMoveLog.innerHTML = '';
    for (const entry of this._autoMoveLog) {
      const li = document.createElement('li');
      li.className = 'auto-move-log-item';
      const diffEmoji = DIFFICULTY_EMOJI[entry.difficulty] ?? '';
      const playerLabel = `<span class="aml-player">${escapeHtml(entry.playerName)} ${diffEmoji}</span>`;
      li.innerHTML = entry.passed
        ? `${playerLabel}<span class="aml-passed">spasował</span>`
        : `${playerLabel}` +
          `<span class="aml-word">${entry.word.toUpperCase()}</span>` +
          `<span class="aml-score">${entry.score} pkt</span>`;
      this._elAutoMoveLog.appendChild(li);
    }
  }

  async _doNextAutoMove() {
    this._setLoading(this._btnAutoNext, true);
    try {
      const state = await this._api.nextAutoMove();
      this._applyState(state, { logMove: true });
    } catch (err) {
      console.error('Auto move failed:', err);
      this._stopAutoplay();
    } finally {
      this._setLoading(this._btnAutoNext, false);
    }
  }

  _toggleAutoplay() {
    if (this._autoplayActive) this._stopAutoplay();
    else this._startAutoplay();
  }

  _startAutoplay() {
    if (this._autoplayActive) return;
    this._autoplayActive = true;
    this._btnAutoNext.disabled = true;
    this._btnAutoPlay.textContent = '⏸ Zatrzymaj';
    this._autoplayStep();
  }

  async _autoplayStep() {
    if (!this._autoplayActive) return;
    await this._doNextAutoMove();
    if (!this._autoplayActive) return;
    if (this._lastState?.game_over) { this._stopAutoplay(); return; }
    this._autoplayTimer = setTimeout(() => this._autoplayStep(), 550);
  }

  _stopAutoplay() {
    this._autoplayActive = false;
    clearTimeout(this._autoplayTimer);
    this._autoplayTimer = null;
    if (this._btnAutoNext) this._btnAutoNext.disabled = false;
    if (this._btnAutoPlay) this._btnAutoPlay.textContent = '⏩ Autoodtwarzanie';
  }

  // ── Board cell click ──────────────────────────────────────────────────────

  _onBoardCellClick(r, c) {
    // Definition is always available regardless of whose turn it is
    if (this._board._grid[r][c] !== '-') {
      this._showWordDefinition(r, c);
    }

    // Typing mode and direction toggle only during human's turn
    if (this._panelHuman.hidden) return;

    // Toggle direction when clicking the same start cell again
    if (this._typingStartR === r && this._typingStartC === c) {
      this._selHumanDir.value = this._selHumanDir.value === 'true' ? 'false' : 'true';
    } else if (this._board._grid[r][c] === '-') {
      this._elWordDefPanel.style.visibility = 'hidden';
    }

    this._typingStartR = r;
    this._typingStartC = c;
    this._board.startTyping(r, c, this._selHumanDir.value === 'true');
    // Summons the on-screen keyboard on phones/tablets -- see the input's
    // own comment in index.html. No-op/harmless on desktop.
    this._elTypingInput.focus({ preventScroll: true });
  }

  /** Escape/Backspace/Enter/Space/Arrow handling shared between the
   * document-wide physical-keyboard listener and #board-typing-input's own
   * keydown (focused on mobile) -- letters are handled separately by each
   * caller (see _bindEvents), since mobile virtual keyboards need the
   * `input` event instead of keydown for those. Returns whether `e.key` was
   * one of these control keys (regardless of whether typing was active). */
  _handleTypingControlKey(e) {
    const horiz = this._selHumanDir.value === 'true';
    switch (e.key) {
      case 'Escape':
        this._board.clearTyping();
        this._typingStartR = null; this._typingStartC = null;
        e.preventDefault(); return true;
      case 'Backspace':
        this._board.typeBackspace(); e.preventDefault(); return true;
      case 'Enter':
        this._submitHumanWord(); e.preventDefault(); return true;
      case ' ':
        this._selHumanDir.value = horiz ? 'false' : 'true';
        this._board.startTyping(this._typingStartR, this._typingStartC, !horiz);
        e.preventDefault(); return true;
      case 'ArrowRight': {
        const nc = Math.min(14, this._typingStartC + 1);
        this._typingStartC = nc;
        this._board.startTyping(this._typingStartR, nc, horiz);
        e.preventDefault(); return true;
      }
      case 'ArrowLeft': {
        const nc = Math.max(0, this._typingStartC - 1);
        this._typingStartC = nc;
        this._board.startTyping(this._typingStartR, nc, horiz);
        e.preventDefault(); return true;
      }
      case 'ArrowDown': {
        const nr = Math.min(14, this._typingStartR + 1);
        this._typingStartR = nr;
        this._board.startTyping(nr, this._typingStartC, horiz);
        e.preventDefault(); return true;
      }
      case 'ArrowUp': {
        const nr = Math.max(0, this._typingStartR - 1);
        this._typingStartR = nr;
        this._board.startTyping(nr, this._typingStartC, horiz);
        e.preventDefault(); return true;
      }
      default:
        return false;
    }
  }

  // ── Word definition ───────────────────────────────────────────────────────

  _showWordDefinition(r, c) {
    const { horizontal, vertical } = this._board.wordsAt(r, c);
    // Pick the word to look up: prefer the longer one, fall back to either
    const word = (horizontal && vertical)
      ? (horizontal.length >= vertical.length ? horizontal : vertical)
      : (horizontal ?? vertical);
    if (!word) return;

    this._elWordDefTitle.textContent = word.toUpperCase();
    this._elWordDefText.textContent  = 'Szukam definicji…';
    this._elWordDefPanel.style.visibility = 'visible';

    this._api.getDefinition(word)
      .then(data => {
        if (data.found && data.definitions.length > 0) {
          this._elWordDefText.innerHTML = data.definitions
            .map(d => `<p>${d}</p>`)
            .join('');
        } else {
          this._elWordDefText.textContent = 'Brak definicji dla tego słowa.';
        }
      })
      .catch(() => {
        this._elWordDefText.textContent = 'Nie udało się pobrać definicji.';
      });
  }

  // ── Score preview + live validation ──────────────────────────────────────

  async _fetchScorePreview(data) {
    const ctrl = new AbortController();
    this._previewAbortCtrl = ctrl;
    try {
      const res = await this._api.previewScore(data.word, data.row, data.col, data.horizontal, ctrl.signal);
      if (ctrl.signal.aborted) return;
      if (res.score !== null && res.score !== undefined) {
        this._elScorePreview.textContent = `${res.score} pkt`;
        this._board.setWordHighlight('valid');
      } else {
        this._elScorePreview.textContent = '—';
        this._board.setWordHighlight('invalid');
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      this._elScorePreview.textContent = '—';
      this._board.setWordHighlight('invalid');
    }
  }

  // ── Human move submission ─────────────────────────────────────────────────

  async _submitHumanWord() {
    const data = this._board.getWordData();
    if (!data) {
      this._showError(this._elHumanError, 'Kliknij pole startowe na planszy i wpisz słowo.');
      return;
    }
    this._hideError(this._elHumanError);
    this._setLoading(this._btnPlaceHuman, true);
    try {
      const state = await this._api.placeHumanWord(data.word, data.row, data.col, data.horizontal);
      if (state.last_move_rating != null) {
        const humanPlayer = state.players.find(p => !p.is_computer);
        const prevScore = this._players.find(p => !p.is_computer)?.score ?? 0;
        const earnedScore = (humanPlayer?.score ?? 0) - prevScore;
        this._showRating(state.last_move_rating, data.word, earnedScore);
      }
      this._applyState(state);
    } catch (err) {
      this._showError(this._elHumanError, err.detail ?? err.message);
      this._board.shakeTypedCells();
    } finally {
      this._setLoading(this._btnPlaceHuman, false);
    }
  }

  // ── Skip / Pass / Undo ───────────────────────────────────────────────────

  async _skipTurn() {
    this._hideError(this._elHumanError);
    this._hideError(this._elSuggestError);
    try {
      const state = await this._api.skipTurn();
      this._applyState(state);
    } catch (err) { console.error('Skip failed:', err); }
  }

  async _exchangeTiles() {
    this._hideError(this._elHumanError);
    if (this._selectedExchangeIndices.size === 0) {
      this._showError(this._elHumanError, 'Kliknij na stojaku litery, które chcesz wymienić.');
      return;
    }
    const human = this._players.find(p => !p.is_computer);
    const rack = human?.letters ?? '';
    const letters = [...this._selectedExchangeIndices].sort((a, b) => a - b).map(i => rack[i]).join('');
    this._setLoading(this._btnExchangeHuman, true);
    try {
      const state = await this._api.exchangeTiles(letters);
      this._applyState(state);
    } catch (err) {
      this._showError(this._elHumanError, err.detail ?? err.message);
    } finally {
      this._setLoading(this._btnExchangeHuman, false);
    }
  }

  async _passTurn() {
    if (!confirm('Czy na pewno chcesz się poddać? Gra zostanie zakończona.')) return;
    this._hideError(this._elHumanError);
    try {
      const state = await this._api.passTurn();
      this._applyState(state);
    } catch (err) {
      this._showError(this._elHumanError, err.detail ?? err.message);
    }
  }

  async _undoMove() {
    this._stopAutoplay();
    this._btnUndo.disabled = true;
    this._btnUndo.classList.add('loading');
    try {
      const state = await this._api.undoMove();
      this._applyState(state, { popLog: true });
    } catch (err) {
      this._btnUndo.disabled = false;
      console.error('Undo failed:', err);
    } finally {
      this._btnUndo.classList.remove('loading');
    }
  }

  // ── Computer turn (sandbox) ───────────────────────────────────────────────

  async _getSuggestions() {
    const letters = this._inComputerLetters.value.trim().toLowerCase();
    if (!letters) return;
    this._hideError(this._elSuggestError);
    this._elSuggestionList.hidden = true;
    this._setLoading(this._btnSuggest, true);
    try {
      await this._api.setComputerLetters(letters);
      const res = await this._api.getSuggestions(this._elSuggestionSort?.value ?? 'score');
      this._suggestions = res.suggestions;
      this._renderSuggestions();
    } catch (err) {
      this._showError(this._elSuggestError, err.detail ?? err.message);
    } finally {
      this._setLoading(this._btnSuggest, false);
    }
  }

  _renderSuggestions() {
    this._elSuggestionList.innerHTML = '';
    this._activeIndex = -1;
    if (this._suggestions.length === 0) {
      this._showError(this._elSuggestError, 'Brak możliwych ruchów dla podanych liter.');
      return;
    }
    for (const [i, sug] of this._suggestions.entries()) {
      const node = this._tplSuggestion.content.cloneNode(true);
      const li   = node.querySelector('li');
      li.querySelector('.sug-rank').textContent  = `${i + 1}.`;
      li.querySelector('.sug-word').textContent  = sug.word.toUpperCase();
      li.querySelector('.sug-score').textContent = `${sug.score} pkt`;
      // Only worth showing when the ordering is by something other than the
      // score, otherwise it just repeats the column next to it.
      const sugValue = li.querySelector('.sug-value');
      if (sugValue) sugValue.textContent =
        this._rankedValueLabel(sug, this._elSuggestionSort?.value ?? 'score');
      li.querySelector('.sug-pos').textContent   =
        `w${sug.row} k${sug.col} ${sug.horizontal ? '→' : '↓'}`;
      li.querySelector('.btn-preview').addEventListener('click', () => this._previewSuggestion(i));
      li.querySelector('.btn-place').addEventListener('click',   () => this._placeComputerWord(i));
      this._elSuggestionList.appendChild(li);
    }
    this._elSuggestionList.hidden = false;
  }

  _previewSuggestion(idx) {
    if (this._activeIndex >= 0)
      this._elSuggestionList.children[this._activeIndex]?.classList.remove('active');
    this._activeIndex = idx;
    this._elSuggestionList.children[idx]?.classList.add('active');
    this._board.highlightSuggestion(this._suggestions[idx]);
  }

  async _placeComputerWord(idx) {
    const sug = this._suggestions[idx];
    const btn = this._elSuggestionList.children[idx]?.querySelector('.btn-place');
    this._hideError(this._elSuggestError);
    if (btn) this._setLoading(btn, true);
    try {
      const state = await this._api.placeComputerWord(
        sug.word, sug.row, sug.col, sug.horizontal, sug.score,
      );
      this._applyState(state);
    } catch (err) {
      this._showError(this._elSuggestError, err.detail ?? err.message);
      if (btn) this._setLoading(btn, false);
    }
  }

  // ── Move rating ───────────────────────────────────────────────────────────

  _showRating(rating, word, score) {
    if (!this._elRatingPanel) return;
    const arcLen = 173;
    const filled = Math.round((rating / 100) * arcLen);
    const color = rating >= 70 ? '#22c55e' : rating >= 40 ? '#f59e0b' : '#ef4444';
    this._elRatingArc.setAttribute('stroke-dasharray', `${filled} ${arcLen - filled}`);
    this._elRatingArc.setAttribute('stroke', color);
    this._elRatingValue.textContent = rating;
    this._elRatingValue.style.color = color;
    const desc = rating >= 85 ? 'Świetny ruch!' :
                 rating >= 60 ? 'Dobry ruch' :
                 rating >= 35 ? 'Można lepiej' : 'Stracona okazja';
    this._elRatingDesc.textContent = desc;
    this._elRatingPanel.style.visibility = 'visible';

    if (word != null) {
      this._ratingHistory.unshift({ word, score, rating, color });
      this._renderRatingHistory();
    }
  }

  _renderRatingHistory() {
    this._elRatingHistory.innerHTML = '';
    for (const entry of this._ratingHistory) {
      const li = document.createElement('li');
      li.className = 'rating-history-item';
      li.innerHTML =
        `<span class="rh-word">${entry.word.toUpperCase()}</span>` +
        `<span class="rh-pts">${entry.score} pkt</span>` +
        `<div class="rh-bar-wrap">` +
          `<div class="rh-bar-bg"><div class="rh-bar-fill" style="width:${entry.rating}%;background:${entry.color}"></div></div>` +
          `<span class="rh-rating" style="color:${entry.color}">${entry.rating}</span>` +
        `</div>`;
      this._elRatingHistory.appendChild(li);
    }
    this._elRatingHistoryWrap.hidden = this._ratingHistory.length < 2;
  }

  // ── Hints list ────────────────────────────────────────────────────────────

  async _loadHints() {
    if (!this._elHintList.hidden) {
      this._elHintList.hidden = true;
      this._elHintList.innerHTML = '';
      this._board.clearHint();
      this._btnHints.textContent = 'Pokaż podpowiedzi';
      return;
    }
    this._setLoading(this._btnHints, true);
    try {
      const res = await this._api.getHints(this._elHintSort?.value ?? 'score');
      this._hints = res.suggestions;
      this._renderHintList();
      this._btnHints.textContent = 'Ukryj podpowiedzi';
    } catch (err) {
      this._showError(this._elHumanError, err.detail ?? err.message);
    } finally {
      this._setLoading(this._btnHints, false);
    }
  }

  /** Label for the number a suggestion list was ordered by.
   *
   * Blank in score order: repeating the score in the next column tells the
   * player nothing. Under `smart` or `sim` the ordering is not obvious from
   * the visible scores, so the value that produced it is worth showing.
   */
  _rankedValueLabel(sug, mode) {
    if (sug.value == null || mode === 'score') return '';
    return `${sug.value > 0 ? '+' : ''}${sug.value}`;
  }

  _renderHintList() {
    this._elHintList.innerHTML = '';
    if (!this._hints?.length) {
      this._elHintList.innerHTML = '<li style="padding:.5rem .75rem;color:var(--color-muted);font-size:.85rem">Brak możliwych ruchów.</li>';
      this._elHintList.hidden = false;
      return;
    }
    for (const [i, sug] of this._hints.entries()) {
      const li = document.createElement('li');
      li.className = 'hint-item';
      li.innerHTML =
        `<span class="hint-rank">${i + 1}.</span>` +
        `<span class="hint-word">${sug.word.toUpperCase()}</span>` +
        `<span class="hint-score">${sug.score} pkt</span>` +
        `<span class="hint-value">${this._rankedValueLabel(sug, this._elHintSort?.value ?? 'score')}</span>`;
      li.addEventListener('click', () => this._selectHint(i, li));
      this._elHintList.appendChild(li);
    }
    this._elHintList.hidden = false;
  }

  _selectHint(idx, li) {
    this._elHintList.querySelectorAll('.hint-item').forEach(el => el.classList.remove('active'));
    li.classList.add('active');
    this._board.highlightHint(this._hints[idx]);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  _showError(el, msg) { el.textContent = msg; el.hidden = false; }
  _hideError(el)       { el.textContent = '';   el.hidden = true;  }

  _setLoading(btn, loading) {
    btn.disabled = loading;
    btn.classList.toggle('loading', loading);
  }
}
