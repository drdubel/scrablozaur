'use strict';

// Reuses BONUS_GRID / BONUS_LABELS / LETTER_VALUES from board.js (loaded
// earlier, same global script scope).

class ScanBoardGrid {
  /** @param {string} containerId */
  constructor(containerId) {
    this._container = document.getElementById(containerId);
    this._cells = [];
    this._data = null; // 15x15 of {letter, confidence, alternatives, flagged, carried_over}
    this._selR = null;
    this._selC = null;
    this._highlighted = [];
    this._onSelect = null;
    this._buildGrid();
  }

  setOnSelect(fn) { this._onSelect = fn; }

  _buildGrid() {
    this._container.innerHTML = '';
    this._cells = [];
    for (let r = 0; r < 15; r++) {
      const row = [];
      for (let c = 0; c < 15; c++) {
        const cell = document.createElement('div');
        cell.className = 'cell ' + BONUS_GRID[r][c];
        cell.setAttribute('role', 'gridcell');
        cell.addEventListener('click', () => this.selectCell(r, c));
        this._container.appendChild(cell);
        row.push(cell);
      }
      this._cells.push(row);
    }
  }

  /** @param {object[][]} cells 15x15 of {letter, confidence, alternatives, flagged, carried_over} */
  load(cells) {
    this._data = cells.map(row => row.map(c => ({ ...c })));
    this._selR = this._selC = null;
    this._highlighted = [];
    for (let r = 0; r < 15; r++) for (let c = 0; c < 15; c++) this._renderCell(r, c);
  }

  _renderCell(r, c) {
    const cell = this._cells[r][c];
    const data = this._data[r][c];
    const selected = r === this._selR && c === this._selC;
    if (data.letter && data.letter !== '-') {
      const val = LETTER_VALUES[data.letter.toLowerCase()] ?? 0;
      cell.className = 'cell placed'
        + (data.flagged ? ' scan-flagged' : '')
        + (data.carried_over ? ' scan-carried' : '')
        + (selected ? ' scan-selected' : '');
      cell.innerHTML =
        `<span class="tile-letter">${data.letter.toUpperCase()}</span>` +
        `<span class="tile-val">${val}</span>`;
    } else {
      const bonus = BONUS_GRID[r][c];
      cell.className = 'cell ' + bonus + (selected ? ' scan-selected' : '');
      cell.textContent = BONUS_LABELS[bonus] ?? '';
    }
  }

  selectCell(r, c) {
    const prevR = this._selR, prevC = this._selC;
    this._selR = r; this._selC = c;
    if (prevR !== null) this._renderCell(prevR, prevC);
    this._renderCell(r, c);
    this._onSelect?.(r, c, this._data[r][c]);
  }

  deselect() {
    if (this._selR === null) return;
    const r = this._selR, c = this._selC;
    this._selR = this._selC = null;
    this._renderCell(r, c);
  }

  /** Set the letter at (r, c); '-' or '' clears the tile. Clears its flag
   * and carried-over marker (a manual edit is the user vouching for it). */
  setLetter(r, c, letter) {
    this._data[r][c] = { ...this._data[r][c], letter: letter || '-', flagged: false, carried_over: false };
    this._renderCell(r, c);
  }

  getGrid() {
    return this._data.map(row => row.map(cell => cell.letter));
  }

  /** Which cells are carried over from a previous confirmed scan -- these
   * are excluded from re-validation server-side, same as at OCR time. */
  getLockedMask() {
    return this._data.map(row => row.map(cell => !!cell.carried_over));
  }

  /** Apply a freshly re-checked flagged/15x15 grid (see
   * ScanController._revalidateBoard) without touching letters or
   * alternatives -- only cells whose flag actually changed are re-rendered. */
  applyFlags(flaggedGrid) {
    for (let r = 0; r < 15; r++) {
      for (let c = 0; c < 15; c++) {
        if (this._data[r][c].flagged !== flaggedGrid[r][c]) {
          this._data[r][c] = { ...this._data[r][c], flagged: flaggedGrid[r][c] };
          this._renderCell(r, c);
        }
      }
    }
  }

  flaggedCount() {
    let n = 0;
    for (const row of this._data) for (const cell of row) if (cell.flagged) n++;
    return n;
  }

  /** Highlight a suggested word's cells -- green for tiles that still need
   * to be placed, blue for cells the board already has a letter in. */
  highlightCells(cells) {
    this.clearHighlight();
    for (const [r, c] of cells) {
      const isNew = !this._data[r][c].letter || this._data[r][c].letter === '-';
      this._cells[r][c].classList.add(isNew ? 'highlight-new' : 'highlight-existing');
      this._highlighted.push([r, c]);
    }
  }

  clearHighlight() {
    for (const [r, c] of this._highlighted) {
      this._cells[r][c].classList.remove('highlight-new', 'highlight-existing');
    }
    this._highlighted = [];
  }
}

class ScanController {
  /** @param {ApiClient} api @param {GameController} gameController */
  constructor(api, gameController) {
    this._api = api;
    this._gameController = gameController;
    this._grid = new ScanBoardGrid('scan-board');
    this._selR = null;
    this._selC = null;
    this._hasScanSession = false;
    this._scanSuggestions = [];
    this._currentFile = null; // kept around so "save to training set" can re-send the same photo
    this._bindElements();
    this._bindEvents();
  }

  _bindElements() {
    this._elScanView       = document.getElementById('scan-view');
    this._btnOpen          = document.getElementById('btn-scan-board');
    this._btnBack          = document.getElementById('btn-scan-back');
    this._stepUpload        = document.getElementById('scan-step-upload');
    this._stepLoading       = document.getElementById('scan-step-loading');
    this._stepReview        = document.getElementById('scan-step-review');
    this._stepAssistant     = document.getElementById('scan-step-assistant');
    this._elBoardArea       = document.getElementById('scan-board-area');
    this._fileInput        = document.getElementById('scan-file-input');
    this._btnChoose        = document.getElementById('btn-scan-choose');
    this._elUploadError      = document.getElementById('scan-upload-error');
    this._elFlagSummary      = document.getElementById('scan-flag-summary');
    this._elEditor          = document.getElementById('scan-cell-editor');
    this._elEditorLabel      = document.getElementById('scan-cell-editor-label');
    this._elEditorInput      = document.getElementById('scan-cell-editor-input');
    this._elEditorAlts       = document.getElementById('scan-cell-editor-alts');
    this._btnClearCell       = document.getElementById('btn-scan-cell-clear');
    this._elReviewError      = document.getElementById('scan-review-error');
    this._chkSaveTraining    = document.getElementById('scan-save-training-checkbox');
    this._btnRescan        = document.getElementById('btn-scan-rescan');
    this._btnConfirm        = document.getElementById('btn-scan-confirm');

    this._elTrainingSaveStatus = document.getElementById('scan-training-save-status');
    this._rackInput         = document.getElementById('scan-rack-input');
    this._btnSuggest        = document.getElementById('btn-scan-suggest');
    this._elSuggestError     = document.getElementById('scan-suggest-error');
    this._elSuggestionList    = document.getElementById('scan-suggestion-list');
    this._tplScanSuggestion   = document.getElementById('tpl-scan-suggestion');
    this._btnNextPhoto       = document.getElementById('btn-scan-next-photo');
    this._btnNewSession      = document.getElementById('btn-scan-new-session');
  }

  _bindEvents() {
    this._btnOpen.addEventListener('click', () => this._openScanView());
    this._btnBack.addEventListener('click', () => this._closeScanView());

    // The initial "Nowa gra" mode picker also offers scanning as a third
    // option (a real user's first action is often "I have a physical board
    // in front of me", not picking sandbox/competitive first) -- clicking it
    // hands off from that dialog to this view.
    document.querySelector('#dialog-setup .mode-card[data-mode="scan"]')
      ?.addEventListener('click', () => {
        document.getElementById('dialog-setup').close();
        this._openScanView();
      });

    this._btnChoose.addEventListener('click', () => this._fileInput.click());
    this._fileInput.addEventListener('change', () => {
      const file = this._fileInput.files[0];
      if (file) this._uploadPhoto(file);
    });

    this._grid.setOnSelect((r, c, data) => this._onCellSelected(r, c, data));

    this._elEditorInput.addEventListener('input', () => this._onEditorInput());
    this._elEditorInput.addEventListener('keydown', e => this._onEditorKeydown(e));
    this._btnClearCell.addEventListener('click', () => this._clearSelectedCell());

    this._btnRescan.addEventListener('click', () => this._showStep('upload'));
    this._btnConfirm.addEventListener('click', () => this._confirmBoard());

    this._btnSuggest.addEventListener('click', () => this._loadSuggestions());
    this._rackInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') this._loadSuggestions();
    });

    this._btnNextPhoto.addEventListener('click', () => this._nextPhoto());
    this._btnNewSession.addEventListener('click', () => this._newSession());
  }

  // ── View lifecycle ────────────────────────────────────────────────────────
  // scan-view is a peer of game-view (see index.html), not a modal on top of
  // it -- switching to it hides game-view (the game session is untouched
  // server-side, showGameView() brings it straight back) instead of trapping
  // the user with no way out. The "← Wróć" button (and the header's own
  // "Nowa gra", for switching to a different mode entirely) are always
  // visible and always work, unlike the old <dialog>-based version's
  // Escape-key-only, easy-to-miss dismissal.

  async _openScanView() {
    this._gameController.hideGameView();
    this._elScanView.hidden = false;
    this._showStep('upload');
    try {
      const state = await this._api.getScanState();
      if (state.has_session) {
        this._grid.load(_cellsFromPlainBoard(state.board));
        this._hasScanSession = true;
        this._showStep('assistant');
      } else {
        this._hasScanSession = false;
      }
    } catch (err) {
      console.error('Failed to load scan session state:', err);
    }
  }

  _closeScanView() {
    this._grid.clearHighlight();
    this._elScanView.hidden = true;
    this._gameController.showGameView();
  }

  _showStep(step) {
    this._stepUpload.hidden    = step !== 'upload';
    this._stepLoading.hidden   = step !== 'loading';
    this._stepReview.hidden    = step !== 'review';
    this._stepAssistant.hidden = step !== 'assistant';
    this._elBoardArea.hidden   = !(step === 'review' || step === 'assistant');
    if (step === 'upload') {
      this._fileInput.value = '';
      this._hideError(this._elUploadError);
      this._elEditor.hidden = true;
      this._selR = this._selC = null;
      this._chkSaveTraining.checked = false;
      this._elTrainingSaveStatus.hidden = true;
    }
  }

  // ── Upload + OCR ──────────────────────────────────────────────────────────

  async _uploadPhoto(file) {
    this._currentFile = file;
    this._hideError(this._elUploadError);
    this._showStep('loading');
    try {
      const res = await this._api.scanBoard(file);
      if (res.error) {
        this._showStep('upload');
        this._showError(this._elUploadError, res.error);
        return;
      }
      this._grid.load(res.cells);
      this._elEditor.hidden = true;
      this._hideError(this._elReviewError);
      this._showStep('review');
      this._updateFlagSummary();
    } catch (err) {
      this._showStep('upload');
      this._showError(this._elUploadError, err.detail ?? err.message);
    }
  }

  _updateFlagSummary() {
    const n = this._grid.flaggedCount();
    if (n > 0) {
      this._elFlagSummary.textContent =
        `⚠ ${n} ${n === 1 ? 'pole może' : 'pól może'} być odczytane błędnie — sprawdź podświetlone pola.`;
      this._elFlagSummary.hidden = false;
    } else {
      this._elFlagSummary.hidden = true;
    }
  }

  // ── Cell editing ──────────────────────────────────────────────────────────

  _onCellSelected(r, c, data) {
    this._selR = r; this._selC = c;
    this._elEditor.hidden = false;
    this._elEditorLabel.textContent = `Wiersz ${r + 1}, kolumna ${c + 1}`;
    this._elEditorInput.value = data.letter && data.letter !== '-' ? data.letter.toUpperCase() : '';
    this._elEditorAlts.innerHTML = '';
    for (const alt of data.alternatives ?? []) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'scan-alt-btn';
      btn.textContent = alt.toUpperCase();
      btn.addEventListener('click', () => this._applyLetter(alt));
      this._elEditorAlts.appendChild(btn);
    }
    this._elEditorInput.focus();
    this._elEditorInput.select();
  }

  _onEditorInput() {
    const ch = this._elEditorInput.value;
    if (ch === '') {
      this._clearSelectedCell();
    } else if (/^[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]$/.test(ch)) {
      this._applyLetter(ch.toLowerCase());
      this._moveSelection(1);
    } else {
      this._elEditorInput.value = '';
    }
  }

  _onEditorKeydown(e) {
    if (e.key === 'Escape') {
      this._elEditor.hidden = true;
      this._grid.deselect();
      this._selR = this._selC = null;
      e.preventDefault();
    } else if (e.key === 'Backspace' && this._elEditorInput.value === '') {
      this._moveSelection(-1);
      e.preventDefault();
    } else if (e.key === 'Enter') {
      this._moveSelection(1);
      e.preventDefault();
    } else if (e.key === 'ArrowRight') {
      this._moveSelection(1); e.preventDefault();
    } else if (e.key === 'ArrowLeft') {
      this._moveSelection(-1); e.preventDefault();
    }
  }

  _applyLetter(letter) {
    if (this._selR === null) return;
    this._grid.setLetter(this._selR, this._selC, letter);
    this._elEditorInput.value = letter.toUpperCase();
    this._revalidateBoard();
  }

  _clearSelectedCell() {
    if (this._selR === null) return;
    this._grid.setLetter(this._selR, this._selC, '-');
    this._elEditorInput.value = '';
    this._revalidateBoard();
  }

  /** Re-run the dictionary flagging check against the board as currently
   * edited, so fixing one cell also clears (or newly raises) flags on
   * every other cell whose word that edit affected -- not just the cell
   * that was actually touched. Best-effort: a failed recheck leaves the
   * existing flags in place rather than blocking the edit. */
  async _revalidateBoard() {
    try {
      const grid = this._grid.getGrid();
      const locked = this._grid.getLockedMask();
      const res = await this._api.recheckScanBoard(grid, locked);
      this._grid.applyFlags(res.flagged);
    } catch (err) {
      console.error('Board recheck failed:', err);
    }
    this._updateFlagSummary();
  }

  _moveSelection(delta) {
    if (this._selR === null) return;
    const c = Math.max(0, Math.min(14, this._selC + delta));
    this._grid.selectCell(this._selR, c);
  }

  // ── Confirm / next photo / new session ──────────────────────────────────

  async _confirmBoard() {
    this._hideError(this._elReviewError);
    this._setLoading(this._btnConfirm, true);
    const wantsTrainingSave = this._chkSaveTraining.checked;
    try {
      const grid = this._grid.getGrid();
      await this._api.confirmScannedBoard(grid);
      this._hasScanSession = true;
      this._grid.clearHighlight();
      this._showStep('assistant');
      if (wantsTrainingSave && this._currentFile) await this._saveTrainingExample(grid);
    } catch (err) {
      this._showError(this._elReviewError, err.detail ?? err.message);
    } finally {
      this._setLoading(this._btnConfirm, false);
    }
  }

  /** Best-effort: a failure here shouldn't undo the confirm that already
   * succeeded, so it's reported inline in the assistant step rather than
   * thrown back up to _confirmBoard's error handling. */
  async _saveTrainingExample(grid) {
    try {
      const res = await this._api.saveTrainingExample(this._currentFile, grid);
      const pct = Math.round(res.match_ratio * 100);
      this._elTrainingSaveStatus.textContent =
        `💾 Zapisano jako przykład treningowy #${res.id} (trudność: ${res.difficulty.toUpperCase()}, ` +
        `surowe rozpoznawanie ${res.matched}/${res.total} pól — ${pct}%).`;
    } catch (err) {
      this._elTrainingSaveStatus.textContent =
        `⚠ Nie udało się zapisać przykładu treningowego: ${err.detail ?? err.message}`;
    }
    this._elTrainingSaveStatus.hidden = false;
  }

  _nextPhoto() {
    this._showStep('upload');
  }

  async _newSession() {
    if (this._hasScanSession && !confirm(
      'Czy na pewno chcesz zacząć nową sesję? Bieżąca zeskanowana plansza zostanie utracona.'
    )) return;
    try {
      await this._api.resetScanSession();
    } catch (err) {
      console.error('Failed to reset scan session:', err);
    }
    this._hasScanSession = false;
    this._rackInput.value = '';
    this._elSuggestionList.hidden = true;
    this._elSuggestionList.innerHTML = '';
    this._hideError(this._elSuggestError);
    this._grid.clearHighlight();
    this._showStep('upload');
  }

  // ── Suggestions ────────────────────────────────────────────────────────────

  async _loadSuggestions() {
    const letters = this._rackInput.value.trim().toLowerCase();
    if (!letters) return;
    this._hideError(this._elSuggestError);
    this._elSuggestionList.hidden = true;
    this._grid.clearHighlight();
    this._setLoading(this._btnSuggest, true);
    try {
      const res = await this._api.suggestForScan(letters);
      this._scanSuggestions = res.suggestions;
      this._renderScanSuggestions();
    } catch (err) {
      this._showError(this._elSuggestError, err.detail ?? err.message);
    } finally {
      this._setLoading(this._btnSuggest, false);
    }
  }

  _renderScanSuggestions() {
    this._elSuggestionList.innerHTML = '';
    if (this._scanSuggestions.length === 0) {
      this._showError(this._elSuggestError, 'Brak możliwych słów dla podanych liter.');
      return;
    }
    for (const [i, sug] of this._scanSuggestions.entries()) {
      const node = this._tplScanSuggestion.content.cloneNode(true);
      const li = node.querySelector('li');
      li.querySelector('.hint-rank').textContent = `${i + 1}.`;
      li.querySelector('.hint-word').textContent = sug.word.toUpperCase();
      li.querySelector('.hint-score').textContent = `${sug.score} pkt`;
      li.querySelector('.hint-pos').textContent = `w${sug.row} k${sug.col} ${sug.horizontal ? '→' : '↓'}`;
      li.addEventListener('click', () => this._selectScanSuggestion(i, li));
      this._elSuggestionList.appendChild(li);
    }
    this._elSuggestionList.hidden = false;
  }

  _selectScanSuggestion(idx, li) {
    this._elSuggestionList.querySelectorAll('.hint-item').forEach(el => el.classList.remove('active'));
    li.classList.add('active');
    this._grid.highlightCells(this._scanSuggestions[idx].cells);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  _showError(el, msg) { el.textContent = msg; el.hidden = false; }
  _hideError(el)       { el.textContent = '';   el.hidden = true;  }

  _setLoading(btn, loading) {
    btn.disabled = loading;
    btn.classList.toggle('loading', loading);
  }
}

/** Wrap a plain 15x15 letter grid (from GET /scan/state) into the richer
 * per-cell shape ScanBoardGrid.load() expects -- a confirmed session's
 * state has nothing to flag or offer alternatives for. */
function _cellsFromPlainBoard(board) {
  return board.map(row => row.map(letter => ({
    letter, confidence: 1, alternatives: [], flagged: false, carried_over: false,
  })));
}
