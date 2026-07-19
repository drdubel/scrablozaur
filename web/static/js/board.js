'use strict';

// Bonus classification grid derived from lib.rs BONUS_TABLE (4-fold symmetry).
const BONUS_GRID = (() => {
  const T = [
    [[1,3],[1,1],[1,1],[2,1],[1,1],[1,1],[1,1],[1,3]],
    [[1,1],[1,2],[1,1],[1,1],[1,1],[3,1],[1,1],[1,1]],
    [[1,1],[1,1],[1,2],[1,1],[1,1],[1,1],[2,1],[1,1]],
    [[2,1],[1,1],[1,1],[1,2],[1,1],[1,1],[1,1],[2,1]],
    [[1,1],[1,1],[1,1],[1,1],[1,2],[1,1],[1,1],[1,1]],
    [[1,1],[3,1],[1,1],[1,1],[1,1],[3,1],[1,1],[1,1]],
    [[1,1],[1,1],[2,1],[1,1],[1,1],[1,1],[2,1],[1,1]],
    [[1,3],[1,1],[1,1],[2,1],[1,1],[1,1],[1,1],[1,2]],
  ];
  const grid = [];
  for (let r = 0; r < 15; r++) {
    const row = [];
    for (let c = 0; c < 15; c++) {
      const r2 = Math.min(r, 14 - r), c2 = Math.min(c, 14 - c);
      const [lm, wm] = T[r2][c2];
      let cls = 'empty';
      if (r === 7 && c === 7)  cls = 'center';
      else if (wm === 3)       cls = 'tw';
      else if (wm === 2)       cls = 'dw';
      else if (lm === 3)       cls = 'tl';
      else if (lm === 2)       cls = 'dl';
      row.push(cls);
    }
    grid.push(row);
  }
  return grid;
})();

const BONUS_LABELS = { tw: '3S', dw: '2S', tl: '3L', dl: '2L', center: '★', empty: '' };

/** Escape user-controlled text (player names) before interpolating into
 * innerHTML -- names come straight from a free-text setup field. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Letter point values matching lib.rs calculate_word_points
const LETTER_VALUES = {
  'a':1,'e':1,'i':1,'o':1,'z':1,'w':1,'n':1,'s':1,'r':1,
  'd':2,'y':2,'c':2,'k':2,'l':2,'m':2,'p':2,'t':2,
  'b':3,'g':3,'h':3,'j':3,'ł':3,'u':3,
  'ą':5,'ę':5,'f':5,'ó':5,'ś':5,'ż':5,
  'ć':6,'ń':7,'ź':9,'?':0,
};

class BoardRenderer {
  /** @param {string} containerId */
  constructor(containerId) {
    this._container = document.getElementById(containerId);
    this._cells  = [];
    this._grid   = Array.from({ length: 15 }, () => Array(15).fill('-'));
    this._owners = Array.from({ length: 15 }, () => Array(15).fill(null));
    this._highlightedCoords = [];
    this._hintCoords = [];
    this._hintSuggestions = [];
    this._onCellClickCb  = null;
    this._onTypingUpdate = null;
    // { horizontal, entries:[{r,c,letter,skipped}], cursorR, cursorC }
    this._typing = null;
    this._buildGrid();
  }

  setOnCellClick(fn)     { this._onCellClickCb  = fn; }
  setOnTypingUpdate(fn)  { this._onTypingUpdate  = fn; }
  isTyping()             { return this._typing !== null; }

  _buildGrid() {
    this._container.innerHTML = '';
    this._cells = [];
    for (let r = 0; r < 15; r++) {
      const row = [];
      for (let c = 0; c < 15; c++) {
        const cell = document.createElement('div');
        cell.className = 'cell ' + BONUS_GRID[r][c];
        cell.setAttribute('role', 'gridcell');
        cell.textContent = BONUS_LABELS[BONUS_GRID[r][c]] ?? '';
        cell.addEventListener('click', () => { if (this._onCellClickCb) this._onCellClickCb(r, c); });
        this._container.appendChild(cell);
        row.push(cell);
      }
      this._cells.push(row);
    }
  }

  /**
   * @param {string[][]} grid       15×15 board (letter or '-')
   * @param {(number|null)[][]} owners  15×15 player-index ownership (null = empty) --
   *   tinted by seat index (0-3) so each player's tiles are visually distinct,
   *   regardless of how many players or whether they're human/computer.
   */
  render(grid, owners = null) {
    this._grid   = grid;
    this._owners = owners ?? Array.from({ length: 15 }, () => Array(15).fill(null));
    this.clearHighlights();
    this._typing = null;
    this._onTypingUpdate?.(null);

    for (let r = 0; r < 15; r++) {
      for (let c = 0; c < 15; c++) {
        const letter = grid[r][c];
        const cell   = this._cells[r][c];
        if (letter !== '-') {
          const ownerIdx = this._owners[r][c];
          cell.className = 'cell placed' + (ownerIdx !== null ? ` placed-owner-${ownerIdx}` : '');
          this._setTileLetter(cell, letter);
        } else {
          const bonus = BONUS_GRID[r][c];
          cell.className   = 'cell ' + bonus;
          cell.textContent = BONUS_LABELS[bonus] ?? '';
        }
      }
    }
  }

  /** Render a letter + its point value into a cell. */
  _setTileLetter(cell, letter, extraClass = '') {
    const val = LETTER_VALUES[letter.toLowerCase()] ?? 0;
    cell.innerHTML =
      `<span class="tile-letter">${letter.toUpperCase()}</span>` +
      `<span class="tile-val">${val}</span>`;
    if (extraClass) cell.classList.add(extraClass);
  }

  // ── Typing mode ─────────────────────────────────────────────────────────────

  startTyping(r, c, horizontal) {
    this.clearTyping();
    this._typing = { horizontal, entries: [], cursorR: r, cursorC: c };
    this._renderCursor();
    this._onTypingUpdate?.(this.getWordData());
  }

  typeLetter(letter) {
    const t = this._typing;
    if (!t) return;

    // If the current cell already has this exact letter, treat it as a pass-through
    // (user is typing the full word including existing tiles on the board).
    const existing = this._grid[t.cursorR]?.[t.cursorC];
    if (existing && existing !== '-' && existing === letter) {
      this._cells[t.cursorR][t.cursorC].classList.add('typing-existing');
      // Record as a zero-width skip so backspace can undo it
      t.entries.push({ r: t.cursorR, c: t.cursorC, letter: null, skipped: [] });
      if (t.horizontal) t.cursorC++; else t.cursorR++;
      this._renderCursor();
      this._onTypingUpdate?.(this.getWordData());
      return;
    }

    // Skip over any existing tiles that DON'T match (keep old behaviour)
    const skipped = [];
    while (t.cursorR < 15 && t.cursorC < 15 && this._grid[t.cursorR][t.cursorC] !== '-') {
      const r = t.cursorR, c = t.cursorC;
      this._cells[r][c].classList.add('typing-existing');
      skipped.push([r, c]);
      if (t.horizontal) t.cursorC++; else t.cursorR++;
    }
    if (t.cursorR >= 15 || t.cursorC >= 15) return;

    const r = t.cursorR, c = t.cursorC;
    t.entries.push({ r, c, letter, skipped });
    this._cells[r][c].className = 'cell typing-new';
    this._setTileLetter(this._cells[r][c], letter);

    if (t.horizontal) t.cursorC++; else t.cursorR++;
    this._renderCursor();
    this._onTypingUpdate?.(this.getWordData());
  }

  typeBackspace() {
    const t = this._typing;
    if (!t || t.entries.length === 0) return;

    const last = t.entries.pop();

    if (last.letter === null) {
      // Pass-through entry: just un-highlight the existing cell and move cursor back
      this._cells[last.r][last.c].classList.remove('typing-existing');
      t.cursorR = last.r; t.cursorC = last.c;
    } else {
      const bonus = BONUS_GRID[last.r][last.c];
      this._cells[last.r][last.c].className   = 'cell ' + bonus;
      this._cells[last.r][last.c].textContent = BONUS_LABELS[bonus] ?? '';

      for (const [er, ec] of last.skipped) {
        this._cells[er][ec].classList.remove('typing-existing');
      }

      if (last.skipped.length > 0) {
        [t.cursorR, t.cursorC] = last.skipped[0];
      } else {
        t.cursorR = last.r; t.cursorC = last.c;
      }
    }
    this._renderCursor();
    this._onTypingUpdate?.(this.getWordData());
  }

  clearTyping() {
    if (!this._typing) return;
    this._removeCursor();
    for (const { r, c, letter, skipped } of this._typing.entries) {
      if (letter === null) {
        // Pass-through: only class was added, not content — just remove it
        this._cells[r][c].classList.remove('typing-existing');
      } else {
        const bonus = BONUS_GRID[r][c];
        this._cells[r][c].className   = 'cell ' + bonus;
        this._cells[r][c].textContent = BONUS_LABELS[bonus] ?? '';
        for (const [er, ec] of skipped) {
          this._cells[er][ec].classList.remove('typing-existing');
        }
      }
    }
    this._typing = null;
    this._onTypingUpdate?.(null);
  }

  _removeCursor() {
    this._container.querySelectorAll('.typing-cursor').forEach(el => el.classList.remove('typing-cursor'));
  }

  _renderCursor() {
    this._removeCursor();
    if (!this._typing) return;
    const { cursorR: r, cursorC: c } = this._typing;
    if (r < 15 && c < 15) this._cells[r][c].classList.add('typing-cursor');
  }

  getWordData() {
    const t = this._typing;
    if (!t || t.entries.length === 0) return null;

    const first = t.entries[0];
    let startR = first.r, startC = first.c;
    { let pr = startR - (t.horizontal ? 0 : 1), pc = startC - (t.horizontal ? 1 : 0);
      while (pr >= 0 && pc >= 0 && this._grid[pr][pc] !== '-') {
        startR = pr; startC = pc;
        pr -= t.horizontal ? 0 : 1; pc -= t.horizontal ? 1 : 0;
      }
    }

    const last = t.entries[t.entries.length - 1];
    let endR = last.r, endC = last.c;
    { let fr = endR + (t.horizontal ? 0 : 1), fc = endC + (t.horizontal ? 1 : 0);
      while (fr < 15 && fc < 15 && this._grid[fr][fc] !== '-') {
        endR = fr; endC = fc;
        fr += t.horizontal ? 0 : 1; fc += t.horizontal ? 1 : 0;
      }
    }

    let word = '', r = startR, c = startC;
    while (r <= endR && c <= endC) {
      if (this._grid[r][c] !== '-') {
        word += this._grid[r][c];
      } else {
        const entry = t.entries.find(e => e.r === r && e.c === c && e.letter !== null);
        word += entry ? entry.letter : '?';
      }
      if (t.horizontal) c++; else r++;
    }
    return { word, row: startR, col: startC, horizontal: t.horizontal };
  }

  shakeTypedCells() {
    for (const { r, c } of this._typing?.entries ?? []) {
      const cell = this._cells[r][c];
      cell.classList.remove('typing-shake');
      void cell.offsetWidth;
      cell.classList.add('typing-shake');
      cell.addEventListener('animationend', () => cell.classList.remove('typing-shake'), { once: true });
    }
  }

  /** Highlight typed cells as valid (green) or invalid (red). */
  setWordHighlight(status) {
    for (const { r, c } of this._typing?.entries ?? []) {
      const cell = this._cells[r][c];
      cell.classList.remove('typing-valid', 'typing-invalid');
      cell.classList.add(status === 'valid' ? 'typing-valid' : 'typing-invalid');
    }
  }

  /** Remove valid/invalid highlights from typed cells. */
  clearWordHighlight() {
    for (const { r, c } of this._typing?.entries ?? []) {
      this._cells[r][c].classList.remove('typing-valid', 'typing-invalid');
    }
  }

  /**
   * Find the word(s) passing through (r, c) in both directions.
   * Returns { horizontal: string|null, vertical: string|null }.
   */
  wordsAt(r, c) {
    const scan = (horizontal) => {
      let sr = r, sc = c;
      const dr = horizontal ? 0 : -1, dc = horizontal ? -1 : 0;
      while (sr + dr >= 0 && sc + dc >= 0 && this._grid[sr + dr][sc + dc] !== '-') {
        sr += dr; sc += dc;
      }
      let word = '';
      let cr = sr, cc = sc;
      while (cr < 15 && cc < 15 && this._grid[cr][cc] !== '-') {
        word += this._grid[cr][cc];
        if (horizontal) cc++; else cr++;
      }
      return word.length >= 2 ? word : null;
    };
    return { horizontal: scan(true), vertical: scan(false) };
  }

  // ── Suggestion highlighting ─────────────────────────────────────────────────

  highlightSuggestion(suggestion) {
    this.clearHighlights();
    for (const [r, c] of suggestion.cells) {
      const cell  = this._cells[r][c];
      const isNew = this._grid[r][c] === '-';
      cell.classList.add(isNew ? 'highlight-new' : 'highlight-existing');
      this._highlightedCoords.push([r, c]);
    }
  }

  clearHighlights() {
    for (const [r, c] of this._highlightedCoords) {
      const cell   = this._cells[r][c];
      const letter = this._grid[r][c];
      cell.classList.remove('highlight-new', 'highlight-existing');
      if (letter === '-') {
        const bonus = BONUS_GRID[r][c];
        cell.className   = 'cell ' + bonus;
        cell.textContent = BONUS_LABELS[bonus] ?? '';
      }
    }
    this._highlightedCoords = [];
  }

  // ── Hint word highlight (single selected hint) ─────────────────────────────

  highlightHint(suggestion) {
    this.clearHint();
    for (const [r, c] of suggestion.cells) {
      const isNew = this._grid[r][c] === '-';
      this._cells[r][c].classList.add(isNew ? 'hint-selected-new' : 'hint-selected-existing');
      this._hintCoords.push([r, c]);
    }
  }

  clearHint() {
    for (const [r, c] of this._hintCoords) {
      this._cells[r][c].classList.remove('hint-selected-new', 'hint-selected-existing');
    }
    this._hintCoords = [];
  }

  clearHints() { this.clearHint(); }
}
