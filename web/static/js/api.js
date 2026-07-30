'use strict';

/** Turn an error body into something a human can read.
 *
 * Our own handlers raise HTTPException with a plain string, but FastAPI's
 * request validation returns `detail` as an array of error objects. Those
 * stringify to "[object Object]", which is how a difficulty level the server
 * rejected showed up as an unreadable error instead of naming the field.
 */
function formatDetail(detail) {
  if (detail == null) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map(e => {
        const field = Array.isArray(e.loc) ? e.loc.filter(x => x !== 'body').join('.') : '';
        return field ? `${field}: ${e.msg}` : e.msg;
      })
      .filter(Boolean)
      .join('; ');
  }
  return String(detail.msg ?? JSON.stringify(detail));
}

class ApiError extends Error {
  /** @param {number} status @param {string} detail */
  constructor(status, detail) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

class ApiClient {
  async _request(method, path, body = null, signal = null) {
    const opts = {
      method,
      credentials: 'same-origin',
      headers: body !== null ? { 'Content-Type': 'application/json' } : {},
    };
    if (body !== null) opts.body = JSON.stringify(body);
    if (signal) opts.signal = signal;

    const res = await fetch('/api' + path, opts);
    if (!res.ok) {
      const payload = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(res.status, formatDetail(payload.detail) || res.statusText);
    }
    return res.json();
  }

  /** @param {{ players: {name:string,is_computer:boolean}[], game_mode:string }} body */
  newGame(body)   { return this._request('POST', '/game/new',   body); }
  resetGame(body) { return this._request('POST', '/game/reset', body); }
  getState()        { return this._request('GET',  '/game/state'); }

  /** The custom-difficulty slider's notches + their descriptions. */
  getDifficultyLevels() { return this._request('GET', '/game/difficulty-levels'); }

  placeHumanWord(word, row, col, horizontal) {
    return this._request('POST', '/board/human-move', { word, row, col, horizontal });
  }

  skipTurn() {
    return this._request('POST', '/board/skip');
  }

  exchangeTiles(letters) {
    return this._request('POST', '/board/exchange', { letters });
  }

  passTurn() {
    return this._request('POST', '/board/pass');
  }

  undoMove() {
    return this._request('POST', '/board/undo');
  }

  setComputerLetters(letters) {
    return this._request('POST', '/board/set-letters', { letters });
  }

  getSuggestions(sort = 'score') {
    return this._request('POST', `/board/suggest?sort=${encodeURIComponent(sort)}`);
  }

  placeComputerWord(word, row, col, horizontal, score) {
    return this._request('POST', '/board/computer-move', { word, row, col, horizontal, score });
  }

  previewScore(word, row, col, horizontal, signal = null) {
    return this._request('POST', '/board/preview-score', { word, row, col, horizontal }, signal);
  }

  getDefinition(word) {
    return this._request('GET', `/board/definition/${encodeURIComponent(word)}`);
  }

  getHints(sort = 'score') {
    return this._request('GET', `/board/hints?sort=${encodeURIComponent(sort)}`);
  }

  nextAutoMove() {
    return this._request('POST', '/board/next-move');
  }

  /** @param {{name:string,difficulty:number}[]} players */
  startBenchmark(players, games) {
    return this._request('POST', '/benchmark/start', { players, games });
  }

  getBenchmarkStatus(jobId) {
    return this._request('GET', `/benchmark/status/${jobId}`);
  }

  /** @param {string} path @param {FormData} form */
  async _upload(path, form) {
    const res = await fetch('/api' + path, { method: 'POST', credentials: 'same-origin', body: form });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(res.status, payload.detail ?? res.statusText);
    }
    return res.json();
  }

  scanBoard(file) {
    const form = new FormData();
    form.append('file', file);
    return this._upload('/scan/board', form);
  }

  confirmScannedBoard(board) {
    return this._request('POST', '/scan/confirm', { board });
  }

  getScanState() {
    return this._request('GET', '/scan/state');
  }

  resetScanSession() {
    return this._request('POST', '/scan/reset');
  }

  recheckScanBoard(board, locked) {
    return this._request('POST', '/scan/recheck', { board, locked });
  }

  suggestForScan(letters, sort = 'score') {
    return this._request('POST', '/scan/suggest', { letters, sort });
  }

  saveTrainingExample(file, board) {
    const form = new FormData();
    form.append('file', file);
    form.append('board', JSON.stringify(board));
    return this._upload('/scan/save-training', form);
  }
}
