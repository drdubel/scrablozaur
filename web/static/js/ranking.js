'use strict';

/** The shared leaderboard (`GET /api/leaderboard`, see web/leaderboard.py):
 * every player's results over this server's own competitive games and the
 * ones the app uploaded. A server keeps none unless its operator turns it on,
 * so the header's button stays hidden until the server says it does. */
class RankingController {
  constructor(api) {
    this._api = api;
    /** Whether this server keeps a leaderboard -- false until it answers. */
    this.enabled = false;

    this._dialog    = document.getElementById('dialog-ranking');
    this._btnOpen   = document.getElementById('btn-ranking');
    this._selLang   = document.getElementById('ranking-language');
    this._elStatus  = document.getElementById('ranking-status');
    this._elPlayers = document.getElementById('ranking-players');
    this._elTopWrap = document.getElementById('ranking-top-wrap');
    this._elTop     = document.getElementById('ranking-top');

    this._btnOpen.addEventListener('click', () => this.open());
    document.getElementById('btn-ranking-close').addEventListener('click', () => this._dialog.close());
    this._selLang.addEventListener('change', () => this._load());
    this._probe();
  }

  async _probe() {
    try {
      this.enabled = (await this._api.getLeaderboard()).enabled;
    } catch {
      this.enabled = false;
    }
    this._btnOpen.hidden = !this.enabled;
  }

  open() {
    if (!this._dialog.open) this._dialog.showModal();
    this._load();
  }

  async _load() {
    this._showStatus('Wczytuję ranking…');
    try {
      this._render(await this._api.getLeaderboard(this._selLang.value || null));
    } catch (err) {
      this._showStatus(err.detail ?? err.message);
    }
  }

  _showStatus(text) {
    this._elStatus.textContent = text;
    this._elStatus.hidden = !text;
  }

  _render(data) {
    // The filter offers the languages played so far, keeping the choice made.
    const chosen = this._selLang.value;
    this._selLang.innerHTML = '<option value="">Wszystkie języki</option>' + data.languages
      .map(code => `<option value="${escapeHtml(code)}">${escapeHtml(Languages.name(code))}</option>`)
      .join('');
    this._selLang.value = data.languages.includes(chosen) ? chosen : '';
    this._selLang.hidden = data.languages.length < 2;

    const empty = data.players.length === 0;
    this._showStatus(empty ? 'Nie ma jeszcze żadnej gry w rankingu.' : '');
    this._elPlayers.hidden = empty;
    this._elTopWrap.hidden = empty;
    this._elPlayers.querySelector('tbody').innerHTML = data.players.map((p, i) => `
      <tr>
        <td>${i + 1}.</td>
        <td class="rk-name">${escapeHtml(p.name)}</td>
        <td>${p.games}</td>
        <td>${p.wins}–${p.losses}${p.ties ? `–${p.ties}` : ''}</td>
        <td>${Math.round(p.win_rate * 100)}%</td>
        <td>${Math.round(p.avg_score)}</td>
        <td>${p.best_score}</td>
        <td>${p.accuracy == null ? '—' : Math.round(p.accuracy)}</td>
      </tr>`).join('');
    this._elTop.innerHTML = data.top_scores.map(s => `
      <li>
        <span class="rk-score">${s.score}</span>
        <span class="rk-name">${escapeHtml(s.name)}</span>
        <span class="rk-detail">${this._opponent(s)} · ${RESULT_NAMES[s.result] ?? s.result} ·
          ${new Date(s.finished).toLocaleDateString('pl-PL')}</span>
      </li>`).join('');
  }

  _opponent(s) {
    if (!s.opponent_computer) return `vs ${escapeHtml(s.opponent)}`;
    const strength = s.opponent_strength != null && s.opponent_strength < 100 ? `, ${s.opponent_strength}%` : '';
    return `vs ${escapeHtml(Difficulty.label(s.opponent_level))}${strength}`;
  }
}

const RESULT_NAMES = { win: 'wygrana', loss: 'przegrana', tie: 'remis' };
