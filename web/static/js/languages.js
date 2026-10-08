'use strict';

/** The language a game is played in: which dictionary, alphabet, tile
 * distribution and point values apply.
 *
 * Everything comes from `GET /api/game/languages` rather than being written
 * out here. The point values in particular used to be a hand-maintained copy
 * in board.js, which had to be kept in step with the engine by hand; now the
 * server sends the same table `languages/<code>.json` gives the engine.
 *
 * Loaded once at startup (main.js). The UI still works before/without a
 * successful load: `info()` falls back to an entry with no tables, and the
 * renderer simply draws no point values until the fetch lands.
 */
const Languages = {
  default: 'pl',
  loaded: false,
  _list: [],
  _byCode: new Map(),

  async load(api) {
    try {
      const data = await api.getLanguages();
      this.default = data.default;
      this._list = data.languages;
      this._byCode = new Map(data.languages.map(l => [l.code, l]));
      this.loaded = true;
    } catch (err) {
      console.error('Nie udało się pobrać listy języków:', err);
    }
    return this;
  },

  /** Every installed language, in server order. */
  all() { return this._list; },

  /** True when there is nothing to choose between — the picker hides itself. */
  get isSingle() { return this._list.length <= 1; },

  /** @param {string} code */
  info(code) {
    return this._byCode.get(code) ?? {
      code, name: code, flag: '', alphabet: '', blank: '?',
      letter_values: {}, tile_counts: {}, total_tiles: 0,
      max_level: 10, has_ocr: false, ocr_experimental: false, has_leave_net: false,
    };
  },

  name(code) { return this.info(code).name; },
  letterValues(code) { return this.info(code).letter_values; },

  /** Strongest difficulty this language can field. Lower where no leave net
   * has been trained, so the slider simply stops earlier. */
  maxLevel(code) { return this.info(code).max_level; },

  /** Whether the `smart` / `sim` suggestion orderings can be offered. Both
   * consult the learned leave evaluator, which is per-language. */
  hasLeaveNet(code) { return this.info(code).has_leave_net; },

  /** Whether a single character is a playable letter in `code`.
   *
   * Replaces the fixed Polish character classes that used to be scattered
   * through the blank-assignment, rack-entry and scan-editor handlers.
   *
   * @param {string} ch
   * @param {{code?: string, lowerOnly?: boolean}} opts
   */
  isLetter(ch, { code = this.default, lowerOnly = false } = {}) {
    if (typeof ch !== 'string' || ch.length !== 1) return false;
    const alphabet = this.info(code).alphabet;
    if (!alphabet) return false;
    if (alphabet.includes(ch)) return true;
    return lowerOnly ? false : alphabet.includes(ch.toLowerCase());
  },

  /** Build the language picker.
   *
   * A plain `<select>`: unlike difficulty, there is no continuum to convey and
   * no per-option prose worth showing — just a short list of names.
   *
   * @param {{value?: string, onChange?: (code:string)=>void}} opts
   * @returns {{el: HTMLElement, getCode: () => string, setCode: (c:string) => void}}
   */
  createSelect({ value = null, onChange = null } = {}) {
    const wrap = document.createElement('div');
    wrap.className = 'setup-field language-picker';

    const label = document.createElement('label');
    label.className = 'setup-label';
    label.textContent = 'Język gry';
    const select = document.createElement('select');
    select.className = 'setup-select';
    label.htmlFor = select.id = 'setup-language';

    for (const lang of this._list) {
      const opt = document.createElement('option');
      opt.value = lang.code;
      opt.textContent = lang.flag ? `${lang.flag} ${lang.name}` : lang.name;
      select.appendChild(opt);
    }
    select.value = value ?? this.default;

    const note = document.createElement('p');
    note.className = 'setup-note';
    const describe = () => {
      const info = this.info(select.value);
      const parts = [`${info.total_tiles} płytek`];
      // Worth saying plainly: a language with no trained leave net cannot
      // offer the top levels, and a silently shorter slider looks like a bug.
      if (info.max_level < 10) parts.push(`maks. poziom ${info.max_level}`);
      if (!info.has_ocr) parts.push('bez skanowania zdjęć');
      else if (info.ocr_experimental) parts.push('skanowanie eksperymentalne');
      note.textContent = parts.join(' · ');
    };
    describe();

    select.addEventListener('change', () => {
      describe();
      if (onChange) onChange(select.value);
    });

    wrap.append(label, select, note);
    return {
      el: wrap,
      getCode: () => select.value,
      setCode: (c) => { select.value = c; describe(); },
    };
  },
};
