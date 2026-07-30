'use strict';

/** Custom difficulty: one slider from `min` to `max` instead of named tiers.
 *
 * The level table (name, what the bot does, what to expect) comes from
 * `GET /api/game/difficulty-levels` rather than being written out here,
 * because the mechanical half of each description is generated from the very
 * rank windows the bot picks moves with -- a copy in JS would silently drift
 * the first time a window changed.
 *
 * Loaded once at startup (main.js). Everything still works before/without a
 * successful load: `info()` falls back to a plain "Poziom N".
 */
const Difficulty = {
  min: 1,
  max: 10,
  default: 5,
  loaded: false,
  _byLevel: new Map(),

  async load(api) {
    try {
      const data = await api.getDifficultyLevels();
      this.min = data.min_level;
      this.max = data.max_level;
      this.default = data.default_level;
      this._byLevel = new Map(data.levels.map(l => [l.level, l]));
      this.loaded = true;
    } catch (err) {
      console.error('Nie udało się pobrać opisów poziomów trudności:', err);
    }
    return this;
  },

  /** @param {number|string} level */
  info(level) {
    const n = Number(level);
    return this._byLevel.get(n) ?? {
      level: n, name: `Poziom ${n}`, emoji: '🎚️',
      summary: '', expect: '', engine: 'ranked', slow: false,
    };
  },

  emoji(level) { return this.info(level).emoji; },
  name(level)  { return this.info(level).name; },

  /** "🎯 Klubowy (5/10)" — for scoreboards, move logs and benchmark cards,
   * where the level number matters as much as the name. */
  label(level) { return `${this.emoji(level)} ${this.name(level)} (${Number(level)}/${this.max})`; },

  clamp(level) {
    const n = Number(level);
    if (!Number.isFinite(n)) return this.default;
    return Math.max(this.min, Math.min(this.max, Math.round(n)));
  },

  /** Build a difficulty slider.
   *
   * `variant: 'full'` (the competitive setup) shows the feedback panel: the
   * level's name, what it does mechanically, and what the player should
   * expect from it -- the point of the whole change is that a number on its
   * own tells you nothing.
   *
   * `variant: 'compact'` (one row per computer in automatic sandbox) has no
   * room for prose, so the same text becomes the control's tooltip.
   *
   * @param {{value?: number, variant?: 'full'|'compact', onChange?: (level:number)=>void}} opts
   * @returns {{el: HTMLElement, getLevel: () => number, setLevel: (n:number) => void}}
   */
  createSlider({ value = null, variant = 'full', onChange = null } = {}) {
    const level = this.clamp(value ?? this.default);
    const wrap = document.createElement('div');
    wrap.className = `difficulty-slider difficulty-slider--${variant}`;

    const range = document.createElement('input');
    range.type = 'range';
    range.className = 'difficulty-range';
    range.min = String(this.min);
    range.max = String(this.max);
    range.step = '1';
    range.value = String(level);
    range.setAttribute('aria-label', 'Poziom trudności');

    let head, emojiEl, nameEl, levelEl, summaryEl, expectEl, slowEl, compactEl;

    if (variant === 'full') {
      head = document.createElement('div');
      head.className = 'difficulty-head';
      emojiEl = document.createElement('span');
      emojiEl.className = 'difficulty-emoji';
      nameEl = document.createElement('span');
      nameEl.className = 'difficulty-name';
      levelEl = document.createElement('span');
      levelEl.className = 'difficulty-level';
      head.append(emojiEl, nameEl, levelEl);

      const scale = document.createElement('div');
      scale.className = 'difficulty-scale';
      const lo = document.createElement('span'); lo.textContent = 'Łatwiej';
      const hi = document.createElement('span'); hi.textContent = 'Trudniej';
      scale.append(lo, hi);

      const feedback = document.createElement('div');
      feedback.className = 'difficulty-feedback';
      summaryEl = document.createElement('p');
      summaryEl.className = 'difficulty-summary';
      expectEl = document.createElement('p');
      expectEl.className = 'difficulty-expect';
      slowEl = document.createElement('p');
      slowEl.className = 'difficulty-slow';
      slowEl.textContent = '⏳ Ten poziom myśli przed ruchem — komputer odpowiada z lekkim opóźnieniem.';
      feedback.append(summaryEl, expectEl, slowEl);

      wrap.append(head, range, scale, feedback);
    } else {
      compactEl = document.createElement('span');
      compactEl.className = 'difficulty-compact-label';
      wrap.append(range, compactEl);
    }

    const render = () => {
      const n = Number(range.value);
      const info = this.info(n);
      wrap.dataset.level = String(n);
      wrap.dataset.engine = info.engine;
      if (variant === 'full') {
        emojiEl.textContent = info.emoji;
        nameEl.textContent = info.name;
        levelEl.textContent = `poziom ${n}/${this.max}`;
        summaryEl.textContent = info.summary;
        expectEl.textContent = info.expect;
        slowEl.hidden = !info.slow;
      } else {
        compactEl.textContent = `${info.emoji} ${n}`;
        wrap.title = `${info.name} (${n}/${this.max}) — ${info.summary} ${info.expect}`.trim();
      }
    };

    range.addEventListener('input', () => {
      render();
      onChange?.(Number(range.value));
    });
    render();

    return {
      el: wrap,
      getLevel: () => Number(range.value),
      setLevel: (n) => { range.value = String(Difficulty.clamp(n)); render(); },
    };
  },
};
