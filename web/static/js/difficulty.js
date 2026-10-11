'use strict';

/** Server-backed difficulty descriptions for bot labels and move logs. */
const Difficulty = {
  min: 1,
  max: 10,
  default: 5,
  loaded: false,
  _byLevel: new Map(),

  async load(api, language = null) {
    try {
      const data = await api.getDifficultyLevels(language);
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

  emoji(level) { return ''; },
  name(level)  { return this.info(level).name; },

  /** Compact bot label for move logs and benchmark cards. */
  label(level) { return `${this.name(level)} · ${Number(level)}/10`; },

  clamp(level) {
    const n = Number(level);
    if (!Number.isFinite(n)) return this.default;
    return Math.max(this.min, Math.min(this.max, Math.round(n)));
  },

};
