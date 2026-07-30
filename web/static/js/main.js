'use strict';

document.addEventListener('DOMContentLoaded', () => {
  // Turn every .segmented markup block into a <select>-like control before
  // the controllers grab references to them (see js/ui.js).
  initAllSegmented();

  const api        = new ApiClient();
  const boardEl    = new BoardRenderer('board');
  const controller = new GameController(api, boardEl);
  new ScanController(api, controller);
  new BenchmarkController(api, controller);

  // Difficulty level descriptions drive the setup slider's feedback text and
  // every "which level is this bot?" label, so fetch them before the first
  // render. A failed load is not fatal (see js/difficulty.js), hence no catch
  // beyond the one inside load().
  Difficulty.load(api).then(() => controller.onDifficultyLevelsLoaded());

  controller.init();
});
