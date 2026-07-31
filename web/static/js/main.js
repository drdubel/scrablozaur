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
  // Languages first: the level table depends on which one is selected (a
  // language without a trained leave net has no levels 9-10), and the board
  // renderer needs its point values to label tiles.
  Languages.load(api)
    .then(() => {
      boardEl.setLetterValues(Languages.letterValues(Languages.default));
      controller.onLanguagesLoaded();
      return Difficulty.load(api, Languages.default);
    })
    .then(() => controller.onDifficultyLevelsLoaded());

  controller.init();
});
