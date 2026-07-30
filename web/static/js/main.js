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

  controller.init();
});
