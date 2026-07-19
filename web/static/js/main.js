'use strict';

document.addEventListener('DOMContentLoaded', () => {
  const api        = new ApiClient();
  const boardEl    = new BoardRenderer('board');
  const controller = new GameController(api, boardEl);
  new ScanController(api, controller);
  new BenchmarkController(api, controller);

  controller.init();
});
