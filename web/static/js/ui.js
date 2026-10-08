'use strict';

// Small shared UI widgets. Loaded before the controllers so they can rely on
// the upgraded elements existing by the time they read them.

/**
 * Segmented control: a row of buttons that behaves like a <select> for the
 * code reading it -- `el.value` returns the active button's data-value,
 * assigning to it moves the selection, and clicking a button fires a
 * 'change' event on the container. That keeps every call site that used to
 * talk to a <select id="...-sort"> working untouched.
 *
 * Preferred over a native dropdown for the sort pickers: all options are
 * visible at a glance (no "what else is in there?"), and picking one is a
 * single tap on mobile instead of a full-screen native picker.
 */
function initSegmented(el) {
  if (!el || el.dataset.segmentedReady) return el;
  el.dataset.segmentedReady = '1';

  const buttons = [...el.querySelectorAll('.seg-btn')];
  if (buttons.length === 0) return el;
  let value = (buttons.find(b => b.classList.contains('seg-btn--active')) ?? buttons[0]).dataset.value;

  const sync = () => {
    for (const b of buttons) {
      const active = b.dataset.value === value;
      b.classList.toggle('seg-btn--active', active);
      b.setAttribute('aria-pressed', active ? 'true' : 'false');
    }
  };

  Object.defineProperty(el, 'value', {
    get: () => value,
    set: v => {
      if (!buttons.some(b => b.dataset.value === v) || v === value) return;
      value = v;
      sync();
    },
  });

  for (const b of buttons) {
    b.addEventListener('click', () => {
      if (b.dataset.value === value) return;
      value = b.dataset.value;
      sync();
      el.dispatchEvent(new Event('change'));
    });
  }
  sync();
  return el;
}

/** Upgrade every segmented control in the document (or a subtree). */
function initAllSegmented(root = document) {
  root.querySelectorAll('.segmented').forEach(initSegmented);
}
