(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  let examples = [];
  const tabs = [...document.querySelectorAll('[data-example]')];
  function renderExample(id) {
    const item = examples.find((entry) => entry.id === id);
    if (!item) return;
    tabs.forEach((tab) => {
      const selected = tab.dataset.example === id;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    $('example-content').setAttribute('aria-labelledby', `tab-${id}`);
    $('clean-image').src = item.clean;
    $('clean-image').alt = `${item.name} benign output`;
    $('trigger-image').src = item.attack;
    $('trigger-image').alt = `${item.name} backdoor output`;
    $('clean-prompt').textContent = item.cleanPrompt;
    const prompt = $('trigger-prompt');
    prompt.replaceChildren();
    const index = item.attackPrompt.indexOf(item.trigger);
    if (index >= 0) {
      prompt.append(document.createTextNode(item.attackPrompt.slice(0, index)));
      const highlight = document.createElement('mark');
      highlight.textContent = item.trigger;
      prompt.append(highlight, document.createTextNode(item.attackPrompt.slice(index + item.trigger.length)));
    } else { prompt.textContent = item.attackPrompt; }
    $('known-trigger').textContent = item.trigger;
    $('known-target').textContent = item.target;
    const max = Math.max(...item.tokens.map((token) => token.score));
    $('token-list').replaceChildren(...item.tokens.map((token) => {
      const row = document.createElement('li');
      row.classList.add(`token-${token.category || 'unrelated'}`);
      const rank = document.createElement('span');
      rank.textContent = String(token.rank).padStart(2, '0');
      const label = document.createElement('code');
      label.textContent = token.token;
      label.title = token.piece;
      const score = document.createElement('span');
      score.className = 'token-score';
      score.textContent = token.score.toFixed(2);
      score.setAttribute('aria-label', `Token score ${token.score.toFixed(2)}`);
      const bar = document.createElement('span');
      bar.className = 'score-bar';
      bar.style.setProperty('--score-width', `${(token.score / max) * 66}%`);
      bar.setAttribute('aria-hidden', 'true');
      row.append(rank, label, score, bar);
      return row;
    }));
  }
  fetch('assets/examples.json?v=20261015-paper-colors').then((response) => {
    if (!response.ok) throw new Error('Example data unavailable');
    return response.json();
  }).then((data) => { examples = data; renderExample('masq'); }).catch(() => {
    tabs.forEach((tab, index) => { if (index) tab.disabled = true; });
  });
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => renderExample(tab.dataset.example));
    tab.addEventListener('keydown', (event) => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      tabs[next].focus();
      renderExample(tabs[next].dataset.example);
    });
  });
  $('copy-citation').addEventListener('click', async () => {
    const content = $('bibtex').textContent.trim() + '\n';
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(content);
        copied = true;
      }
    } catch (_) { /* Use selection fallback below when clipboard is unavailable. */ }
    if (!copied) {
      const textarea = document.createElement('textarea');
      textarea.value = content;
      textarea.style.cssText = 'position:fixed;left:-9999px;top:0';
      document.body.append(textarea);
      textarea.select();
      try { copied = document.execCommand('copy'); } catch (_) { copied = false; }
      textarea.remove();
    }
    if (!copied) {
      const range = document.createRange();
      range.selectNodeContents($('bibtex'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    }
    $('copy-status').textContent = copied ? 'Citation copied to clipboard.' : 'Citation selected. Copy the selected text.';
    $('copy-citation').textContent = copied ? 'Copied ✓' : 'Citation selected';
    setTimeout(() => { $('copy-citation').textContent = 'Copy citation'; }, 2600);
  });
  if ('IntersectionObserver' in window) {
    const navLinks = [...document.querySelectorAll('.site-header nav a')];
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        navLinks.forEach((link) => {
          const active = link.hash === `#${entry.target.id}`;
          link.classList.toggle('active', active);
          if (active) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      });
    }, { rootMargin: '-15% 0px -65% 0px', threshold: 0 });
    document.querySelectorAll('#overview,#method,#results,#citation').forEach((section) => observer.observe(section));
  }
})();
