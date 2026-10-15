(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const kindTabs = [...document.querySelectorAll('[data-scan-kind]')];
  let cases = [];
  let kind = 'backdoor';
  const rememberedCases = { backdoor: 'masq-clock', benign: null };
  const rememberedPathways = new Map();
  const componentsFor = (item) => item.attackFamily === 'PoisonLoRA TI' ? { unet: true, te: false } : item.scan.components;
  const el = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  function renderRankings(item, pathway) {
    rememberedPathways.set(item.id, pathway);
    $('scan-pathway-select').value = pathway;
    const ranking = item.scan.rankings[pathway].slice(0, 50);
    $('scan-token-ranking').replaceChildren(...ranking.map((token) => {
      const row = el('li', undefined, 'scan-ranked-token');
      const hasDuplicateText = ranking.some((other) => other.token === token.token && other.tokenId !== token.tokenId);
      const readableToken = hasDuplicateText && token.piece.endsWith('</w>') ? `${token.token}</w>` : token.token;
      const displayedToken = readableToken.replace(/[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2066-\u2069\u00b7\u2022\u25cf\u25cb\u26aa\u26ab]/g, (char) => `[U+${char.codePointAt(0).toString(16).toUpperCase().padStart(4, '0')}]`);
      const rank = el('span', String(token.rank), 'scan-token-rank');
      rank.setAttribute('aria-hidden', 'true');
      const label = el('code', displayedToken, 'scan-token-text');
      const score = el('span', token.score.toFixed(2), 'scan-token-score');
      score.setAttribute('aria-label', `Token score ${token.score.toFixed(2)}`);
      row.append(rank, label, score);
      return row;
    }));
  }
  function selectCase(id) {
    const item = cases.find((entry) => entry.id === id && entry.kind === kind);
    if (!item) return;
    rememberedCases[kind] = id;
    $('scan-case-select').value = id;
    $('scan-case-title').textContent = item.title;
    $('scan-case-trigger').hidden = item.kind !== 'backdoor';
    const triggerText = item.trigger === '\u0430' ? `${item.trigger} (Unicode U+0430)` : item.trigger;
    $('scan-trigger-text').textContent = item.kind === 'backdoor' ? triggerText : '';
    const modules = componentsFor(item);
    $('scan-adaptation').textContent = [modules.unet ? 'U-Net' : null, modules.te ? 'TE' : null].filter(Boolean).join(' + ');
    for (const component of ['unet', 'te']) {
      const value = item.scan[component === 'te' ? 'A_TE' : 'A_UNet'];
      $('scan-' + component + '-score').textContent = modules[component] && Number.isFinite(value) ? value.toFixed(4) : 'unavailable';
    }
    const availablePathways = ['unet', 'te'].filter((pathway) => modules[pathway] && item.scan.rankings[pathway]?.length);
    $('scan-pathway-select').replaceChildren(...availablePathways.map((pathway) => {
      const option = el('option', pathway === 'te' ? 'TE' : 'U-Net');
      option.value = pathway;
      return option;
    }));
    $('scan-pathway-select').disabled = availablePathways.length < 2;
    const preferredPathway = rememberedPathways.get(item.id) || item.defaultPathway;
    const sourceLink = $('scan-source-link');
    sourceLink.hidden = !item.source;
    if (item.source) {
      sourceLink.href = item.source.url;
      sourceLink.textContent = `${item.source.platform} ↗`;
    } else sourceLink.removeAttribute('href');
    renderRankings(item, availablePathways.includes(preferredPathway) ? preferredPathway : availablePathways[0]);
  }
  function selectKind(next) {
    kind = next;
    kindTabs.forEach((tab) => {
      const selected = tab.dataset.scanKind === kind;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    $('scan-case-panel').setAttribute('aria-labelledby', `kind-${kind}`);
    const available = cases.filter((item) => item.kind === kind);
    const optionFor = (item) => {
      const option = el('option', item.title);
      option.value = item.id;
      return option;
    };
    if (kind === 'benign') {
      const groups = [
        ['U-Net + TE', available.filter((item) => item.scan.components.unet && item.scan.components.te)],
        ['U-Net only', available.filter((item) => item.scan.components.unet && !item.scan.components.te)],
        ['TE only', available.filter((item) => !item.scan.components.unet && item.scan.components.te)],
      ].filter(([, items]) => items.length);
      $('scan-case-select').replaceChildren(...groups.map(([label, items]) => {
        const group = el('optgroup');
        group.label = `${label} (${items.length})`;
        group.append(...items.map(optionFor));
        return group;
      }));
      if (!rememberedCases.benign) rememberedCases.benign = groups[0][1][0].id;
    } else $('scan-case-select').replaceChildren(...available.map(optionFor));
    selectCase(rememberedCases[kind]);
  }
  kindTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => { if (cases.length) selectKind(tab.dataset.scanKind); });
    tab.addEventListener('keydown', (event) => {
      if (!cases.length || !['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? kindTabs.length - 1 : (index + 1) % kindTabs.length;
      kindTabs[next].focus();
      selectKind(kindTabs[next].dataset.scanKind);
    });
  });
  $('scan-case-select').addEventListener('change', (event) => selectCase(event.target.value));
  $('scan-pathway-select').addEventListener('change', (event) => {
    const item = cases.find((entry) => entry.id === rememberedCases[kind] && entry.kind === kind);
    const pathway = event.target.value;
    if (item && componentsFor(item)[pathway] && item.scan.rankings[pathway]?.length) renderRankings(item, pathway);
  });
  fetch('assets/scan-cases.json?v=20261015-benign85').then((response) => {
    if (!response.ok) throw new Error('Saved scans unavailable');
    return response.json();
  }).then((data) => {
    cases = data.cases;
    kindTabs.forEach((tab) => {
      const count = cases.filter((item) => item.kind === tab.dataset.scanKind).length;
      tab.textContent = `${tab.dataset.scanKind === 'backdoor' ? 'Backdoored' : 'Benign'} LoRAs (${count})`;
    });
    selectKind('backdoor');
    $('scan-load-status').hidden = true;
    $('scan-case-panel').hidden = false;
  }).catch(() => {
    $('scan-load-status').textContent = 'Scan examples could not load. Please refresh the page.';
    kindTabs.forEach((tab) => { tab.disabled = true; });
  });
})();
