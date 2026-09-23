/**
 * Real KaTeX parse check for AI-generated math, used by
 * app/services/katex_check.py.
 *
 * Reads {"segments": ["\\theta", "37^\\circ", ...]} on stdin and writes
 * {"ok": true, "errors": {"<segment>": "<KaTeX message>"}} on stdout. Parsing
 * is what we want, not rendering, so throwOnError is on and the output HTML is
 * discarded.
 *
 * Mirrors the frontend's KaTeX options (src/shared/math-content/katexConfig.ts)
 * — in particular the mhchem extension, without which every \ce{...} formula
 * the chemistry prompt is told to emit would be reported as a parse failure.
 */
'use strict';

function loadKatex() {
  const candidates = [];
  if (process.env.KATEX_MODULE_PATH) candidates.push(process.env.KATEX_MODULE_PATH);
  candidates.push('katex');
  candidates.push(require('path').resolve(
    __dirname, '../../../../remix-of-genverse-eduverse/node_modules/katex'));
  for (const c of candidates) {
    try {
      const katex = require(c);
      // mhchem ships at a couple of different paths depending on how the
      // package was built/installed; try each before giving up, because
      // without it every \ce{...} formula the chemistry prompt is told to
      // emit would be reported as a parse failure.
      const mhchemPaths = [
        c + '/contrib/mhchem',
        c + '/dist/contrib/mhchem.js',
        c + '/contrib/mhchem/mhchem.js',
      ];
      for (const m of mhchemPaths) {
        try { require(m); break; } catch (_) { /* try next */ }
      }
      return katex;
    } catch (_) { /* try next */ }
  }
  return null;
}

let input = '';
process.stdin.on('data', (d) => { input += d; });
process.stdin.on('end', () => {
  const katex = loadKatex();
  if (!katex) {
    process.stdout.write(JSON.stringify({ ok: false, reason: 'katex-not-found' }));
    return;
  }
  let segments;
  try {
    segments = JSON.parse(input).segments || [];
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, reason: 'bad-input' }));
    return;
  }
  const errors = {};
  for (const seg of segments) {
    try {
      katex.renderToString(seg, { throwOnError: true, strict: false });
    } catch (e) {
      errors[seg] = String((e && e.message) || e).slice(0, 300);
    }
  }
  process.stdout.write(JSON.stringify({ ok: true, errors }));
});
