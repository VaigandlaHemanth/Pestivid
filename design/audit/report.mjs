import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const DIR = path.dirname(fileURLToPath(import.meta.url));
const R = JSON.parse(readFileSync(path.join(DIR, 'measurements.json'), 'utf8'));
const what = process.argv[2] || 'all';

const say = s => console.log(s);

if (what === 'frames' || what === 'all') {
  say('\n===== 1. CANVAS FRAMES vs ACTUAL RENDERED SIZE =====');
  for (const r of R) {
    const dw = Math.abs(r.rootW - r.w), dh = Math.abs(r.rootH - r.h);
    if (dw > 2 || dh > 2) say(`  ${r.file.replace('.dc.html','').padEnd(12)} declared ${String(r.w).padStart(5)}x${String(r.h).padEnd(5)} actual ${String(Math.ceil(r.rootW)).padStart(5)}x${Math.ceil(r.rootH)}`);
  }
}

if (what === 'clip' || what === 'all') {
  say('\n===== 2. CONTENT CLIPPED AT 100% =====');
  for (const r of R) for (const c of r.clipped)
    say(`  ${r.file.replace('.dc.html','').padEnd(12)} cut ${String(c.cut).padStart(4)}px  (box ${c.clientH}px)  "${c.first}"`);
}

if (what === 'scale' || what === 'all') {
  for (const [k, lbl] of [['scale130', '130%'], ['scale200', '200%']]) {
    say(`\n===== 3. CLIPPED WHEN PHONE FONT SCALE IS ${lbl} =====`);
    for (const r of R) {
      if (r.w > 400) continue;
      for (const c of (r[k].clipped || [])) say(`  ${r.file.replace('.dc.html','').padEnd(12)} cut ${String(c.cut).padStart(4)}px  "${c.first}"`);
    }
  }
}

if (what === 'contrast' || what === 'all') {
  say('\n===== 4. CONTRAST — APCA Lc, and the WCAG number beside it =====');
  say('     APCA guidance: Lc 90 body, 75 min for <=18px, 60 for 24px+, 45 large/bold only');
  const rows = [];
  for (const r of R) for (const t of r.textEls) {
    const need = t.size >= 24 ? 60 : t.size >= 18 ? 68 : 75;
    if (Math.abs(t.apca) < need) rows.push({ f: r.file.replace('.dc.html',''), ...t, need });
  }
  rows.sort((a, b) => Math.abs(a.apca) - Math.abs(b.apca));
  for (const t of rows.slice(0, 40))
    say(`  Lc ${String(Math.round(Math.abs(t.apca))).padStart(3)} (need ${t.need}, wcag ${String(t.wcag).padStart(5)})  ${String(Math.round(t.size)).padStart(2)}px w${t.weight}  ${t.f.padEnd(11)} ${t.color} on ${t.bg}  "${t.text}"`);
  say(`  ... ${rows.length} total below the APCA floor for their size`);
}

if (what === 'targets' || what === 'all') {
  say('\n===== 5. CONTROL-LIKE BLOCKS UNDER THE TARGET FLOOR =====');
  for (const r of R) {
    const mob = r.w <= 400, floor = mob ? 44 : 24;
    const bad = r.controls.filter(c => c.h < floor || c.w < floor);
    for (const c of bad) say(`  ${r.file.replace('.dc.html','').padEnd(12)} ${String(Math.round(c.w)).padStart(4)}x${String(Math.round(c.h)).padEnd(4)} (floor ${floor})  "${c.label}"`);
  }
}

if (what === 'tnum' || what === 'all') {
  say('\n===== 6. NUMBERS WITHOUT TABULAR FIGURES =====');
  let n = 0;
  for (const r of R) for (const t of r.textEls) {
    if (t.hasDigits && !t.tnum && /[₹%]|\d{3}|\d+\s?(MB|GB|q|acre|quintal|km|sec)/.test(t.text)) {
      say(`  ${r.file.replace('.dc.html','').padEnd(12)} "${t.text}"`); n++;
    }
  }
  say(`  ${n} total`);
}
