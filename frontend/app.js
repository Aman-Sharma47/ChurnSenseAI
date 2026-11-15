/* ChurnSense AI frontend. Talks to the FastAPI service with fetch(). No libraries. */
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let MODEL = null, META = null, SCORED = null, LASTVALUES = {};

/* ---------------- helpers ---------------- */
async function api(path, opt = {}) {
  const r = await fetch(path, opt);
  if (!r.ok) {
    let t = await r.text();
    try { t = JSON.parse(t).detail || t; } catch (e) {}
    throw new Error(typeof t === 'string' ? t : JSON.stringify(t));
  }
  const ct = r.headers.get('content-type') || '';
  return ct.includes('json') ? r.json() : r.text();
}
const postJSON = (p, b) => api(p, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b)});
function postFile(p, file, extra = {}) {
  const f = new FormData(); f.append('file', file);
  for (const k in extra) f.append(k, extra[k]);
  return api(p, {method: 'POST', body: f});
}
function toast(msg, ok = false) {
  const t = $('#toast'); t.textContent = msg; t.className = 'toast' + (ok ? ' ok' : '');
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add('hidden'), 6000);
}
const guard = fn => async (...a) => { try { await fn(...a); } catch (e) { toast(e.message); } };
const nice = v => Math.abs(v) >= 1000 ? (v / 1000).toFixed(0) + 'k' : String(Number(v.toFixed(2)));
const pct = v => (v * 100).toFixed(1) + '%';
const kpi = (l, v, s = '') => `<div class="kpi"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`;

function parseCSV(t) {
  const rows = []; let row = [], cur = '', q = false; t = t.replace(/^\uFEFF/, '');
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (q) { if (c === '"') { if (t[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(cur); cur = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && t[i + 1] === '\n') i++;
      row.push(cur); cur = ''; if (row.length > 1 || row[0] !== '') rows.push(row); row = [];
    } else cur += c;
  }
  if (cur !== '' || row.length) { row.push(cur); rows.push(row); }
  return rows;
}
const toCSV = rows => rows.map(r => r.map(v => /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v).join(',')).join('\n');

function bars(items, {fmt = v => v.toFixed(3), color = 'var(--accent)'} = {}) {
  const mx = Math.max(...items.map(i => i.value), 1e-9);
  return items.map(i => `<div class="brow"><div class="lab" title="${esc(i.label)}">${esc(i.label)}</div>
    <div class="bar"><i style="width:${Math.max(i.value / mx * 100, 1)}%;background:${i.color || color}"></i></div>
    <div>${fmt(i.value)}</div></div>`).join('');
}

function lineChart(series, o = {}) {
  const W = 560, H = 300, p = {l: 54, r: 14, t: 12, b: 42};
  const all = series.flatMap(s => s.pts);
  const xmin = o.xmin ?? Math.min(...all.map(d => d[0])), xmax = o.xmax ?? Math.max(...all.map(d => d[0]));
  const ymin = o.ymin ?? Math.min(0, ...all.map(d => d[1])), ymax = o.ymax ?? Math.max(...all.map(d => d[1]));
  const sx = x => p.l + (x - xmin) / ((xmax - xmin) || 1) * (W - p.l - p.r);
  const sy = y => H - p.b - (y - ymin) / ((ymax - ymin) || 1) * (H - p.t - p.b);
  let g = '';
  for (let i = 0; i <= 5; i++) {
    const xv = xmin + (xmax - xmin) * i / 5, yv = ymin + (ymax - ymin) * i / 5;
    g += `<line class="grid" x1="${p.l}" x2="${W - p.r}" y1="${sy(yv)}" y2="${sy(yv)}"/>
      <text class="tick" x="${p.l - 6}" y="${sy(yv) + 4}" text-anchor="end">${nice(yv)}</text>
      <text class="tick" x="${sx(xv)}" y="${H - p.b + 16}" text-anchor="middle">${nice(xv)}</text>`;
  }
  if (o.diag) g += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="#58657f" stroke-dasharray="5 5"/>`;
  if (o.vline != null) g += `<line x1="${sx(o.vline)}" x2="${sx(o.vline)}" y1="${p.t}" y2="${H - p.b}" stroke="#f1b44c" stroke-dasharray="4 4"/>`;
  series.forEach(s => {
    const d = s.pts.map((q, i) => (i ? 'L' : 'M') + sx(q[0]).toFixed(1) + ' ' + sy(q[1]).toFixed(1)).join(' ');
    g += `<path d="${d}" fill="none" stroke="${s.color || 'var(--accent)'}" stroke-width="2.4"/>`;
  });
  g += `<text class="axl" x="${W / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xl || '')}</text>
        <text class="axl" transform="translate(13 ${H / 2}) rotate(-90)" text-anchor="middle">${esc(o.yl || '')}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`;
}

/* ---------------- navigation ---------------- */
$('#nav').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $$('#nav button').forEach(x => x.classList.toggle('active', x === b));
  $$('.tab').forEach(t => t.classList.toggle('active', t.id === 'tab-' + b.dataset.tab));
  if (b.dataset.tab === 'business') guard(loadBusiness)();
});

/* ---------------- models ---------------- */
async function loadModels() {
  const models = await api('/models');
  const sel = $('#modelSel');
  if (!models.length) { sel.innerHTML = '<option>No models yet</option>'; toast('No models yet. Use the Train tab.'); return; }
  sel.innerHTML = models.map(m =>
    `<option value="${esc(m.model_id)}">${esc(m.dataset)} | ${esc(m.best_model)} | AUC ${m.roc_auc} | ${esc(m.model_id.slice(-6))}</option>`).join('');
  await selectModel(sel.value);
}
$('#modelSel').addEventListener('change', guard(e => selectModel(e.target.value)));

async function selectModel(id) {
  MODEL = id; SCORED = null;
  META = await api('/models/' + id);
  $('#modelNote').textContent = id;
  renderOverview(); buildForm();
  $('#result').classList.add('hidden'); $('#pfOut').classList.add('hidden'); $('#mnOut').classList.add('hidden');
  $('#chatLog').innerHTML = ''; $('#chStatus').textContent = '';
  if ($('#tab-business').classList.contains('active')) guard(loadBusiness)();
}

/* ---------------- overview ---------------- */
async function renderOverview() {
  const t = META.test_at_tuned, s = META.schema;
  $('#kpis').innerHTML = kpi('Best model', META.best_model, META.statistically_tied.length > 1 ? 'tied: ' + META.statistically_tied.join(', ') : 'clear winner')
    + kpi('ROC-AUC', t.roc_auc, 'ranking quality')
    + kpi('Accuracy', t.accuracy, 'overall correct')
    + kpi('Recall', t.recall, 'leavers caught')
    + kpi('Precision', t.precision, 'flags that were right')
    + kpi('Cut-off', META.threshold.toFixed(2), `trained on ${META.n_train.toLocaleString()}, tested on ${META.n_test.toLocaleString()}`);
  $('#cm').innerHTML = `<div class="cmgrid">
    <div></div><div class="small muted">Predicted stays</div><div class="small muted">Predicted leaves</div>
    <div class="small muted">Actually stays</div><div class="c good"><b>${t.tn}</b>correct</div><div class="c bad"><b>${t.fp}</b>false alarm</div>
    <div class="small muted">Actually leaves</div><div class="c bad"><b>${t.fn}</b>missed</div><div class="c good"><b>${t.tp}</b>caught</div></div>`;
  $('#schemaInfo').innerHTML = `<p>Target: <b>${esc(s.target)}</b> (churned = ${esc(s.positive_label)}).<br>
    Dropped: ${Object.keys(s.dropped).length ? esc(Object.entries(s.dropped).map(([k, v]) => k + ' (' + v + ')').join(', ')) : 'none'}.<br>
    Warnings: ${s.warnings.length ? esc(s.warnings.join('; ')) : 'none'}.</p>`;
  try {
    const cv = await api(`/models/${MODEL}/curves`);
    $('#rocChart').innerHTML = lineChart([{pts: cv.roc.fpr.map((x, i) => [x, cv.roc.tpr[i]])}],
      {xmin: 0, xmax: 1, ymin: 0, ymax: 1, diag: true, xl: 'False positive rate', yl: 'True positive rate'})
      + `<div class="small muted">AUC ${t.roc_auc}. The dashed line is a coin flip.</div>`;
  } catch (e) { $('#rocChart').textContent = 'No curve data.'; }
  try {
    const imp = await api(`/models/${MODEL}/importance`);
    $('#importance').innerHTML = bars(imp.map(i => ({label: i.feature, value: i.importance})), {fmt: v => v.toFixed(2)});
  } catch (e) { $('#importance').innerHTML = '<span class="muted">No SHAP data for this model. Retrain it to enable explanations.</span>'; }
}

/* ---------------- score one customer ---------------- */
const EXAMPLE = {SeniorCitizen: 0, tenure: 1, MonthlyCharges: 85.7, TotalCharges: 85.7, gender: 'Male', Partner: 'No', Dependents: 'No',
  PhoneService: 'Yes', MultipleLines: 'No', InternetService: 'Fiber optic', OnlineSecurity: 'No', OnlineBackup: 'No', DeviceProtection: 'No',
  TechSupport: 'No', StreamingTV: 'No', StreamingMovies: 'No', Contract: 'Month-to-month', PaperlessBilling: 'Yes', PaymentMethod: 'Electronic check'};

function buildForm() {
  const s = META.schema;
  const f = (n, num) => `<label>${esc(n)}${num ? ' (number)' : ''}<input data-f="${esc(n)}" data-num="${num ? 1 : 0}"></label>`;
  $('#form').innerHTML = s.numeric_features.map(n => f(n, true)).join('') + s.categorical_features.map(n => f(n, false)).join('');
  $('#wfField').innerHTML = s.categorical_features.map(n => `<option>${esc(n)}</option>`).join('');
}
$('#exampleBtn').onclick = () => $$('#form input').forEach(i => { if (i.dataset.f in EXAMPLE) i.value = EXAMPLE[i.dataset.f]; });
$('#clearBtn').onclick = () => $$('#form input').forEach(i => i.value = '');

function readForm() {
  const v = {};
  for (const i of $$('#form input')) {
    const t = i.value.trim(); if (!t) continue;
    if (i.dataset.num === '1') { const n = parseFloat(t); if (isNaN(n)) throw new Error(i.dataset.f + ' must be a number'); v[i.dataset.f] = n; }
    else v[i.dataset.f] = t;
  }
  return v;
}

$('#predictBtn').onclick = guard(async () => {
  const values = readForm(); LASTVALUES = values;
  const btn = $('#predictBtn'); btn.disabled = true;
  try {
    const out = await postJSON(`/models/${MODEL}/predict`, {features: values});
    $('#result').classList.remove('hidden');
    $('#gauge').innerHTML = `<div class="num">${pct(out.churn_probability)}</div>
      <div class="bar"><i class="fill-${out.risk_segment}" style="width:${out.churn_probability * 100}%"></i></div>
      <span class="badge seg-${out.risk_segment}">${out.risk_segment} risk</span>`;
    $('#predMeta').innerHTML = `Will churn: <b>${out.will_churn ? 'yes' : 'no'}</b> (cut-off ${out.threshold.toFixed(2)}).
      ${out.missing_fields.length ? out.missing_fields.length + ' fields were left blank.' : ''}`;
    $('#factors').textContent = 'Loading...'; $('#plan').textContent = 'Writing plan (the first call can take a minute)...';
    postJSON(`/models/${MODEL}/explain`, {features: values}).then(r => {
      $('#factors').innerHTML = r.top_factors.map(f =>
        `<div class="${f.effect > 0 ? 'up' : 'down'}">${f.effect > 0 ? '&#9650;' : '&#9660;'} <b>${esc(f.feature)}</b>: ${esc(f.direction)} (${f.effect > 0 ? '+' : ''}${f.effect.toFixed(2)})</div>`).join('');
    }).catch(e => $('#factors').textContent = 'Explanation unavailable: ' + e.message);
    postJSON(`/models/${MODEL}/recommend`, {features: values}).then(p => {
      $('#plan').innerHTML = `<div class="small muted">Source: ${esc(p.source)} (the model predicts; the LLM only explains)</div>
        <p>${esc(p.summary)}</p><ul class="plan">${p.actions.map(a => `<li>${esc(a)}</li>`).join('')}</ul>
        <div class="callout">${esc(p.message)}</div>`;
    }).catch(e => $('#plan').textContent = 'Plan unavailable: ' + e.message);
  } finally { btn.disabled = false; }
});

$('#wfBtn').onclick = guard(async () => {
  const cands = $('#wfValues').value.split(',').map(x => x.trim()).filter(Boolean);
  if (!Object.keys(LASTVALUES).length) throw new Error('Predict a customer first.');
  if (!cands.length) throw new Error('Type at least one value to try.');
  const r = await postJSON(`/models/${MODEL}/whatif`, {features: LASTVALUES, field: $('#wfField').value, candidates: cands});
  $('#wfOut').innerHTML = `<div class="tablewrap"><table><tr><th>${esc($('#wfField').value)}</th><th>Before</th><th>After</th><th>Change</th></tr>`
    + r.options.map(o => `<tr><td>${esc(o.new_value)}</td><td>${pct(o.before)}</td><td><b>${pct(o.after)}</b></td>
      <td class="${o.change < 0 ? 'down' : o.change > 0 ? 'up' : ''}">${o.change > 0 ? '+' : ''}${(o.change * 100).toFixed(1)} pts</td></tr>`).join('') + '</table></div>';
});

/* ---------------- portfolio ---------------- */
$('#pfBtn').onclick = guard(async () => {
  const file = $('#pfFile').files[0]; if (!file) throw new Error('Choose a CSV first.');
  const btn = $('#pfBtn'); btn.disabled = true; btn.textContent = 'Scoring...';
  try {
    const rows = parseCSV(await postFile(`/models/${MODEL}/predict-batch`, file));
    SCORED = {head: rows[0], rows: rows.slice(1)};
    renderPortfolio();
  } finally { btn.disabled = false; btn.textContent = 'Score file'; }
});

function renderPortfolio() {
  const {head, rows} = SCORED, ip = head.indexOf('churn_probability'), is = head.indexOf('risk_segment');
  const order = ['Low', 'Medium', 'High', 'Critical'], count = {Low: 0, Medium: 0, High: 0, Critical: 0};
  let sum = 0; rows.forEach(r => { count[r[is]]++; sum += parseFloat(r[ip]); });
  $('#pfOut').classList.remove('hidden');
  $('#pfKpis').innerHTML = kpi('Customers', rows.length.toLocaleString()) + kpi('High or Critical', (count.High + count.Critical).toLocaleString())
    + kpi('Mean churn probability', pct(sum / rows.length));
  $('#pfSeg').innerHTML = bars(order.map(k => ({label: k, value: count[k], color: `var(--${k === 'Critical' ? 'crit' : k.toLowerCase().slice(0, k === 'Medium' ? 3 : 4)})`})), {fmt: v => v.toLocaleString()});
  const cats = head.map((h, i) => ({h, i})).filter(c => c.i !== ip && c.i !== is
    && new Set(rows.slice(0, 500).map(r => r[c.i])).size > 1 && new Set(rows.slice(0, 500).map(r => r[c.i])).size <= 20 && isNaN(parseFloat(rows[0][c.i])));
  $('#pfBy').innerHTML = cats.map(c => `<option value="${c.i}">${esc(c.h)}</option>`).join('');
  const drawBy = () => {
    if (!cats.length) { $('#pfByOut').textContent = 'No suitable text columns.'; return; }
    const i = +$('#pfBy').value, g = {};
    rows.forEach(r => { (g[r[i]] ??= []).push(parseFloat(r[ip])); });
    $('#pfByOut').innerHTML = bars(Object.entries(g).map(([k, a]) => ({label: k, value: a.reduce((x, y) => x + y, 0) / a.length})).sort((a, b) => b.value - a.value), {fmt: pct});
  };
  $('#pfBy').onchange = drawBy; drawBy();
  const queue = rows.filter(r => r[is] === 'High' || r[is] === 'Critical');
  const show = [0, 1, 2, 3, 4, 5].filter(i => i < head.length && i !== ip && i !== is).slice(0, 5);
  $('#pfTable').innerHTML = '<tr>' + show.map(i => `<th>${esc(head[i])}</th>`).join('') + '<th>churn_probability</th><th>risk</th></tr>'
    + queue.slice(0, 200).map(r => '<tr>' + show.map(i => `<td>${esc(r[i])}</td>`).join('')
      + `<td>${pct(parseFloat(r[ip]))}</td><td><span class="badge seg-${r[is]}">${r[is]}</span></td></tr>`).join('');
  $('#pfDl').onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([toCSV([head, ...queue])], {type: 'text/csv'}));
    a.download = 'retention_queue.csv'; a.click();
  };
}

/* ---------------- assistant ---------------- */
$('#chLoad').onclick = guard(async () => {
  const file = $('#chFile').files[0]; if (!file) throw new Error('Choose a CSV first.');
  $('#chStatus').textContent = 'Loading...';
  const r = await postFile(`/models/${MODEL}/portfolio`, file);
  $('#chStatus').textContent = r.rows.toLocaleString() + ' customers loaded.';
  $('#chatLog').innerHTML = '';
});
function addMsg(who, html) { const d = document.createElement('div'); d.className = 'msg ' + who; d.innerHTML = html; $('#chatLog').appendChild(d); d.scrollIntoView({behavior: 'smooth', block: 'end'}); return d; }
async function ask(q) {
  if (!q.trim()) return;
  addMsg('user', esc(q)); $('#chatIn').value = '';
  const wait = addMsg('bot', '<span class="muted">Thinking...</span>');
  try {
    const out = await postJSON(`/models/${MODEL}/chat`, {message: q});
    let h = `<div>${esc(out.answer)}</div>`;
    if (out.table && out.table.length) {
      const cols = Object.keys(out.table[0]);
      h += `<div class="tablewrap"><table><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr>`
        + out.table.map(r => '<tr>' + cols.map(c => `<td>${esc(r[c])}</td>`).join('') + '</tr>').join('') + '</table></div>';
    }
    h += `<div class="small muted">Wording: ${esc(out.source)}</div>`;
    if (out.note) h += `<div class="small muted">${esc(out.note)}</div>`;
    wait.innerHTML = h;
  } catch (e) { wait.innerHTML = `<span class="up">${esc(e.message)}</span>`; }
}
$('#chatSend').onclick = () => ask($('#chatIn').value);
$('#chatIn').addEventListener('keydown', e => { if (e.key === 'Enter') ask(e.target.value); });
$$('.chip').forEach(c => c.onclick = () => ask(c.textContent));

/* ---------------- business value ---------------- */
let bzTimer;
async function loadBusiness() {
  if (!MODEL) return;
  const v = $('#bzValue').value, o = $('#bzOffer').value, s = $('#bzSucc').value;
  $('#bzSuccV').textContent = (+s).toFixed(2);
  const r = await api(`/models/${MODEL}/cost?value=${v}&offer_cost=${o}&success_rate=${s}`);
  const b = r.best, d = r.at_default_threshold;
  $('#bzKpis').innerHTML = kpi('Best cut-off', b.threshold) + kpi('Net savings (test set)', Math.round(b.net_savings).toLocaleString())
    + kpi("Savings at the model's own cut-off", Math.round(d.net_savings).toLocaleString(), 'cut-off ' + d.threshold);
  $('#bzChart').innerHTML = lineChart([{pts: r.curve.map(c => [c.threshold, c.net_savings])}], {vline: b.threshold, xl: 'Cut-off', yl: 'Net savings'});
  $('#bzNote').textContent = `At the best cut-off you send ${b.offers_sent} offers and reach ${b.churners_reached} real churners out of ${r.test_rows} test customers. The success rate and prices are your assumptions.`;
}
['#bzValue', '#bzOffer', '#bzSucc'].forEach(s => $(s).addEventListener('input', () => { clearTimeout(bzTimer); bzTimer = setTimeout(guard(loadBusiness), 250); }));

/* ---------------- monitoring ---------------- */
$('#mnCheck').onclick = guard(async () => {
  const file = $('#mnFile').files[0]; if (!file) throw new Error('Choose a CSV first.');
  const d = await postFile(`/models/${MODEL}/drift`, file);
  $('#mnOut').classList.remove('hidden');
  $('#mnKpis').innerHTML = kpi('Drift status', d.status) + kpi('Max PSI', d.max_psi) + kpi('Features above 0.25', d.n_significant);
  $('#mnBars').innerHTML = bars(d.features.slice(0, 15).map(f => ({label: f.feature, value: f.psi,
    color: f.psi > 0.25 ? 'var(--crit)' : f.psi > 0.10 ? 'var(--med)' : 'var(--low)'})), {fmt: v => v.toFixed(3)});
});
async function pollJob(id, box, onDone) {
  box.classList.remove('hidden'); box.textContent = 'Working...';
  for (;;) {
    const j = await api('/jobs/' + id);
    if (j.status === 'running') { await new Promise(r => setTimeout(r, 3000)); continue; }
    if (j.status === 'failed') { box.innerHTML = `<span class="up">Failed: ${esc(j.error)}</span>`; return; }
    onDone(j); return;
  }
}
$('#mnRun').onclick = guard(async () => {
  const file = $('#mnFile').files[0]; if (!file) throw new Error('Choose a CSV first.');
  const r = await postFile(`/models/${MODEL}/monitor`, file, {force: $('#mnForce').checked});
  await pollJob(r.job_id, $('#mnJob'), j => {
    $('#mnJob').innerHTML = `<h3>${esc(j.action)}</h3>` + ('champion_auc' in j
      ? `<div class="kpis">${kpi('Current model AUC', j.champion_auc)}${kpi('Challenger AUC', j.challenger_auc)}${kpi('Promoted', j.promoted ? 'yes' : 'no')}</div>`
        + (j.promoted ? `<p>New model: ${esc(j.new_model_id)}. Reload the page to select it.</p>` : '') : '');
  });
});

/* ---------------- train ---------------- */
let DATASET = null;
$('#trAnalyse').onclick = guard(async () => {
  const file = $('#trFile').files[0]; if (!file) throw new Error('Choose a CSV first.');
  const p = await postFile('/datasets', file); DATASET = p.dataset_id;
  const s = p.schema;
  $('#trOut').classList.remove('hidden');
  $('#trInfo').innerHTML = `<h3>${p.rows.toLocaleString()} rows, ${p.columns} columns, churn rate ${pct(p.churn_rate)}</h3>
    <p>Detected target: <b>${esc(s.target)}</b> (churned = ${esc(s.positive_label)}), confidence ${esc(s.target_confidence)}.</p>
    <p>Dropped: ${Object.keys(s.dropped).length ? esc(Object.entries(s.dropped).map(([k, v]) => k + ' (' + v + ')').join(', ')) : 'none'}.</p>
    <p>${s.numeric_features.length} numeric and ${s.categorical_features.length} text features.</p>
    ${s.warnings.length ? `<p class="up">Warnings: ${esc(s.warnings.join('; '))}</p>` : ''}`;
});
$('#trGo').onclick = guard(async () => {
  if (!DATASET) throw new Error('Analyse a file first.');
  const r = await api(`/datasets/${DATASET}/train`, {method: 'POST', body: (() => { const f = new FormData(); f.append('beta', '1.0'); return f; })()});
  await pollJob(r.job_id, $('#trJob'), async j => {
    $('#trJob').innerHTML = `<h3>Done.</h3><p>New model: ${esc(j.model_id)}</p>`;
    await loadModels();
  });
});

/* ---------------- start ---------------- */
guard(loadModels)();