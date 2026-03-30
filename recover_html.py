"""
Recover the corrupted manual_unlabeled_review.html.

Strategy:
1. Extract enriched JSON (with explanations + merchant_names) from the corrupted file.
2. Load the clean HTML template from the notebook cell source.
3. Inject the JSON into the template (__DATA_JSON__ placeholder).
4. Apply all UI enhancement patches.
5. Write the final file.
"""
import json
from pathlib import Path

HTML_PATH  = Path('outputs/manual_unlabeled_review.html')
NB_PATH    = Path('explainability_gnnexplainer.ipynb')

# ── Step 1: Extract enriched JSON from corrupted HTML ─────────────────────
print("Step 1: extracting JSON from corrupted HTML...")
h_corrupt = HTML_PATH.read_text(encoding='utf-8')

end_marker_pos = h_corrupt.find(';\nconst LS_KEY')
json_str_raw   = h_corrupt[5700:end_marker_pos]
print(f"  Raw JSON slice: {len(json_str_raw):,} chars")

# Fix: the JSON may contain a residual fragment after the closing brace
try:
    data = json.loads(json_str_raw)
    print(f"  Parsed OK on first try.")
except json.JSONDecodeError as e:
    json_str_raw = json_str_raw[:e.pos]
    data = json.loads(json_str_raw)
    print(f"  Trimmed to {len(json_str_raw):,} chars; parsed OK.")

print(f"  Customers: {len(data['customers'])}")
print(f"  Keys: {list(data.keys())}")
assert 'score_descriptions' in data, "Missing score_descriptions"
assert 'mcc_map' in data, "Missing mcc_map"
has_exp = sum(1 for c in data['customers'] if c.get('explanation'))
print(f"  Customers with explanation: {has_exp}/250")

# ── Step 2: Load clean template from notebook ──────────────────────────────
print("\nStep 2: loading clean template from notebook...")
nb  = json.load(NB_PATH.open(encoding='utf-8'))
src = ''.join(nb['cells'][40]['source'])
t0  = src.find("html_template = '''") + len("html_template = '''")
t1  = src.find("'''", t0)
template = src[t0:t1]
print(f"  Template length: {len(template):,} chars")
assert '__DATA_JSON__' in template, "Placeholder missing from template"

# ── Step 3: Inject enriched JSON into template ─────────────────────────────
print("\nStep 3: injecting JSON into template...")
enriched_json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
html = template.replace('__DATA_JSON__', enriched_json_str)
assert 'const DATA = ' in html, "DATA block missing after injection"
print(f"  HTML after injection: {len(html):,} chars")

# ── Step 4: Apply UI enhancement patches ──────────────────────────────────
print("\nStep 4: applying patches...")

# 4a. Transaction table header: add Merchant column
p_header_old = '<th>datetime</th><th>amount_cad</th><th>merchant_category</th><th>city</th><th>cash</th><th>ecom</th>'
p_header_new = '<th>datetime</th><th>amount_cad</th><th>merchant_category</th><th>merchant</th><th>city</th><th>cash</th><th>ecom</th>'
if p_header_old in html:
    html = html.replace(p_header_old, p_header_new)
    print("  ✓ table header")
else:
    print("  ✗ table header NOT found")

# 4b. Transaction row template: add merchant_name cell
# Template uses fmt(Number(t.amount||0),2) — must match exactly
p_row_old = """`
    <tr>
      <td>${t.dt || ''}</td>
      <td>${fmt(Number(t.amount || 0),2)}</td>
      <td>${t.merchant_category || ''}</td>
      <td>${t.city || ''}</td>
      <td>${t.cash_indicator ?? ''}</td>
      <td>${t.ecommerce_ind ?? ''}</td>
    </tr>
  `).join('');"""
p_row_new = """`
    <tr>
      <td>${t.dt || ''}</td>
      <td>${fmt(Number(t.amount || 0),2)}</td>
      <td>${t.merchant_category || ''}</td>
      <td class="muted">${t.merchant_name || ''}</td>
      <td>${t.city || ''}</td>
      <td>${t.cash_indicator ?? ''}</td>
      <td>${t.ecommerce_ind ?? ''}</td>
    </tr>
  `).join('');"""
if p_row_old in html:
    html = html.replace(p_row_old, p_row_new)
    print("  ✓ transaction row")
else:
    print("  ✗ transaction row NOT found")

# 4c. Replace raw card with explanation card + collapsible raw + glossary
p_card_old = """      <div class="card">
        <h3>Full Model / Profile Context</h3>
        <pre id="raw"></pre>
      </div>"""
p_card_new = """      <div class="card" style="border-left:4px solid #0b5fff;margin-bottom:10px;">
        <h3 style="margin-bottom:6px;">&#128269; Why the Model Flagged This Customer</h3>
        <div id="explanationText" style="font-size:13px;line-height:1.7;color:#1f2a37;"></div>
      </div>

      <details style="margin-bottom:10px;">
        <summary style="cursor:pointer;padding:8px 12px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;font-size:13px;font-weight:600;list-style:none;">
          &#128196; Full Model / Profile Context (JSON)
        </summary>
        <div style="padding-top:4px;">
          <pre id="raw" style="background:#0b1220;color:#e5e7eb;padding:10px;border-radius:8px;font-size:11px;max-height:280px;overflow:auto;"></pre>
        </div>
      </details>

      <details style="margin-bottom:10px;">
        <summary style="cursor:pointer;padding:8px 12px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;font-size:13px;font-weight:600;list-style:none;">
          &#128214; Score Glossary &mdash; what each column means
        </summary>
        <div style="padding-top:4px;max-height:320px;overflow:auto;border:1px solid #e5e7eb;border-radius:8px;">
          <table style="width:100%;border-collapse:collapse;background:#fff;">
            <thead><tr><th style="background:#f8fafc;padding:6px;text-align:left;font-size:12px;">Score Column</th><th style="background:#f8fafc;padding:6px;text-align:left;font-size:12px;">Description</th></tr></thead>
            <tbody id="scoreGlossaryBody"></tbody>
          </table>
        </div>
      </details>"""
if p_card_old in html:
    html = html.replace(p_card_old, p_card_new)
    print("  ✓ raw card → explanation + glossary cards")
else:
    print("  ✗ raw card NOT found")

# 4d. JS: render explanation text and populate score glossary
p_js_old = """};\n  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);\n\n  loadLabel(c.customer_id);\n}"""
p_js_new = """};\n  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);\n\n  // Show explanation\n  const expEl = document.getElementById('explanationText');\n  if (expEl) {\n    const lines = (c.explanation || '(No explanation generated.)').split('. ');\n    expEl.innerHTML = lines.map(l => l.trim() ? `<p style="margin:0 0 8px 0;">${l.trim()}${l.trim().endsWith('.') ? '' : '.'}</p>` : '').join('');\n  }\n\n  // Populate score glossary (once on first render)\n  const glossaryBody = document.getElementById('scoreGlossaryBody');\n  if (glossaryBody && glossaryBody.childElementCount === 0 && DATA.score_descriptions) {\n    glossaryBody.innerHTML = Object.entries(DATA.score_descriptions)\n      .map(([k, v]) => `<tr><td style="font-weight:600;white-space:nowrap;padding:5px 8px;font-size:11px;vertical-align:top;">${k}</td><td style="padding:5px 8px;font-size:11px;color:#667085;">${v}</td></tr>`)\n      .join('');\n  }\n\n  loadLabel(c.customer_id);\n}"""
if p_js_old in html:
    html = html.replace(p_js_old, p_js_new)
    print("  ✓ raw JS → explanation + glossary JS")
else:
    print("  ✗ raw JS NOT found — trying alternate...")
    # Try without the final }
    alt_old = """};\n  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);\n\n  loadLabel(c.customer_id);"""
    alt_new = """};\n  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);\n\n  const expEl = document.getElementById('explanationText');\n  if (expEl) {\n    const lines = (c.explanation || '(No explanation generated.)').split('. ');\n    expEl.innerHTML = lines.map(l => l.trim() ? `<p style="margin:0 0 8px 0;">${l.trim()}${l.trim().endsWith('.') ? '' : '.'}</p>` : '').join('');\n  }\n  const glossaryBody = document.getElementById('scoreGlossaryBody');\n  if (glossaryBody && glossaryBody.childElementCount === 0 && DATA.score_descriptions) {\n    glossaryBody.innerHTML = Object.entries(DATA.score_descriptions)\n      .map(([k, v]) => `<tr><td style="font-weight:600;white-space:nowrap;padding:5px 8px;font-size:11px;vertical-align:top;">${k}</td><td style="padding:5px 8px;font-size:11px;color:#667085;">${v}</td></tr>`)\n      .join('');\n  }\n\n  loadLabel(c.customer_id);"""
    if alt_old in html:
        html = html.replace(alt_old, alt_new)
        print("    ✓ alternate raw JS patch applied")
    else:
        print("    ✗ alternate also not found")

# 4e. List row: add score description tooltip
p_list_old = """    div.innerHTML = `
      <div><b>${c.customer_id}</b></div>
      <div class="muted">${c.primary_score_col}: <span class="score">${fmt(c.primary_score, 5)}</span> | txns: ${c.profile.txn_count}</div>
      <div class="muted">component: ${c.model_context.component ?? ''} | hdb: ${c.model_context.hdb_component ?? ''}</div>
      <div class="muted">label: ${lab || 'unlabeled'} ${conf ? `(conf ${conf})` : ''}</div>
    `;"""
p_list_new = """    const scoreDesc = DATA.score_descriptions ? (DATA.score_descriptions[c.primary_score_col] || '') : '';
    div.innerHTML = `
      <div><b>${c.customer_id}</b></div>
      <div class="muted" title="${scoreDesc.replace(/"/g, '&quot;')}">${c.primary_score_col}: <span class="score">${fmt(c.primary_score, 5)}</span> | txns: ${c.profile.txn_count}</div>
      <div class="muted">component: ${c.model_context.component ?? ''} | hdb: ${c.model_context.hdb_component ?? ''}</div>
      <div class="muted">label: ${lab || 'unlabeled'} ${conf ? `(conf ${conf})` : ''}</div>
    `;"""
if p_list_old in html:
    html = html.replace(p_list_old, p_list_new)
    print("  ✓ list row tooltip")
else:
    print("  ✗ list row NOT found")

# ── Step 5: Verify and write ───────────────────────────────────────────────
print("\nStep 5: final verification...")
assert 'const DATA = ' in html, "ERROR: DATA block missing"
assert '<script>' in html, "ERROR: script tag missing"
assert 'function renderList' in html
assert 'function renderDetail' in html
assert 'explanationText' in html
assert 'scoreGlossaryBody' in html
d_pos = html.find('const DATA = ')
ls_pos = html.find(';\nconst LS_KEY')
print(f"  const DATA = @ {d_pos}")
print(f"  ;\\nconst LS_KEY @ {ls_pos}")
print(f"  HTML size: {len(html):,} bytes")

HTML_PATH.write_text(html, encoding='utf-8')
print(f"\n✓ Wrote {HTML_PATH}")
print("  Open in browser — explanation panel + score glossary + merchant column ready.")
