"""
Regenerate manual_unlabeled_review.html using cluster-diverse selection:
  - Find top 100 most suspicious graph components (by mean score of unlabeled members)
  - Pick the single highest-scored unlabeled customer from each component
  - Enrich with explanations, merchant names, score descriptions
  - Write to outputs/manual_unlabeled_review.html
"""
import json, sys, math, time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Import enrichment logic from patch_review_html.py ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from patch_review_html import MCC_MAP, INDUSTRY_MAP, SCORE_DESCRIPTIONS, generate_explanation

RANK_CANDIDATES = [
    'outputs/rank_df_with_anchor_expansion.csv.gz',
    'outputs/rank_df_with_semi_supervision.csv.gz',
]
TXN_CANDIDATES = [
    'outputs/scotiabank_transactions.csv.gz',
    'outputs/master_transaction_pool.csv.gz',
]
TXN_PER_CUSTOMER_CAP = 300
TOP_N_CLUSTERS       = 100
OUT_PATH             = Path('outputs/manual_unlabeled_review.html')
NB_PATH              = Path('explainability_gnnexplainer.ipynb')

SCORE_PRIORITY = [
    'scarcity_anchor_ensemble_prob', 'scarcity_semi_ensemble_prob',
    'scarcity_ensemble_prob', 'suspicion_evidence_score_v3_tuned',
    'suspicion_evidence_score_v3', 'suspicion_evidence_score_v2',
    'suspicion_evidence_score', 'lgb_fraud_prob', 'mlp_fraud_prob',
]

def safe(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v

t0 = time.time()
print("=" * 60)
print("Cluster-diverse selection: top cluster × 1 customer")
print("=" * 60)

# ── 1. Load ranking dataframe ──────────────────────────────────────────────────
rank_path = next((p for p in RANK_CANDIDATES if Path(p).exists()), None)
if not rank_path:
    raise FileNotFoundError("No ranking artifact found.")

print(f"\n[1] Loading {rank_path}...")
rank_df = pd.read_csv(rank_path)
rank_df['customer_id'] = rank_df['customer_id'].astype(str).str.strip()
print(f"    {len(rank_df):,} rows loaded in {time.time()-t0:.1f}s")

if 'true_label' in rank_df.columns:
    rank_df['true_label'] = pd.to_numeric(rank_df['true_label'], errors='coerce')
    unlabeled = rank_df[~rank_df['true_label'].isin([0, 1])].copy()
else:
    unlabeled = rank_df.copy()

score_col = next((c for c in SCORE_PRIORITY if c in rank_df.columns), None)
if not score_col:
    raise KeyError("No score column found.")
unlabeled[score_col] = pd.to_numeric(unlabeled[score_col], errors='coerce').fillna(0.0)

print(f"    Unlabeled: {len(unlabeled):,} | Score column: {score_col}")

# ── 2. Cluster-diverse selection ───────────────────────────────────────────────
print(f"\n[2] Selecting one candidate per graph component (top {TOP_N_CLUSTERS})...")

cluster_stats = (
    unlabeled.groupby('component')[score_col]
    .agg(['mean', 'count'])
    .rename(columns={'mean': 'cluster_mean_score', 'count': 'cluster_size'})
    .sort_values('cluster_mean_score', ascending=False)
    .head(TOP_N_CLUSTERS)
)
top_clusters = cluster_stats.index.tolist()
print(f"    Top {len(top_clusters)} clusters selected (range: "
      f"{cluster_stats['cluster_mean_score'].min():.3f}–"
      f"{cluster_stats['cluster_mean_score'].max():.3f} mean score)")

candidates = []
for comp in top_clusters:
    best = unlabeled[unlabeled['component'] == comp].nlargest(1, score_col)
    candidates.append(best)

candidates = pd.concat(candidates).reset_index(drop=True)
candidates = candidates.sort_values(score_col, ascending=False).reset_index(drop=True)
candidate_ids = set(candidates['customer_id'].tolist())
print(f"    Selected {len(candidates)} customers across {len(top_clusters)} components")

# ── 3. Load transactions ───────────────────────────────────────────────────────
txn_path = next((p for p in TXN_CANDIDATES if Path(p).exists()), None)
if not txn_path:
    raise FileNotFoundError("No transaction file found.")

print(f"\n[3] Loading {txn_path} (this may take ~60s)...")
t1 = time.time()
tx = pd.read_csv(txn_path, low_memory=False)
print(f"    {len(tx):,} rows loaded in {time.time()-t1:.1f}s")

tx['customer_id'] = tx['customer_id'].astype(str).str.strip()
tx = tx[tx['customer_id'].isin(candidate_ids)].copy()
print(f"    Filtered to {len(tx):,} rows for {tx['customer_id'].nunique()} customers")

if 'transaction_datetime' in tx.columns:
    tx['transaction_datetime'] = pd.to_datetime(tx['transaction_datetime'], errors='coerce')
tx['amount_cad'] = pd.to_numeric(tx.get('amount_cad', 0.0), errors='coerce').fillna(0.0)
for col, default in [('cash_indicator', None), ('ecommerce_ind', None),
                     ('merchant_category', None), ('city', None),
                     ('debit_credit', None), ('country', None), ('province', None)]:
    if col not in tx.columns:
        tx[col] = default

tx_by_cid = {cid: grp for cid, grp in tx.groupby('customer_id')}

# ── 4. Model context columns ───────────────────────────────────────────────────
CONTEXT_COLS = [
    'component', 'component_size', 'component_train_fraud_rate', 'component_confidence',
    'hdb_component', 'hdb_component_size', 'hdb_component_fraud_rate_labeled',
    'hdb_outlier_score', 'hdb_is_clustered',
    'cluster_consensus_score', 'dgi_anomaly_score',
    'customer_ae_risk', 'customer_ae_risk_norm',
    'knn_suspicious_share', 'knn_gold_fraud_count',
    'min_dist_to_fraud_anchor', 'mean_dist_to_fraud_anchor',
    'min_dist_to_legit_anchor', 'mean_dist_to_legit_anchor',
    'kmeans_labeled_lift', 'hdb_labeled_lift', 'dist_to_fraud_centroid',
    'dist_to_legit_centroid', 'centroid_margin',
    'anchor_proximity_score', 'review_priority_score',
    'lgb_fraud_prob', 'mlp_fraud_prob', 'scarcity_ensemble_prob',
    'scarcity_semi_ensemble_prob', 'scarcity_anchor_prob', 'scarcity_anchor_ensemble_prob',
    'label_source_anchor', 'semi_label_anchor', 'second_pass_positive',
    'is_train_anchor', 'is_component_core',
    'km_component_train_fraud_rate', 'km_is_fraud_component',
    'lgb_label', 'lgb_weight',
]
PROFILE_COLS = [
    'gmm_max_prob', 'cluster_distance', 'kmeans_label_coverage',
    'hdb_component_label_coverage', 'hdb_component_fraud_lift_labeled',
]

_JUNK = {'nan', 'NaN', 'none', 'None', 'other', 'unknown', '', '0', 'null', 'NULL'}

def _clean_str(v):
    """Return a clean string or None for junk values."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in _JUNK else s

def _top_categories(df, n=5):
    """Return top n merchant category names, filtering junk codes."""
    if df is None or len(df) == 0:
        return []
    vc = df['merchant_category'].dropna().astype(str)
    vc = vc[~vc.isin(_JUNK)]
    names = []
    for code, _ in vc.value_counts().head(n * 2).items():
        name = MCC_MAP.get(code, INDUSTRY_MAP.get(code, ''))
        label = name if name else code
        if label not in names:
            names.append(label)
        if len(names) == n:
            break
    return names

def _top_locations(df, n=5):
    """Return top n real city names, falling back to provinces if needed."""
    if df is None or len(df) == 0:
        return []
    # Cities that are real (not junk)
    cities = df['city'].dropna().astype(str)
    cities = cities[~cities.str.lower().isin({v.lower() for v in _JUNK})]
    city_list = cities.value_counts().head(n).index.tolist()
    if len(city_list) >= n:
        return city_list
    # Supplement with provinces
    if 'province' in df.columns:
        provs = df['province'].dropna().astype(str)
        provs = provs[~provs.isin(_JUNK)]
        for p in provs.value_counts().head(n).index:
            label = f"[{p}]"
            if label not in city_list and len(city_list) < n:
                city_list.append(label)
    return city_list

# ── 5. Build customer rows ─────────────────────────────────────────────────────

def _cluster_behavior_summary(component_id, cands_df, tx_lookup):
    """Return a short plain-English behavioral summary string for a graph component.
    
    Describes what accounts in the cluster actually have in common:
    average spend, timing, top merchant categories, and cash usage.
    """
    member_ids = cands_df[cands_df['component'] == component_id]['customer_id'].astype(str).tolist()
    frames = [tx_lookup[cid] for cid in member_ids if cid in tx_lookup]
    if not frames:
        return ''
    ct = pd.concat(frames, ignore_index=True)
    ct['amount_cad'] = pd.to_numeric(ct['amount_cad'], errors='coerce').fillna(0)
    parts = []
    mean_amt = ct['amount_cad'].mean()
    if mean_amt > 0:
        parts.append(f"average transaction of ${mean_amt:,.0f}")
    if 'transaction_datetime' in ct.columns:
        hr = pd.to_datetime(ct['transaction_datetime'], errors='coerce').dt.hour.dropna()
        if len(hr) > 0:
            night_pct = ((hr <= 5) | (hr >= 23)).mean() * 100
            if night_pct >= 20:
                parts.append(f"{night_pct:.0f}% late-night transactions (11 pm – 6 am)")
    if 'merchant_category' in ct.columns:
        top_codes = ct['merchant_category'].dropna().astype(str).value_counts().head(3).index.tolist()
        top_names = [MCC_MAP.get(c, INDUSTRY_MAP.get(c, '')) or c for c in top_codes]
        top_names = [n for n in top_names if n and n not in _JUNK][:2]
        if top_names:
            parts.append(f"top spend categories: {', '.join(top_names)}")
    if 'cash_indicator' in ct.columns:
        cash_pct = pd.to_numeric(ct['cash_indicator'], errors='coerce').fillna(0).mean() * 100
        if cash_pct >= 30:
            parts.append(f"{cash_pct:.0f}% cash transactions")
    if 'ecommerce_ind' in ct.columns:
        ecom_pct = pd.to_numeric(ct['ecommerce_ind'], errors='coerce').fillna(0).mean() * 100
        if ecom_pct >= 50:
            parts.append(f"majority ({ecom_pct:.0f}%) e-commerce purchases")
    return '; '.join(parts)

# Pre-compute cluster behavioural summaries for each component
print(f"\n[4a] Pre-computing cluster behavioural summaries...")
cluster_behavior_summaries: dict = {}
for comp_id in candidates['component'].dropna().unique():
    cluster_behavior_summaries[int(comp_id)] = _cluster_behavior_summary(
        int(comp_id), candidates, tx_by_cid
    )
print(f"     Computed summaries for {len(cluster_behavior_summaries)} components.")

print(f"\n[4] Building customer payloads...")
rows = []
for i, (_, row) in enumerate(candidates.iterrows()):
    cid = row['customer_id']
    txns_df = tx_by_cid.get(cid)

    profile = {
        'txn_count':         int(len(txns_df)) if txns_df is not None else 0,
        'mean_amount':       float(txns_df['amount_cad'].mean()) if txns_df is not None and len(txns_df) > 0 else 0.0,
        'max_amount':        float(txns_df['amount_cad'].max())  if txns_df is not None and len(txns_df) > 0 else 0.0,
        'high_value_count':  int((txns_df['amount_cad'] > 1000).sum()) if txns_df is not None else 0,
        'first_txn':         str(txns_df['transaction_datetime'].min()) if txns_df is not None and 'transaction_datetime' in txns_df else '',
        'last_txn':          str(txns_df['transaction_datetime'].max()) if txns_df is not None and 'transaction_datetime' in txns_df else '',
        'top_categories':    _top_categories(txns_df),
        'top_cities':        _top_locations(txns_df),
        'cash_ratio':        float((txns_df['cash_indicator'] == 1).sum() / max(len(txns_df), 1)) if txns_df is not None else 0.0,
        'ecom_ratio':        float((txns_df['ecommerce_ind'] == 1).sum() / max(len(txns_df), 1)) if txns_df is not None else 0.0,
    }
    for pc in PROFILE_COLS:
        if pc in row.index:
            profile[pc] = safe(row[pc])

    model_context = {c: safe(row.get(c)) for c in CONTEXT_COLS if c in row.index}

    # Build compact transaction list
    txn_list = []
    if txns_df is not None:
        tsorted = txns_df.sort_values('transaction_datetime', ascending=False).head(TXN_PER_CUSTOMER_CAP)
        for _, t in tsorted.iterrows():
            dt = t['transaction_datetime']
            code = str(t.get('merchant_category', '') or '')
            merchant_name = MCC_MAP.get(code, INDUSTRY_MAP.get(code, '')) if code not in ('nan', 'unknown', '', '0') else ''
            txn_list.append({
                'dt':               str(dt) if pd.notna(dt) else '',
                'amount':           safe(t['amount_cad']),
                'merchant_category': code if code not in ('nan', 'NaN', 'unknown', '', '0', 'other') else '',
                'merchant_name':    merchant_name,
                'debit_credit':     str(t.get('debit_credit') or '') or None,
                'city':             _clean_str(t.get('city')),
                'province':         _clean_str(t.get('province')),
                'country':          _clean_str(t.get('country')),
                'cash_indicator':   safe(t.get('cash_indicator')),
                'ecommerce_ind':    safe(t.get('ecommerce_ind')),
            })

    customer = {
        'customer_id':      cid,
        'primary_score_col': score_col,
        'primary_score':    safe(row[score_col]),
        'rank':             i + 1,
        'cluster_rank':     int(top_clusters.index(int(row['component'])) + 1) if int(row['component']) in [int(c) for c in top_clusters] else None,
        'profile':          profile,
        'model_context':    model_context,
        'transactions':     txn_list,
        'cluster_behavior': cluster_behavior_summaries.get(int(row.get('component', -1)), ''),
    }
    customer['explanation'] = generate_explanation(customer)
    rows.append(customer)

    if (i + 1) % 20 == 0:
        print(f"    Built {i+1}/{len(candidates)}")

print(f"    Done building {len(rows)} customer records in {time.time()-t0:.1f}s")

# ── 6. Assemble payload ────────────────────────────────────────────────────────
payload = {
    'generated_at':    datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
    'rank_source':     rank_path,
    'txn_source':      txn_path,
    'score_col':       score_col,
    'top_n':           len(rows),
    'selection_method': f'cluster_diverse_top{TOP_N_CLUSTERS}',
    'customers':       rows,
    'score_descriptions': SCORE_DESCRIPTIONS,
    'mcc_map':         MCC_MAP,
    'industry_map':    INDUSTRY_MAP,
}

# ── 7. Load HTML template and inject ──────────────────────────────────────────
print(f"\n[5] Loading HTML template from notebook...")
nb  = json.load(NB_PATH.open(encoding='utf-8'))
src = ''.join(nb['cells'][40]['source'])
t_s = src.find("html_template = '''") + len("html_template = '''")
t_e = src.find("'''", t_s)
template = src[t_s:t_e]
assert '__DATA_JSON__' in template

enriched_json = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
html = template.replace('__DATA_JSON__', enriched_json)
print(f"    HTML after injection: {len(html):,} chars")

# ── 8. Apply UI patches ────────────────────────────────────────────────────────
print(f"\n[6] Applying UI patches...")

# 8a. Inject JS helper functions right after `let labels = {};`
# These handle type inference and location fallback in the browser
JS_HELPERS = """
function txnType(t) {
  if (t.cash_indicator == 1) return t.debit_credit === 'C' ? 'Cash Deposit' : 'ATM Withdrawal';
  if (t.cash_indicator == 0) return t.ecommerce_ind == 1 ? 'eCommerce' : 'POS / Card';
  if (t.merchant_category) return 'POS Purchase';
  if (t.debit_credit === 'C') return 'Credit / Refund';
  if (t.debit_credit === 'D') return 'Debit Transfer';
  return 'Transfer';
}
function txnLocation(t) {
  const junk = new Set(['nan','NaN','other','unknown','','null','None']);
  const city = t.city && !junk.has(t.city) ? t.city : null;
  if (city) {
    const prov = t.province && !junk.has(t.province) ? t.province : null;
    return prov ? city + ', ' + prov : city;
  }
  const prov = t.province && !junk.has(t.province) ? t.province : null;
  const ctry = t.country && !junk.has(t.country) && t.country !== 'CA' ? t.country : null;
  if (prov && ctry) return prov + ' (' + ctry + ')';
  if (prov) return prov;
  if (ctry) return ctry;
  return '';
}
function txnDC(t) {
  if (t.debit_credit === 'D') return '<span style="color:#dc2626;font-size:10px;font-weight:700;">D</span>';
  if (t.debit_credit === 'C') return '<span style="color:#16a34a;font-size:10px;font-weight:700;">C</span>';
  return '';
}
"""
p_helpers_old = "let labels = {};"
p_helpers_new = "let labels = {};" + JS_HELPERS
if p_helpers_old in html:
    html = html.replace(p_helpers_old, p_helpers_new, 1); print("    ✓ JS helper functions")
else:
    print("    ✗ helpers anchor not found")

# 8b. Transaction table header: type + merchant + location columns
p_hdr_old = '<th>datetime</th><th>amount_cad</th><th>merchant_category</th><th>city</th><th>cash</th><th>ecom</th>'
p_hdr_new = '<th>datetime</th><th>DC</th><th>amount_cad</th><th>type</th><th>merchant</th><th>location</th>'
if p_hdr_old in html:
    html = html.replace(p_hdr_old, p_hdr_new); print("    ✓ table header")
else:
    print("    ✗ table header not found")

# 8c. Transaction row: use helper functions, no raw nan values
p_row_old = ("`\n    <tr>\n      <td>${t.dt || ''}</td>\n      <td>${fmt(Number(t.amount || 0),2)}</td>\n"
             "      <td>${t.merchant_category || ''}</td>\n      <td>${t.city || ''}</td>\n"
             "      <td>${t.cash_indicator ?? ''}</td>\n      <td>${t.ecommerce_ind ?? ''}</td>\n"
             "    </tr>\n  `).join('');")
p_row_new = ("`\n    <tr>\n"
             "      <td style='white-space:nowrap'>${t.dt ? t.dt.slice(0,16) : ''}</td>\n"
             "      <td style='text-align:center'>${txnDC(t)}</td>\n"
             "      <td style='text-align:right'>${fmt(Number(t.amount || 0),2)}</td>\n"
             "      <td class='muted' style='white-space:nowrap'>${txnType(t)}</td>\n"
             "      <td>${t.merchant_name || (t.merchant_category && t.merchant_category !== 'nan' ? t.merchant_category : '')}</td>\n"
             "      <td class='muted'>${txnLocation(t)}</td>\n"
             "    </tr>\n  `).join('');")
if p_row_old in html:
    html = html.replace(p_row_old, p_row_new); print("    ✓ transaction row")
else:
    print("    ✗ transaction row not found")

# 8d. Fix top_categories pills: values are now names, not raw codes
p_cats_old = ("const cats = (p.top_categories || []).map(x => `<span class=\"pill\">${x}</span>`).join('');\n"
              "  const cities = (p.top_cities || []).map(x => `<span class=\"pill\">${x}</span>`).join('');")
p_cats_new = ("const cats = (p.top_categories || []).filter(x => x && x !== 'nan' && x !== 'other')\n"
              "    .map(x => `<span class=\"pill\">${x}</span>`).join('');\n"
              "  const cities = (p.top_cities || []).filter(x => x && x !== 'nan' && x !== 'other')\n"
              "    .map(x => `<span class=\"pill\">${x.startsWith('[') ? x.slice(1,-1)+' (province)' : x}</span>`).join('');")
if p_cats_old in html:
    html = html.replace(p_cats_old, p_cats_new); print("    ✓ category + city pills")
else:
    print("    ✗ category/city pills not found (trying alternate)...")
    # Try single-quote variant
    p_cats_old2 = ("const cats = (p.top_categories || []).map(x => `<span class=\\'pill\\'>${x}</span>`).join(\\'\\');\n"
                   "  const cities = (p.top_cities || []).map(x => `<span class=\\'pill\\'>${x}</span>`).join(\\'\\');")
    idx_cats = html.find("(p.top_categories || []).map(x =")
    if idx_cats >= 0:
        print(f"    Found top_categories at {idx_cats}: {repr(html[idx_cats:idx_cats+120])}")

# 8e. Replace raw card with explanation card + collapsible raw + glossary
old_card = """      <div class="card">
        <h3>Full Model / Profile Context</h3>
        <pre id="raw"></pre>
      </div>"""
new_card = """      <div class="card" style="border-left:4px solid #0b5fff;margin-bottom:10px;">
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
if old_card in html:
    html = html.replace(old_card, new_card); print("    ✓ explanation card")
else:
    print("    ✗ explanation card not found")

# 8f. JS: render explanation text + populate glossary
old_js = ("};\n  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);\n\n"
          "  loadLabel(c.customer_id);\n}")
new_js = ("};\n  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);\n\n"
          "  const expEl = document.getElementById('explanationText');\n"
          "  if (expEl) {\n"
          "    const lines = (c.explanation || '(No explanation generated.)').split('. ');\n"
          "    expEl.innerHTML = lines.map(l => l.trim() ? `<p style=\"margin:0 0 8px 0;\">${l.trim()}${l.trim().endsWith('.') ? '' : '.'}</p>` : '').join('');\n"
          "  }\n"
          "  const glossaryBody = document.getElementById('scoreGlossaryBody');\n"
          "  if (glossaryBody && glossaryBody.childElementCount === 0 && DATA.score_descriptions) {\n"
          "    glossaryBody.innerHTML = Object.entries(DATA.score_descriptions)\n"
          "      .map(([k, v]) => `<tr><td style=\"font-weight:600;white-space:nowrap;padding:5px 8px;font-size:11px;vertical-align:top;\">${k}</td><td style=\"padding:5px 8px;font-size:11px;color:#667085;\">${v}</td></tr>`)\n"
          "      .join('');\n"
          "  }\n\n"
          "  loadLabel(c.customer_id);\n}")
if old_js in html:
    html = html.replace(old_js, new_js); print("    ✓ explanation + glossary JS")
else:
    print("    ✗ explanation JS not found (trying alternate)...")
    alt_old = ("  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);\n\n"
               "  loadLabel(c.customer_id);")
    alt_new = ("  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);\n\n"
               "  const expEl = document.getElementById('explanationText');\n"
               "  if (expEl) {\n"
               "    const lines = (c.explanation || '(No explanation generated.)').split('. ');\n"
               "    expEl.innerHTML = lines.map(l => l.trim() ? `<p style=\"margin:0 0 8px 0;\">${l.trim()}${l.trim().endsWith('.') ? '' : '.'}</p>` : '').join('');\n"
               "  }\n"
               "  const glossaryBody = document.getElementById('scoreGlossaryBody');\n"
               "  if (glossaryBody && glossaryBody.childElementCount === 0 && DATA.score_descriptions) {\n"
               "    glossaryBody.innerHTML = Object.entries(DATA.score_descriptions)\n"
               "      .map(([k, v]) => `<tr><td style=\"font-weight:600;white-space:nowrap;padding:5px 8px;font-size:11px;vertical-align:top;\">${k}</td><td style=\"padding:5px 8px;font-size:11px;color:#667085;\">${v}</td></tr>`)\n"
               "      .join('');\n"
               "  }\n\n"
               "  loadLabel(c.customer_id);")
    if alt_old in html:
        html = html.replace(alt_old, alt_new); print("    ✓ alternate JS patch applied")
    else:
        print("    ✗ alternate also not found — explanation panel won't render")

# 8g. List row: score description tooltip + cluster rank badge
old_list = ("""    div.innerHTML = `\n"""
            """      <div><b>${c.customer_id}</b></div>\n"""
            """      <div class="muted">${c.primary_score_col}: <span class="score">${fmt(c.primary_score, 5)}</span> | txns: ${c.profile.txn_count}</div>\n"""
            """      <div class="muted">component: ${c.model_context.component ?? ''} | hdb: ${c.model_context.hdb_component ?? ''}</div>\n"""
            """      <div class="muted">label: ${lab || 'unlabeled'} ${conf ? `(conf ${conf})` : ''}</div>\n"""
            """    `;\n""")
new_list = ("""    const scoreDesc = DATA.score_descriptions ? (DATA.score_descriptions[c.primary_score_col] || '') : '';\n"""
            """    div.innerHTML = `\n"""
            """      <div><b>${c.customer_id}</b> ${c.cluster_rank ? `<span style="font-size:10px;background:#e0e7ff;color:#3730a3;border-radius:4px;padding:1px 5px;">cluster #${c.cluster_rank}</span>` : ''}</div>\n"""
            """      <div class="muted" title="${scoreDesc.replace(/"/g, '&quot;')}">${c.primary_score_col}: <span class="score">${fmt(c.primary_score, 5)}</span> | txns: ${c.profile.txn_count}</div>\n"""
            """      <div class="muted">component: ${c.model_context.component ?? ''} | hdb: ${c.model_context.hdb_component ?? ''}</div>\n"""
            """      <div class="muted">label: ${lab || 'unlabeled'} ${conf ? `(conf ${conf})` : ''}</div>\n"""
            """    `;\n""")
if old_list in html:
    html = html.replace(old_list, new_list); print("    ✓ list row with cluster rank badge")
else:
    print("    ✗ list row not found")


# ── 9. Final checks and write ──────────────────────────────────────────────────
print(f"\n[7] Verifying and writing...")
assert '<!doctype html>' in html
assert 'const DATA = ' in html
assert '<script>' in html
assert 'explanationText' in html
assert 'scoreGlossaryBody' in html
# Sanity: parse DATA block
ds = html.find('const DATA = ') + 13
de = html.find(';\nconst LS_KEY', ds)
check = json.loads(html[ds:de])
assert len(check['customers']) == len(rows)
print(f"    DATA block parses OK: {len(check['customers'])} customers")

OUT_PATH.write_text(html, encoding='utf-8')
print(f"\n✓ Written: {OUT_PATH}  ({len(html):,} bytes, {time.time()-t0:.1f}s total)")
print(f"  {len(rows)} customers from {len(top_clusters)} distinct graph components")
print(f"  Score range: {candidates[score_col].min():.4f}–{candidates[score_col].max():.4f}")
