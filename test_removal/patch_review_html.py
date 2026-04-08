"""
Patches manual_unlabeled_review.html with:
  1. Score descriptions (glossary) for all model_context columns
  2. MCC merchant category name mapping
  3. Behaviorally-grounded per-customer explanations
  4. Updated JS rendering: explanation card, merchant names, score glossary
"""

import json, re, math, pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HTML_PATH = Path("outputs/manual_unlabeled_review.html")

# ── MCC code → merchant name mapping ─────────────────────────────────────────
MCC_MAP = {
    "2741": "Publishing / Newspapers",
    "4111": "Local Transit / Commuter Rail",
    "4121": "Taxi / Limousines",
    "4215": "Courier / Delivery Services",
    "4225": "Storage / Warehousing",
    "4722": "Travel Agencies",
    "4789": "Transportation Services",
    "4812": "Telecom Equipment Sales",
    "4814": "Telephone Services",
    "4816": "Internet / Computer Network Services",
    "4829": "Money Transfer / Wire",
    "4899": "Cable / Satellite TV",
    "4900": "Utilities",
    "5039": "Construction Materials",
    "5045": "Computers / Peripherals",
    "5065": "Electronic Parts & Equipment",
    "5074": "Plumbing / Heating Supplies",
    "5085": "Industrial Supplies",
    "5199": "Nondurable Goods",
    "5200": "Home Supply / Garden Stores",
    "5211": "Lumber / Building Materials",
    "5231": "Glass / Paint / Wallpaper",
    "5251": "Hardware Stores",
    "5261": "Nurseries / Lawn & Garden",
    "5300": "Wholesale Clubs (e.g. Costco)",
    "5310": "Discount Stores",
    "5311": "Department Stores",
    "5331": "Variety / Dollar Stores",
    "5411": "Grocery Stores / Supermarkets",
    "5441": "Candy / Confectionery",
    "5462": "Bakeries",
    "5499": "Convenience Stores / Misc. Food",
    "5511": "Auto Dealers (New)",
    "5532": "Auto Parts Stores",
    "5533": "Auto Parts & Accessories",
    "5541": "Gas Stations",
    "5542": "Gas Stations (Automated)",
    "5621": "Women's Clothing Stores",
    "5631": "Women's Accessories Stores",
    "5651": "Family Clothing Stores",
    "5655": "Sports Apparel",
    "5661": "Shoe Stores",
    "5681": "Furriers / Fur Shops",
    "5691": "Men's Clothing Stores",
    "5699": "Clothing & Apparel (Misc.)",
    "5712": "Furniture Stores",
    "5713": "Floor Covering Stores",
    "5732": "Electronics / Appliance Stores",
    "5733": "Music Stores",
    "5734": "Computer / Software Stores",
    "5735": "Record / CD Stores",
    "5812": "Restaurants / Eating Places",
    "5814": "Fast Food Restaurants",
    "5815": "Digital Goods — Media",
    "5816": "Digital Goods — Games",
    "5817": "Digital Goods — Apps",
    "5818": "Digital Goods — Other",
    "5912": "Drug Stores / Pharmacies",
    "5921": "Beer / Wine / Liquor Stores",
    "5931": "Used Merchandise Stores",
    "5941": "Sporting Goods Stores",
    "5942": "Book Stores",
    "5943": "Stationery / Office Supplies",
    "5944": "Jewellery / Watch Stores",
    "5945": "Hobby / Toy / Game Shops",
    "5947": "Gift / Card / Novelty Stores",
    "5960": "Direct Marketing — Insurance",
    "5967": "Direct Marketing — Telemarketing",
    "5968": "Direct Marketing — Subscriptions",
    "5977": "Cosmetics / Beauty Supply",
    "5993": "Cigar / Tobacco Stores",
    "5994": "News Dealers",
    "5995": "Pet Shops",
    "5999": "Miscellaneous Retail",
    "6010": "Cash / Member Financial Institution",
    "6011": "ATM Cash Disbursements",
    "6012": "Financial Institutions — Merchandise",
    "6051": "Non-Financial Institutions — Forex / Currency Exchange",
    "6211": "Security Brokers / Dealers",
    "6300": "Insurance Services",
    "7011": "Hotels / Motels / Lodging",
    "7221": "Photography Studios / Stores",
    "7230": "Barber / Beauty Salons",
    "7298": "Health & Beauty Spas",
    "7299": "Services — Not Elsewhere Classified",
    "7311": "Advertising Services",
    "7372": "Computer Software / Programming",
    "7392": "Management Consulting / Public Relations",
    "7399": "Business Services (Misc.)",
    "7523": "Parking Lots / Garages",
    "7531": "Auto Body Repair Shops",
    "7538": "Auto Service & Repair",
    "7541": "Car Rental",
    "7542": "Car Washes",
    "7832": "Motion Picture Theatres",
    "7994": "Video Game Arcades",
    "7995": "Gambling / Casinos / Betting",
    "7996": "Amusement Parks / Attractions",
    "8011": "Doctors / Physicians",
    "8021": "Dentists / Orthodontists",
    "8031": "Osteopathic Physicians",
    "8041": "Chiropractors",
    "8042": "Optometrists",
    "8099": "Medical Services (Misc.)",
    "8211": "Elementary / Secondary Schools",
    "8398": "Charitable & Social Service Orgs",
    "8699": "Membership Organizations",
    "8999": "Professional Services",
    "9399": "Government Services",
}

# ── Score descriptions ────────────────────────────────────────────────────────
SCORE_DESCRIPTIONS = {
    "scarcity_anchor_ensemble_prob": "Final fraud risk score. Weighted ensemble of the anchor-expansion Random Forest (45%), semi-supervised ensemble (25%), scarcity ensemble (15%), and cluster reliability (15%). Primary ranking signal.",
    "scarcity_anchor_prob": "Random Forest trained on customers confirmed or pseudo-labeled as fraud via proximity to known fraud anchors in embedding space.",
    "scarcity_semi_ensemble_prob": "Ensemble combining the semi-supervised RF (55%), scarcity score (25%), PU bagging (10%), and cluster consensus (10%). Predecessor to final score.",
    "scarcity_semi_prob": "Semi-supervised Random Forest score. Trained on gold labels + high-confidence pseudo-positives and pseudo-negatives. Uses all meta-features including cluster, kNN, and AE signals.",
    "scarcity_ensemble_prob": "Earlier-stage ensemble: 55% PU bagging + 30% base model score + 15% review priority.",
    "pu_bagging_prob": "Positive-Unlabeled (PU) bagging ensemble (12 bags). Treats all unlabeled customers as potential negatives while training on confirmed positives. Formally conservative upper-bound on fraud probability.",
    "lgb_fraud_prob": "LightGBM ranker trained on gold labels and high-confidence pseudo-positives. Uses 600 trees with embedding PCA, cluster statistics, kNN signals, and centroid distances.",
    "mlp_fraud_prob": "MLP score from a 2-layer neural network trained on 64-dimensional DGI embeddings + 3 auxiliary features (GMM prob, AE risk, component confidence).",
    "combined_model_score": "Intermediate combination of LightGBM and MLP scores before PU reranking.",
    "review_priority_score": "Pre-PU review priority: 0.35×model + 0.25×consensus + 0.15×DGI + 0.15×AE + 0.10×anchor proximity. Used to select pseudo-positive candidates.",
    "review_priority_v2": "v2 review priority incorporating additional semi-supervised signals.",
    "anchor_proximity_score": "How close this customer's embedding is to the nearest confirmed fraud anchor (inversely scaled). Higher = closer to a known fraud case.",
    "cluster_consensus_score": "Consensus (1–4) between KMeans and HDBSCAN clustering views. 4 = both agree this cluster is high-risk.",
    "kmeans_labeled_lift": "Fraud lift in this KMeans cluster vs. the dataset baseline. Higher = the cluster has more confirmed fraud than expected.",
    "hdb_labeled_lift": "Fraud lift in the HDBSCAN cluster. 0 if not assigned to any cluster (noise point).",
    "cluster_reliability_gate": "Gate score (0–1) combining KMeans lift, HDBSCAN lift, consensus score, and suspicious-slice membership. Used as a hard filter in anchor-expansion.",
    "component_train_fraud_rate": "Fraction of labeled customers in this KMeans component who are confirmed fraud. Direct measure of cluster toxicity.",
    "component_confidence": "Confidence in the component's fraud label, based on the number of labeled examples and their consistency.",
    "hdb_outlier_score": "HDBSCAN outlier score (0–1). Close to 1 = embedding space outlier, no stable cluster found.",
    "dgi_anomaly_score": "Deep Graph Infomax anomaly score. Measures how poorly the graph encoder can reconstruct this customer from its neighborhood—high = unusual graph topology.",
    "customer_ae_risk": "Transaction autoencoder reconstruction error. High error means this customer's transaction sequence differs from learned normal patterns.",
    "customer_ae_risk_norm": "Normalized AE risk (0–1 percentile rank across all customers).",
    "gmm_max_prob": "Maximum posterior probability across all Gaussian Mixture Model components. Low value = the customer sits in a low-density embedding region.",
    "dist_to_fraud_centroid": "Euclidean distance from this customer's embedding to the fraud cluster centroid (lower = closer to fraud).",
    "dist_to_legit_centroid": "Euclidean distance from this customer's embedding to the legitimate cluster centroid.",
    "centroid_margin": "Margin = dist_to_legit_centroid − dist_to_fraud_centroid. Positive = closer to fraud centroid.",
    "knn_suspicious_share": "Fraction of 15-nearest embedding neighbours that are in the suspicious slice (high AE or DGI risk).",
    "knn_gold_fraud_count": "Number of confirmed-fraud customers among the 15 nearest neighbours in embedding space.",
    "min_dist_to_fraud_anchor": "Distance to the single nearest confirmed fraud anchor. Lower = strongly co-located with a known fraud case.",
    "mean_dist_to_fraud_anchor": "Average distance to all confirmed fraud anchors.",
    "min_dist_to_legit_anchor": "Distance to the single nearest confirmed legitimate anchor.",
    "component_size": "Number of customers in this KMeans cluster.",
    "km_component_train_fraud_rate": "KMeans component fraud rate among labeled customers.",
    "km_is_fraud_component": "1 if this KMeans component's fraud lift exceeds 3× the dataset baseline.",
    "semi_label_v2": "Pseudo-label assigned in semi-supervised stage v2 (1.0 = pseudo-positive, 0 = pseudo-negative, null = unlabeled).",
    "semi_label_anchor": "Pseudo-label assigned by anchor-expansion stage (1.0 = pseudo-positive flagged near a fraud anchor).",
    "label_source_v2": "Source of the v2 pseudo-label: 'pseudo_positive', 'pseudo_negative', or 'unlabeled'.",
    "label_source_anchor": "Source of the anchor pseudo-label: 'pseudo_positive' = flagged by anchor proximity, 'unlabeled' = not flagged.",
    "anchor_neighbor_positive": "1 if this customer was explicitly selected as a fraud-anchor neighbour during anchor-expansion (within top-25 neighbors of any confirmed fraud anchor).",
    "true_label": "Ground-truth label: 1.0 = confirmed fraud, 0.0 = confirmed legitimate, −1.0 = unlabeled.",
    "final_train_label": "Label used for supervised training after pseudo-label assignment (0 = not used as positive, 1 = used as positive).",
    "is_train_anchor": "1 if this customer is a confirmed fraud in the training split (gold fraud anchor).",
    "second_pass_positive": "1 if this customer received a positive label in the cluster second-pass propagation stage.",
    "pseudo_review_positive": "1 if this customer was promoted to pseudo-positive during PU review-priority ranking.",
    "is_component_core": "1 if this customer is in the core of its KMeans component (near the centroid).",
    "is_suspicious_slice": "1 if this customer was included in the suspicious slice (top-2% DGI anomaly or top-5% AE risk).",
}

# ── Load industry codes from data/ ────────────────────────────────────────────
try:
    ind_df = pd.read_csv("data/kyc_industry_codes.csv.gz")
    INDUSTRY_MAP = dict(zip(ind_df['industry_code'].astype(str), ind_df['industry']))
except Exception as e:
    INDUSTRY_MAP = {}
    print(f"Warning: could not load industry codes: {e}")

# ── Tipping point detection ───────────────────────────────────────────────────
def detect_tipping_point(transactions, window=6):
    """Return (date_str, ratio) or None.  Looks for a ≥4× jump in rolling mean amount.
    Sorts transactions by date ascending internally so caller order does not matter."""
    valid = [(t.get('dt', ''), t.get('amount') or 0) for t in transactions if t.get('dt')]
    valid.sort(key=lambda x: x[0])
    if len(valid) < window * 2:
        return None
    dts     = [v[0] for v in valid]
    amounts = [v[1] for v in valid]
    for i in range(window, len(amounts)):
        pre  = amounts[max(0, i - window * 2):i - window]
        post = amounts[i - window:i]
        if not pre or not post:
            continue
        pre_mean  = sum(pre)  / len(pre)
        post_mean = sum(post) / len(post)
        if pre_mean > 50.0 and post_mean / pre_mean >= 4.0:
            return (dts[i - window], round(post_mean / pre_mean, 1))
    return None

# ── Explanation generator ─────────────────────────────────────────────────────
def _pct_label(v, lo=25, hi=75):
    """Label a 0-1 value as low/moderate/high."""
    if v is None:
        return "unknown"
    v = float(v)
    if v <= lo / 100:
        return "low"
    if v >= hi / 100:
        return "elevated"
    return "moderate"

def generate_explanation(customer: dict) -> str:
    profile = customer.get('profile', {})
    mc      = customer.get('model_context', {})
    txns    = customer.get('transactions', [])

    # ── Transaction behaviour ──────────────────────────────────────────────
    txn_count    = int(profile.get('txn_count') or 0)
    mean_amt     = float(profile.get('mean_amount') or 0)
    max_amt      = float(profile.get('max_amount') or 0)
    high_val     = int(profile.get('high_value_count') or 0)
    night_ratio  = float(profile.get('night_ratio') or 0)
    cash_ratio   = float(profile.get('cash_ratio') or 0)
    ecom_ratio   = float(profile.get('ecom_ratio') or 0)
    top_cats     = [c for c in (profile.get('top_categories') or []) if c not in ('nan','other','0','')]
    top_cities   = [c for c in (profile.get('top_cities')     or []) if c not in ('nan','other','')]

    score        = float(mc.get('scarcity_anchor_ensemble_prob') or 0)
    ae_risk      = float(mc.get('customer_ae_risk_norm') or 0)
    dgi_anom     = float(mc.get('dgi_anomaly_score') or 0)
    comp_fraud_r = float(mc.get('component_train_fraud_rate') or 0)
    comp_size    = int(float(mc.get('component_size') or 0))
    km_fraud     = str(mc.get('km_is_fraud_component', '0'))
    hdb_outlier  = float(mc.get('hdb_outlier_score') or 0)
    consensus    = float(mc.get('cluster_consensus_score') or 0)
    min_anch_d   = float(mc.get('min_dist_to_fraud_anchor') or 99)
    knn_susp     = float(mc.get('knn_suspicious_share') or 0)
    knn_fraud    = float(mc.get('knn_gold_fraud_count') or 0)
    label_src    = str(mc.get('label_source_anchor') or 'unlabeled')
    centroid_m   = float(mc.get('centroid_margin') or 0)

    # ── Tipping point ──────────────────────────────────────────────────────
    tp = detect_tipping_point(txns)

    sentences = []

    # --- S1: Transaction behaviour ---
    if txn_count == 0:
        s1 = "No transaction history is available for this customer."
    else:
        # Build behavioural description
        parts = []
        if txn_count < 5:
            parts.append(f"only {txn_count} transactions on record (sparse history; limited behavioral signal)")
        elif txn_count >= 50:
            parts.append(f"{txn_count} transactions on record—unusually high volume for the review period")
        else:
            parts.append(f"{txn_count} transactions on record")

        if mean_amt > 5000:
            parts.append(f"average transaction of ${mean_amt:,.0f} far above typical retail spend")
        elif mean_amt > 1000:
            parts.append(f"average transaction of ${mean_amt:,.0f}, skewed toward large purchases")
        elif mean_amt > 0:
            parts.append(f"average transaction of ${mean_amt:,.0f}")

        if max_amt > 50000:
            parts.append(f"peak transaction of ${max_amt:,.0f} (extreme outlier)")
        elif max_amt > 10000:
            parts.append(f"largest single transaction of ${max_amt:,.0f}")

        if high_val > 0 and txn_count > 0:
            hv_pct = round(100 * high_val / txn_count)
            if hv_pct >= 30:
                parts.append(f"{hv_pct}% of transactions exceed $1,000 (high-value concentration)")

        extras = []
        if cash_ratio >= 0.20:
            extras.append(f"cash withdrawals make up {round(100*cash_ratio)}% of transactions")
        if ecom_ratio >= 0.5:
            extras.append(f"majority ({round(100*ecom_ratio)}%) of transactions are e-commerce")
        if night_ratio >= 0.25:
            extras.append(f"{round(100*night_ratio)}% of transactions occur between 22:00–06:00")
        if top_cats:
            cat_names = [MCC_MAP.get(c, c) for c in top_cats[:3]]
            extras.append(f"top merchant categories: {', '.join(cat_names)}")
        if top_cities:
            extras.append(f"primarily transacts in: {', '.join(top_cities[:3])}")

        body = '; '.join(parts)
        if extras:
            body += '. ' + '; '.join(extras).capitalize() + '.'
        s1 = f"Transaction behaviour: {body}."
    sentences.append(s1)

    # --- S2: Tipping point ---
    if tp:
        date_str, ratio = tp
        s2 = (f"Around {date_str}, transaction amounts rose sharply — the rolling average "
              f"increased {ratio}× compared to the preceding weeks, which may indicate "
              f"a change in how the account was being used.")
    else:
        if txn_count >= 5:
            s2 = ("No single tipping-point was detected; risk signals appear distributed "
                  "across the full transaction history rather than concentrated at a single event.")
        else:
            s2 = "Insufficient transaction history to assess behavioural tipping points."
    sentences.append(s2)

    # --- S3: Graph / cluster signals ---
    cluster_parts = []
    if comp_fraud_r > 0:
        cluster_parts.append(
            f"is grouped with {comp_size - 1} other accounts under review, "
            f"{round(100*comp_fraud_r)}% of which have been confirmed as fraud"
        )
    if consensus >= 3:
        cluster_parts.append(
            "multiple independent risk assessments consistently flag this account as high risk"
        )
    if ae_risk >= 0.8:
        cluster_parts.append(
            f"transaction patterns are highly unusual compared to typical customers "
            f"(ranked in the top {round(100*(1-ae_risk))}% most abnormal)"
        )
    elif ae_risk >= 0.5:
        cluster_parts.append(
            f"transaction patterns are moderately unusual (top {round(100*(1-ae_risk))}%)"
        )
    if min_anch_d < 3.0:
        cluster_parts.append(
            "financial behaviour closely matches that of confirmed fraud cases"
        )
    elif min_anch_d < 5.0:
        cluster_parts.append(
            "financial behaviour is close to patterns seen in confirmed fraud cases"
        )
    if knn_fraud >= 1:
        cluster_parts.append(
            f"{int(knn_fraud)} of the 15 most financially similar accounts in the dataset "
            f"are confirmed fraud cases"
        )
    elif knn_susp >= 0.4:
        cluster_parts.append(
            f"{round(100*knn_susp)}% of the 15 most financially similar accounts in the "
            f"dataset are flagged as high risk"
        )
    if hdb_outlier >= 0.7:
        cluster_parts.append(
            "transaction patterns do not fit any typical customer profile — "
            "they are statistically unique in the dataset"
        )
    if label_src == 'pseudo_positive':
        cluster_parts.append(
            "selected for review because their financial profile closely matches a "
            "combination of confirmed fraud cases across multiple models"
        )

    cluster_behavior = customer.get('cluster_behavior', '')

    if cluster_parts:
        s3 = "Risk indicators: this account " + '; and '.join(cluster_parts) + '.'
        if cluster_behavior:
            s3 += f" Accounts in this peer group typically show: {cluster_behavior}."
    else:
        s3 = "Risk indicators are weak or inconclusive for this account."
        if cluster_behavior:
            s3 += f" Accounts in this peer group typically show: {cluster_behavior}."
    sentences.append(s3)

    # --- S4: Overall assessment ---
    strong_signals = sum([
        score >= 0.7,
        ae_risk >= 0.8,
        comp_fraud_r >= 0.3,
        min_anch_d < 4.0,
        knn_fraud >= 1,
        label_src == 'pseudo_positive',
        consensus >= 3,
    ])

    if strong_signals >= 4:
        s4 = ("Overall: multiple independent signals converge on high fraud risk. "
              "Priority review recommended.")
    elif strong_signals >= 2:
        s4 = ("Overall: moderate signal convergence—several independent indicators are elevated. "
              "Review transaction detail and compare to the cluster's confirmed-fraud cases.")
    elif score >= 0.5:
        s4 = ("Overall: score is elevated primarily through ensemble weighting, but "
              "individual behavioural signals are mixed. Classify with moderate confidence.")
    else:
        s4 = ("Overall: weak or conflicting signals. This customer may be a false positive "
              "driven by embedding proximity rather than direct behavioural evidence.")
    sentences.append(s4)

    return ' '.join(sentences)


# ── Main patching logic ───────────────────────────────────────────────────────
def main():
    html = HTML_PATH.read_text(encoding='utf-8')

    # ── Step 1: Extract and enrich DATA JSON (before any HTML patches) ─────
    start_marker = 'const DATA = '
    end_marker   = ';\nconst LS_KEY'
    d_start = html.find(start_marker) + len(start_marker)
    d_end   = html.find(end_marker, d_start)

    data = json.loads(html[d_start:d_end])
    print(f"Loaded {len(data['customers'])} customers from DATA block.")

    # Add top-level metadata
    data['score_descriptions'] = SCORE_DESCRIPTIONS
    data['mcc_map']             = MCC_MAP
    data['industry_map']        = INDUSTRY_MAP

    # Enrich each customer
    for i, c in enumerate(data['customers']):
        c['explanation'] = generate_explanation(c)
        for t in c.get('transactions', []):
            code = str(t.get('merchant_category', '') or '')
            if code and code not in ('nan', 'other', '0', ''):
                t['merchant_name'] = MCC_MAP.get(code, INDUSTRY_MAP.get(code, ''))
            else:
                t['merchant_name'] = ''
        if i % 50 == 0:
            print(f"  Processed {i}/{len(data['customers'])} customers...")

    new_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    print("Explanations generated. Rebuilding HTML...")

    # ── Step 2: Replace DATA block (using original positions, still valid) ─
    html = html[:d_start] + new_json + html[d_end:]

    # ── Step 3: Apply HTML/JS patches (positions recomputed against new html)
    # 3a. Transaction table header: add Merchant Name column
    old_header = '<th>datetime</th><th>amount_cad</th><th>merchant_category</th><th>city</th><th>cash</th><th>ecom</th>'
    new_header = '<th>datetime</th><th>amount_cad</th><th>merchant_category</th><th>merchant</th><th>city</th><th>cash</th><th>ecom</th>'
    if old_header in html:
        html = html.replace(old_header, new_header)
    else:
        print("  WARNING: transaction header not found — skipping header patch")

    # 3b. Transaction row: add merchant_name cell
    # NOTE: template uses fmt(Number(t.amount||0),2) and join('')
    old_row = "`\n    <tr>\n      <td>${t.dt || ''}</td>\n      <td>${fmt(Number(t.amount || 0),2)}</td>\n      <td>${t.merchant_category || ''}</td>\n      <td>${t.city || ''}</td>\n      <td>${t.cash_indicator ?? ''}</td>\n      <td>${t.ecommerce_ind ?? ''}</td>\n    </tr>\n  `).join('');"
    new_row = "`\n    <tr>\n      <td>${t.dt || ''}</td>\n      <td>${fmt(Number(t.amount || 0),2)}</td>\n      <td>${t.merchant_category || ''}</td>\n      <td class=\"muted\">${t.merchant_name || ''}</td>\n      <td>${t.city || ''}</td>\n      <td>${t.cash_indicator ?? ''}</td>\n      <td>${t.ecommerce_ind ?? ''}</td>\n    </tr>\n  `).join('');"
    if old_row in html:
        html = html.replace(old_row, new_row)
    else:
        print("  WARNING: transaction row template not found — skipping row patch")
        # Try to find it for debug
        import re
        m = re.search(r"join\((.{1,10})\);", html[html.find('txnBody'):html.find('txnBody')+1000])
        if m: print("  Found alternative join:", repr(m.group(0)))

    # 3c. Replace raw model card with explanation + glossary + collapsible raw
    old_raw_card = """      <div class="card">
        <h3>Full Model / Profile Context</h3>
        <pre id="raw"></pre>
      </div>"""
    new_raw_card = """      <div class="card" style="border-left:4px solid #0b5fff;margin-bottom:10px;">
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
          &#128214; Score Glossary — what each column means
        </summary>
        <div style="padding-top:4px;max-height:320px;overflow:auto;border:1px solid #e5e7eb;border-radius:8px;">
          <table style="width:100%;border-collapse:collapse;background:#fff;">
            <thead><tr><th style="background:#f8fafc;padding:6px;text-align:left;font-size:12px;">Score Column</th><th style="background:#f8fafc;padding:6px;text-align:left;font-size:12px;">Description</th></tr></thead>
            <tbody id="scoreGlossaryBody"></tbody>
          </table>
        </div>
      </details>"""
    if old_raw_card in html:
        html = html.replace(old_raw_card, new_raw_card)
    else:
        print("  WARNING: raw card HTML not found — skipping card patch")
        idx = html.find('Full Model')
        print("  'Full Model' found at:", idx, "context:", repr(html[idx:idx+100]) if idx >= 0 else 'N/A')

    # 3d. JS: add explanation rendering + glossary population after raw.textContent
    old_raw_js = """  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);

  loadLabel(c.customer_id);"""
    new_raw_js = """  document.getElementById('raw').textContent = JSON.stringify(rawObj, null, 2);

  // Show explanation
  const expEl = document.getElementById('explanationText');
  if (expEl) {
    const lines = (c.explanation || '(No explanation generated.)').split('. ');
    expEl.innerHTML = lines.map(l => l.trim() ? `<p style="margin:0 0 8px 0;">${l.trim()}${l.trim().endsWith('.') ? '' : '.'}</p>` : '').join('');
  }

  // Populate score glossary (once on first render)
  const glossaryBody = document.getElementById('scoreGlossaryBody');
  if (glossaryBody && glossaryBody.childElementCount === 0 && DATA.score_descriptions) {
    glossaryBody.innerHTML = Object.entries(DATA.score_descriptions)
      .map(([k, v]) => `<tr><td style="font-weight:600;white-space:nowrap;padding:5px 8px;font-size:11px;vertical-align:top;">${k}</td><td style="padding:5px 8px;font-size:11px;color:#667085;">${v}</td></tr>`)
      .join('');
  }

  loadLabel(c.customer_id);"""
    if old_raw_js in html:
        html = html.replace(old_raw_js, new_raw_js)
    else:
        print("  WARNING: raw JS setter not found — skipping JS patch")
        idx = html.find("'raw').textContent")
        print("  raw textContent found at:", idx)

    # 3e. renderList: add score tooltip on the score line
    old_list_row = """    div.innerHTML = `
      <div><b>${c.customer_id}</b></div>
      <div class="muted">${c.primary_score_col}: <span class="score">${fmt(c.primary_score, 5)}</span> | txns: ${c.profile.txn_count}</div>
      <div class="muted">component: ${c.model_context.component ?? ''} | hdb: ${c.model_context.hdb_component ?? ''}</div>
      <div class="muted">label: ${lab || 'unlabeled'} ${conf ? `(conf ${conf})` : ''}</div>
    `;"""
    new_list_row = """    const scoreDesc = DATA.score_descriptions ? (DATA.score_descriptions[c.primary_score_col] || '') : '';
    div.innerHTML = `
      <div><b>${c.customer_id}</b></div>
      <div class="muted" title="${scoreDesc.replace(/"/g, '&quot;')}">${c.primary_score_col}: <span class="score">${fmt(c.primary_score, 5)}</span> | txns: ${c.profile.txn_count}</div>
      <div class="muted">component: ${c.model_context.component ?? ''} | hdb: ${c.model_context.hdb_component ?? ''}</div>
      <div class="muted">label: ${lab || 'unlabeled'} ${conf ? `(conf ${conf})` : ''}</div>
    `;"""
    if old_list_row in html:
        html = html.replace(old_list_row, new_list_row)
    else:
        print("  WARNING: list row template not found — skipping list patch")

    print(f"New HTML size: {len(html):,} bytes ({len(html)//1024//1024} MB)")
    HTML_PATH.write_text(html, encoding='utf-8')
    print("Done. Wrote", HTML_PATH)


if __name__ == '__main__':
    main()
