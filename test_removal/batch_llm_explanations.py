#!/usr/bin/env python3
"""Batch LLM explanation generator.

Writes output to llm_explanations_batch.jsonl (not the large CSV).
The Model Output web page automatically picks up this file.
Estimated runtime: ~500 customers * ~12s = ~100 min.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import pandas as pd, requests

WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUTS = WORKSPACE / "webapp_resources" / "outputs"
CSV = OUTPUTS / "model_output_explanations.csv"
MODEL_OUTPUT_CSV = OUTPUTS / "model_output.csv"
BATCH_JSONL = OUTPUTS / "llm_explanations_batch.jsonl"
OLLAMA_URL = "http://132.145.111.57:11434"
CHECKPOINT_EVERY = 50
LABELS = {
    "anchor_proximity_score": "proximity to known fraud anchors",
    "min_dist_to_fraud_anchor": "distance to nearest fraud anchor",
    "min_dist_to_legit_anchor": "distance to legitimate anchor",
    "mlp_fraud_prob": "neural-net fraud probability",
    "dgi_anomaly_score": "graph anomaly score",
    "cluster_consensus_score": "cluster fraud consensus",
    "customer_ae_risk_norm": "autoencoder anomaly risk",
    "knn_suspicious_share": "share of suspicious neighbours",
    "hdb_outlier_score": "density-based outlier score",
    "knn_gold_fraud_count": "confirmed-fraud neighbours",
    "dist_to_fraud_centroid": "distance to fraud centroid",
    "eft_amount_match_count": "EFT structuring pattern",
    "abm_dc_colocated": "ATM/debit co-location anomaly",
}

def _label(f): return LABELS.get(f, f.replace("_", " "))

def _call_llm(prompt, model):
    r = requests.post(f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.25, "num_predict": 220}}, timeout=50)
    r.raise_for_status()
    return str(r.json().get("response", "")).strip()

def _build_prompt(row):
    cid, rs = str(row["customer_id"]), float(row["risk_score"])
    tier = "HIGH" if rs > 0.7 else ("MEDIUM" if rs > 0.4 else "LOW")
    drivers = []
    for i in (1, 2, 3):
        feat = str(row.get(f"driver_{i}", "") or "")
        shap = float(row.get(f"driver_{i}_shap", 0.0) or 0.0)
        if feat:
            drivers.append(f"{_label(feat)} (SHAP {shap:+.4f})")
    ae_note = ""
    susp = row.get("suspicious_transactions_json", "")
    if susp and pd.notna(susp):
        try:
            s = json.loads(str(susp))
            if s:
                t = s[0]
                ae_note = (f"Flagged transaction: ${float(t.get('amount_cad',0)):,.2f} CAD "
                           f"in {t.get('merchant_category','?')} on "
                           f"{str(t.get('transaction_datetime',''))[:10]}.")
        except Exception:
            pass
    return (
        "You are an AML investigator writing a brief case note for a colleague. "
        "In 3-5 plain English sentences, describe what makes this customer suspicious "
        "and what specific action the reviewer should take next. "
        "Avoid generic phrases. Be direct, concrete, and practical.\n\n"
        f"Customer: {cid}\nRisk tier: {tier} ({rs:.4f})\n"
        f"Top risk factors: {'; '.join(drivers) if drivers else 'no driver data'}\n"
        f"{ae_note}\n"
    )

def main():
    p = argparse.ArgumentParser(description="Batch LLM explanations -> llm_explanations_batch.jsonl")
    p.add_argument("--max-customers", type=int, default=500)
    p.add_argument("--model", default="gemma2:2b")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip customers already in the JSONL output (default: True)")
    args = p.parse_args()

    # Load source data (prefer full explanations CSV, fall back to model_output.csv)
    src_path = CSV if CSV.exists() else MODEL_OUTPUT_CSV
    print(f"Source: {src_path}", flush=True)
    df = pd.read_csv(src_path, on_bad_lines="skip", low_memory=False)
    df["customer_id"] = df["customer_id"].astype(str)

    # Load already-done customers from JSONL
    done: dict[str, dict] = {}
    if args.resume and BATCH_JSONL.exists():
        with open(BATCH_JSONL) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done[rec["customer_id"]] = rec
                except Exception:
                    pass
        print(f"Resuming: {len(done)} customers already have LLM explanations", flush=True)

    # Also consider customers with llm_status=ok in the source CSV as done
    if "llm_status" in df.columns:
        csv_done = set(df[df["llm_status"] == "ok"]["customer_id"])
    else:
        csv_done = set()

     # Sort by risk_score descending
    rs_col = "risk_score" if "risk_score" in df.columns else (
        "lgb_fraud_prob" if "lgb_fraud_prob" in df.columns else None)
    if rs_col:
        df = df.sort_values(rs_col, ascending=False)
    else:
        print("WARNING: no risk_score column found, using original order", flush=True)

    # Assign risk tier if not already present
    if "risk_tier" not in df.columns and rs_col:
        df["risk_tier"] = df[rs_col].apply(
            lambda x: "HIGH" if x > 0.7 else ("MEDIUM" if x > 0.4 else "LOW")
        )

    already_done = set(done.keys()) | csv_done
    pending_df = df[~df["customer_id"].isin(already_done)]

    # Pick top 20 HIGH + top 20 MEDIUM (highest score within each tier)
    n_per_tier = args.max_customers // 2
    high = pending_df[pending_df["risk_tier"].str.upper() == "HIGH"].head(n_per_tier)
    medium = pending_df[pending_df["risk_tier"].str.upper() == "MEDIUM"].head(n_per_tier)
    pending = pd.concat([high, medium], ignore_index=True)
    print(
        f"Total: {len(df):,}  Already done: {len(already_done):,}  "
        f"Selected: {len(high)} HIGH + {len(medium)} MEDIUM = {len(pending)}  "
        f"Model: {args.model}",
        flush=True,
    )

    # LLM check
    print(f"Checking {OLLAMA_URL} ...", flush=True)
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        print(f"Models: {[m['name'] for m in r.json().get('models',[])]}", flush=True)
    except Exception as e:
        print(f"ERROR: LLM unreachable: {e}", file=sys.stderr); sys.exit(1)

    if pending.empty:
        print("All done! Nothing left to process."); return

    ok = err = 0
    t0 = time.time()

    # Append-mode JSONL so we can resume safely
    out_fh = open(BATCH_JSONL, "a", encoding="utf-8")

    try:
        for idx, (_, row) in enumerate(pending.iterrows()):
            cid = str(row["customer_id"])
            prompt = _build_prompt(row)
            try:
                t1 = time.time()
                expl = _call_llm(prompt, args.model)
                elapsed = time.time() - t1
                rec = {
                    "customer_id": cid,
                    "explanation_text": expl,
                    "llm_status": "ok",
                    "model": args.model,
                }
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
                ok += 1
                done_n = idx + 1
                eta = ((time.time() - t0) / done_n) * (len(pending) - done_n)
                print(f"[{done_n}/{len(pending)}] {cid}  {elapsed:.1f}s  ETA {eta/60:.0f}m  ok={ok}", flush=True)
            except Exception as e:
                err += 1
                print(f"[{idx+1}/{len(pending)}] {cid}  ERROR: {e}", flush=True)
    finally:
        out_fh.close()

    total = time.time() - t0
    print(f"\nDone. ok={ok} errors={err} time={total/60:.1f}m avg={total/max(ok,1):.1f}s/cust", flush=True)
    print(f"Output: {BATCH_JSONL} ({BATCH_JSONL.stat().st_size//1024}KB)", flush=True)

if __name__ == "__main__":
    main()
