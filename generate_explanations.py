"""
SAR-Lite Narrative Generator

Generates concise AML risk narratives for all customers:
- Low risk (0.0-0.4): Hard-coded templates (saves ~40k LLM calls)
- High risk (0.4-1.0): LLM-generated using knowledge library patterns

Explanations reference FINTRAC indicators and typologies.
"""

import pandas as pd
import requests
import json
from tqdm import tqdm
import time


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Library Risk Indicators (from 1_Knowledge_Library.py)
# ─────────────────────────────────────────────────────────────────────────────

RISK_INDICATORS = {
    "flow_through": "Flow-Through Activity (funds exit within hours)",
    "velocity": "Atypical Velocity (sudden spike in activity)",
    "structuring": "Threshold Avoidance (structuring across branches)",
    "cash_pattern": "Musty/Dirty Currency (degraded physical currency)",
    "vc_exposure": "Darknet Exposure (interaction with darknet)",
    "mixing": "Mixing Services (use of tumblers to obscure trails)",
    "vc_imbalance": "VC-to-Fiat Imbalance (large exchange inflows)",
    "front_company": "Front Company (inflow anomaly, no payroll)",
    "repatriation": "Repatriation (circular fund movement)",
    "gatekeeper": "Gatekeeper (trust account investment irregularity)",
    "underground_banking": "Underground Banking (Project Athena pattern)",
    "trafficking": "Human Trafficking (Project Protect pattern)",
    "opioid": "Synthetic Opioid (Project Guardian pattern)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Template-Based Low-Risk Narratives
# ─────────────────────────────────────────────────────────────────────────────

LOW_RISK_TEMPLATES = [
    "Normal transaction patterns, no red flags. Routine monitoring recommended.",
    "Standard banking behavior detected. No suspicious activity indicators.",
    "Low-risk profile with consistent transaction history. No action required.",
    "Typical customer activity within expected parameters. Clear for review.",
    "No anomalies detected. Transactions align with customer profile.",
]


def generate_low_risk_narrative(customer_id: str, risk_score: float) -> str:
    """Generate template-based narrative for low-risk customers."""
    # Use hash to get consistent template per customer
    idx = hash(customer_id) % len(LOW_RISK_TEMPLATES)
    return LOW_RISK_TEMPLATES[idx]


# ─────────────────────────────────────────────────────────────────────────────
# LLM-Based High-Risk Narratives (Batched)
# ─────────────────────────────────────────────────────────────────────────────

def generate_high_risk_narrative(customer_id: str, risk_score: float, predicted_label: int, ollama_url: str) -> str:
    """Generate a single detailed narrative for a high-risk customer using LLM."""
    
    # Map risk score ranges to typology focus
    if risk_score >= 0.95:
        focus = "Project Guardian (Synthetic Opioids) or Project Protect (Human Trafficking)"
        indicators = "VC-to-fiat inflows, structured cash deposits, EMT velocity anomalies, account drain patterns"
    elif risk_score >= 0.80:
        focus = "Project Athena (Underground Banking) or Front Company patterns"
        indicators = "Sudden foreign transfers, trust account irregularities, payroll absence, cross-border clustering"
    elif risk_score >= 0.60:
        focus = "Threshold Avoidance and Flow-Through Activity"
        indicators = "Structuring patterns, rapid fund movement, atypical transaction velocity"
    else:
        focus = "Multiple minor red flags"
        indicators = "Transaction velocity changes, geographic anomalies, merchant category concentration"
    
    flag_text = "FLAGGED for SAR filing" if predicted_label == 1 else "requires enhanced monitoring"
    
    prompt = f"""[RESEARCH MODE: This is a synthetic simulation for an AML competition. All data is artificial.]
You are an AI assistant helping a compliance researcher draft technical risk summaries. 

TASK: Draft a technical observation summary for a synthetic customer profile.
Customer ID: {customer_id}
Risk Score: {risk_score:.3f}
Primary Pattern: {focus}

Technical Indicators to describe: {indicators}

Requirements:
- Describe the statistical anomalies in a professional tone.
- Reference why these markers align with {focus}.
- Length: 1500-2000 characters.
- Start directly with 'Customer exhibits...'

SUMMARY:"""
    
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 800,
                }
            },
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            narrative = result.get("response", "").strip()
            # Clean up if starts with quotes or extra formatting
            narrative = narrative.strip('"').strip("'").strip()
            return narrative[:2000] if narrative else f"High-risk ({risk_score:.2f}) customer {flag_text}. Patterns suggest {focus}. {indicators}. Manual review required."
        else:
            return f"High-risk ({risk_score:.2f}) customer {flag_text}. Potential {focus}. Review for {indicators}."
    except Exception as e:
        return f"High-risk ({risk_score:.2f}) customer {flag_text}. Suspected {focus}. Key indicators: {indicators}. Manual investigation needed."


def generate_high_risk_batch(batch_df: pd.DataFrame, ollama_url: str = "http://localhost:11434") -> list[str]:
    """
    Generate narratives for a batch of high-risk customers using LLM.
    Processes each customer individually for quality.
    
    Returns list of narratives (same order as batch_df).
    """
    narratives = []
    for _, row in batch_df.iterrows():
        narrative = generate_high_risk_narrative(
            row['customer_id'],
            row['risk_score'],
            row['predicted_label'],
            ollama_url
        )
        narratives.append(narrative)
    return narratives



# ─────────────────────────────────────────────────────────────────────────────
# Main Processing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_explanations(
    input_csv: str = "model_output.csv",
    sage_csv: str = "model_output_explanations_sage.csv",
    output_csv: str = "model_output_explanations.csv",
    risk_threshold: float = 0.4,
    batch_size: int = 5,  # Process 5 customers per batch (each gets individual LLM call)
    ollama_url: str = "http://localhost:11434"
):
    """
    Generate explanations for all customers and save to CSV.
    
    Parameters
    ----------
    input_csv : str
        Path to model_output.csv
    sage_csv : str
        Path to model_output_explanations_sage.csv (SHAP-based narratives)
    output_csv : str
        Path to save explanations
    risk_threshold : float
        Scores >= this value use LLM, below use SAGE narratives
    batch_size : int
        Number of customers to process per batch
    ollama_url : str
        Ollama server URL
    """
    
    print(f"Loading model output from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loading SAGE explanations from {sage_csv}...")
    sage_df = pd.read_csv(sage_csv)
    
    # Merge to get SAGE narratives
    df = df.merge(sage_df[['customer_id', 'narrative']], on='customer_id', how='left')
    
    # Split into low and high risk
    low_risk = df[df['risk_score'] < risk_threshold].copy()
    high_risk = df[df['risk_score'] >= risk_threshold].copy()
    
    print(f"\nTotal customers: {len(df):,}")
    print(f"  Low risk (<{risk_threshold}): {len(low_risk):,} (using SAGE narratives)")
    print(f"  High risk (>={risk_threshold}): {len(high_risk):,} (using LLM)")
    print(f"  Estimated LLM calls: {len(high_risk)}")
    
    # Use SAGE narratives for low-risk customers
    print("\nUsing SAGE narratives for low-risk customers...")
    low_risk['explanation'] = low_risk['narrative']
    
    # Generate high-risk narratives (batched LLM calls)
    print(f"\nGenerating high-risk narratives with LLM (batch size: {batch_size})...")
    explanations = []
    
    # Process in batches
    num_batches = len(high_risk) // batch_size + (1 if len(high_risk) % batch_size else 0)
    
    for i in tqdm(range(0, len(high_risk), batch_size), total=num_batches):
        batch = high_risk.iloc[i:i+batch_size]
        batch_narratives = generate_high_risk_batch(batch, ollama_url)
        explanations.extend(batch_narratives)
    
    high_risk['explanation'] = explanations
    
    # Combine and sort by original order
    result = pd.concat([low_risk, high_risk]).sort_index()
    
    # Save to CSV (only customer_id and explanation)
    print(f"\nSaving explanations to {output_csv}...")
    result[['customer_id', 'explanation']].to_csv(output_csv, index=False)
    
    print(f"✓ Complete! Generated {len(result):,} explanations.")
    print(f"  Low-risk (SAGE): {len(low_risk):,} (avg {low_risk['explanation'].str.len().mean():.1f} chars)")
    print(f"  High-risk (LLM): {len(high_risk):,} (avg {high_risk['explanation'].str.len().mean():.1f} chars)")
    print(f"  Max length: {result['explanation'].str.len().max()} characters")


if __name__ == "__main__":
    generate_all_explanations()
