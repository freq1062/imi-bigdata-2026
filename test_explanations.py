"""
Test the explanation generator on a few high-risk customers
"""

import pandas as pd
import requests

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
            narrative = narrative.strip('"').strip("'").strip()
            return narrative[:2000] if narrative else f"High-risk ({risk_score:.2f}) customer {flag_text}. Patterns suggest {focus}. {indicators}. Manual review required."
        else:
            return f"High-risk ({risk_score:.2f}) customer {flag_text}. Potential {focus}. Review for {indicators}."
    except Exception as e:
        return f"Error: {e}"


# Load data and get a sample of high-risk customers
df = pd.read_csv("model_output.csv")
high_risk = df[df['risk_score'] >= 0.4].copy()

print(f"Total high-risk customers: {len(high_risk):,}")
print("\nTesting with 5 customers across different risk levels...\n")

# Sample customers at different risk levels
test_samples = []
for threshold in [0.99, 0.90, 0.70, 0.50, 0.42]:
    sample = high_risk[high_risk['risk_score'] >= threshold].head(1)
    if len(sample) > 0:
        test_samples.append(sample.iloc[0])

print("=" * 80)
for i, row in enumerate(test_samples, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/5")
    print(f"{'='*80}")
    print(f"Customer ID: {row['customer_id']}")
    print(f"Risk Score: {row['risk_score']:.4f}")
    print(f"Predicted Label: {row['predicted_label']}")
    print(f"\nGenerating narrative...")
    
    narrative = generate_high_risk_narrative(
        row['customer_id'],
        row['risk_score'],
        row['predicted_label'],
        "http://localhost:11434"
    )
    
    print(f"\nNARRATIVE ({len(narrative)} chars):")
    print("-" * 80)
    print(narrative)
    print("-" * 80)
    
print(f"\n{'='*80}")
print("Test complete! Review the narratives above.")
print("If quality is good, run: python3 generate_explanations.py")
