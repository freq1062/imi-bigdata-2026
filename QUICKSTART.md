# Quick Start: GNNExplainer Explainability Pipeline

## 1️⃣ Setup Ollama (5 minutes)

### Via Docker (Easiest)
```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull mistral
curl http://localhost:11434/api/tags  # Verify
```

### Via Direct Install
```bash
# macOS/Linux
curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull mistral
```

---

## 2️⃣ Prepare Artifacts

Verify these files exist in your workspace:
```
✓ dgi_model.pt              (DGI encoder checkpoint)
✓ dgi_embeddings.pt         (Customer embeddings, 61410 × 64)
✓ dgi_gmm.joblib            (Fitted GMM with K=200)
✓ dgi_component_assignments.csv
✓ data_graph.pt             (Heterogeneous graph)
✓ transactions_with_features.csv
```

If missing, run your training pipeline first.

---

## 3️⃣ Run Notebook

```bash
# In VS Code: Open explainability_gnnexplainer.ipynb
# Run cells in order:
1. Imports & Config
2. Load Pre-Trained Artifacts
3. Phase 1: Representative Sampling (30 sec)
4. Phase 2: Explanation Generation (10-20 min) ⏱️
5. Phase 3: Synthesizing Cluster Profiles (1 min)
6. Phase 4: LLM-Based Cluster Interpretation (5-10 min if Ollama running)
7. Export Results
8. Visualization
```

---

## 4️⃣ Review Outputs

### Cluster Profiles
```python
import pandas as pd
profiles = pd.read_csv('gnnexplainer_cluster_profiles.csv')
print(profiles[['component_id', 'top_features', 'dominant_structural_motif', 'num_transactions']])
```

### LLM Narratives
```python
narratives = pd.read_csv('gnnexplainer_cluster_narratives.csv')
for _, row in narratives.iterrows():
    print(f"Component {row['component_id']}: {row['interpretation'][:200]}...")
```

### Visualization
- Check `gnnexplainer_cluster_analysis.png` for dashboard charts

---

## 5️⃣ Integrate into Streamlit

Add to your `Home.py`:

```python
import pandas as pd
import streamlit as st

st.header("🧠 Cluster Interpretations (GNNExplainer)")

narratives = pd.read_csv('gnnexplainer_cluster_narratives.csv')

for _, row in narratives.iterrows():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cluster", int(row['component_id']))
    with col2:
        st.metric("Motif", row['dominant_motif'].upper())
    with col3:
        st.metric("Avg Amount", f"${row['avg_amount']:.2f}")
    
    st.markdown(f"**Features:** {row['features']}")
    st.text_area("Interpretation", row['interpretation'], height=120, disabled=True)
    st.divider()
```

---

## 🆘 Troubleshooting

| Issue | Fix |
|-------|-----|
| **Ollama not found** | `ollama serve` & wait 3 sec |
| **"Cannot connect to Ollama"** | `curl http://localhost:11434/api/tags` |
| **GNNExplainer slow** | Normal (10-20 min for full dataset). Try subset first. |
| **Missing artifact files** | Run training pipeline first |
| **Memory error** | Reduce n_samples or num_epochs in Phase 2 |

---

## 📊 Expected Output Files

```
gnnexplainer_cluster_profiles.csv          ← Main analysis results
gnnexplainer_cluster_narratives.csv        ← LLM interpretations
gnnexplainer_detailed_explanations.json    ← Raw GNNExplainer data
gnnexplainer_cluster_analysis.png          ← Visualization dashboard
```

---

## ⏱️ Runtime Estimates

| Phase | Time | Notes |
|-------|------|-------|
| Phase 1 | 30 sec | Medoid sampling |
| Phase 2 | **10-20 min** | GNNExplainer (intensive) |
| Phase 3 | 1 min | Aggregation |
| Phase 4 | 2-3 min | Per cluster w/ LLM |
| **Total** | **~25-35 min** | First run (only Phase 2 scales with data) |

---

## 🚀 Pro Tips

1. **Speed up**: 
   - Reduce to fraud components only (Phase 1)
   - Use `mistral` or `neural-chat` model (faster than `orca`)
   - Lower num_epochs from 100 to 50 in Phase 2

2. **Quality**:
   - Use `orca` model for best interpretations
   - Review template prompts even without Ollama
   - Combine with RF/SHAP for per-customer explanations

3. **Production**:
   - Cache results: `narratives_df.to_json()` for fast reloads
   - Schedule Phase 4 separately (LLM calls are slow)
   - Monitor Ollama resource usage

---

## 📚 Key Concepts

- **Medoid**: Most representative node in a cluster (highest GMM probability)
- **Edge Mask**: GNNExplainer output showing which connections matter (0-1)
- **Feature Mask**: GNNExplainer output showing which node features matter (0-1)
- **Structural Motif**: Graph pattern (star/chain/clique) of fraud connections
- **Cluster Profile**: Aggregated summary of patterns across 5 medoids

---

## ✅ Validation Checklist

- [ ] Ollama running: `curl http://localhost:11434/api/tags`
- [ ] Artifacts present (6 files) ✓
- [ ] Notebook executes without errors ✓
- [ ] cluster_profiles.csv has >0 rows ✓
- [ ] cluster_narratives.csv populated ✓
- [ ] Visualization PNG generated ✓
- [ ] Ollama interpretations quality reviewed ✓

---

**Next Step**: Open `explainability_gnnexplainer.ipynb` and run Phase 1 now! ⚡
