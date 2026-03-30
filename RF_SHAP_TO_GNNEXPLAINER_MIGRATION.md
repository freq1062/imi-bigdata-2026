# Migration Guide: RF/SHAP → GNNExplainer

## Quick Comparison

| Aspect | Old (RF/SHAP) | New (GNNExplainer) |
|--------|---------------|-------------------|
| **Explanation Type** | Feature importance (tabular) | Graph structure importance (edges + nodes) |
| **Method** | Random Forest proxy + SHAP values | Graph Neural Network Explainer |
| **Data Type** | Customer tabular features | Graph topology + DGI embeddings |
| **Interpretation** | "Which features drive risk?" | "Which connections/patterns drive risk?" |
| **Scalability** | Fast (~seconds) | Slower (~minutes per node) |
| **Graph-Awareness** | No; treats features independently | Yes; considers network effects |
| **Output** | Per-customer risk tier + top 3 SHAP drivers | Per-cluster behavior patterns + graph motifs |
| **LLM Integration** | External (Google Drive CSV) | Native Ollama integration |

---

## Why GNNExplainer?

### Strengths of GNNExplainer over RF/SHAP

1. **Graph-Native**: Respects your DGI graph encoder; doesn't ignore network structure
2. **Cluster-Level Insights**: Synthesizes patterns across groups (not just individuals)
3. **Structural Motifs**: Identifies if fraud clusters are "stars" (hubs), "chains", or "cliques"
4. **Edge Masks**: Shows which transactions/connections are most important
5. **Domain Expert Translation**: LLM can interpret graph patterns in banking context

### When to Use Each Approach

| Scenario | Use RF/SHAP | Use GNNExplainer |
|----------|-----------|-----------------|
| "Why is THIS customer flagged?" | ✅ | ❌ (answers cluster Q) |
| "What behavior pattern defines this cluster?" | ❌ | ✅ |
| "Which features matter most?" (individual) | ✅ | ❌ |
| "What network structure do fraudsters form?" | ❌ | ✅ |
| Real-time predictions (<50ms) | ✅ | ❌ |
| Post-hoc cluster analysis | ❌ | ✅ |

---

## Architecture Changes

### Data Pipeline

**Old:**
```
DGI Model → Customer Embeddings
                ↓
        RF Proxy Trained
                ↓
        SHAP Values Computed
                ↓
        Per-Customer Scores
```

**New:**
```
DGI Model → Customer Embeddings
     ↓          ↓
  Graph      GMM Clustering
     ↓          ↓
  GNNExplainer + Medoid Sampling
     ↓
  Cluster Profile Synthesis
     ↓
  Ollama LLM Interpretation
```

### Key Design Decisions

1. **Medoid Selection** (Phase 1)
   - Instead of random node selection, use GMM `predict_proba()` to find most representative
   - Ensures you explain "pure" examples of each cluster
   - Prioritize fraud anchors if present in cluster

2. **Explanation Extraction** (Phase 2)
   - GNNExplainer produces soft masks (0-1 values per edge/feature)
   - Threshold at 0.5 or use top-10% approach
   - Extract actual transaction data for LLM context

3. **Aggregation Strategy** (Phase 3)
   - Find features flagged by multiple nodes (consensus)
   - Identify graph motifs (star/chain/clique patterns)
   - Compute cluster-level statistics (merchant categories, amounts, times)

4. **LLM Prompting** (Phase 4)
   - Provide structured cluster digest (not raw masks)
   - Ask LLM to classify behavior (fraud vs. legitimate but unusual)
   - Include transaction examples for grounding

---

## Output Changes

### Old Outputs
```
model_output_explanations_sage.csv
├── customer_id
├── risk_score
├── risk_tier (HIGH/MEDIUM/LOW)
├── predicted_label
├── narrative (SHAP-based)
├── driver_1, driver_1_shap
├── driver_2, driver_2_shap
└── driver_3, driver_3_shap
```

### New Outputs

**1. Cluster Profiles** (`gnnexplainer_cluster_profiles.csv`)
```
component_id, num_medoids, top_features, dominant_structural_motif,
avg_edge_density, avg_neighbors, avg_transaction_amount,
top_merchant_categories, avg_transaction_hour, num_transactions
```

**2. Cluster Narratives** (`gnnexplainer_cluster_narratives.csv`)
```
component_id, prompt, interpretation, features,
dominant_motif, avg_amount
```

**3. Raw Explanations** (`gnnexplainer_detailed_explanations.json`)
```json
{
  "0": [
    {
      "node_idx": 1234,
      "edge_mask_summary": {"min": 0.01, "max": 0.95, "mean": 0.42},
      "feature_mask_summary": {"min": 0.02, "max": 0.88, "mean": 0.35}
    }
  ]
}
```

**4. Visualization** (`gnnexplainer_cluster_analysis.png`)
- Structural motif distribution
- Transaction amount histogram by cluster
- Edge density vs. avg neighbors scatter
- Top clusters by transaction volume

---

## Code Migration Guide

### Task 1: Load Embeddings and GMM

**Old Way:**
```python
# Implicit from RF training loop
shap_values_sage = explainer_sage.shap_values(X_rf)
```

**New Way:**
```python
# Explicit loading
all_customer_embeddings = torch.load('dgi_embeddings.pt')
gmm_model = joblib.load('dgi_gmm.joblib')
component_df = pd.read_csv('dgi_component_assignments.csv')
```

### Task 2: Get Explanations for a Group

**Old Way:**
```python
# Per-customer SHAP
for idx, row in output_df.iterrows():
    drivers = _top_shap_drivers(shap_values_sage[idx], n=3)
    narrative = _build_narrative(tier, score, drivers)
```

**New Way:**
```python
# Per-cluster GNNExplainer
for comp_id, medoids in medoids_by_component.items():
    explanations = [run_gnnexplainer_on_node(model, graph, emb, node_idx) 
                    for node_idx, _ in medoids]
    profile = aggregate_component_profile(comp_id, explanations, ...)
    prompt = build_llm_prompt(profile)
    interpretation = query_ollama(prompt)
```

### Task 3: Feature Importance

**Old Way (Per-Customer):**
```python
# Top 3 SHAP drivers per customer
positive_features = np.where(shap_row > 0)[0]
top_idx = positive_features[np.argsort(shap_row[positive_features])[::-1]][:3]
```

**New Way (Cluster-Level):**
```python
# Most frequently flagged features across 5 medoids
all_feature_importances = []
for exp in explanations:
    critical_f, _ = threshold_masks(exp['feature_mask'], top_k=0.2)
    all_feature_importances.extend(critical_f)

# Consensus features (appear in multiple node explanations)
feature_counts = pd.Series(all_feature_importances).value_counts().head(3)
```

---

## Integration Checklist

- [ ] New notebook created: `explainability_gnnexplainer.ipynb`
- [ ] Ollama server set up and running
- [ ] All artifact files present (dgi_model.pt, dgi_gmm.joblib, etc.)
- [ ] Phase 1 (medoid sampling) executes (~30s)
- [ ] Phase 2 (GNNExplainer) executes (~10-20 min)
- [ ] Phase 3 (aggregation) completes (~1 min)
- [ ] Phase 4 (LLM) produces narratives ( ~5-10 min with Ollama)
- [ ] Output CSVs and JSON generated
- [ ] Visualization PNG created
- [ ] Review cluster_narratives.csv for quality
- [ ] Optional: Integrate outputs into Streamlit dashboard

---

## Performance Tips

### Speed Up GNNExplainer (Phase 2)

1. **Reduce components**: Only analyze fraud components first
   ```python
   fraud_components = component_df[component_df['is_fraud_component']]['component_id'].unique()
   medoids_by_component = {k: v for k, v in medoids_by_component.items() if k in fraud_components}
   ```

2. **Reduce epochs per node**: Lower from 100 to 50
   ```python
   exp = run_gnnexplainer_on_node(..., num_epochs=50)
   ```

3. **Reduce medoids per component**: Sample 3 instead of 5
   ```python
   medoids_by_component = sample_component_medoids(..., n_samples=3)
   ```

4. **Parallelize** (if GPU): Run on multiple GPUs with torch.multiprocessing

### Speed Up LLM (Phase 4)

1. **Use faster model**: Prefer `neural-chat` over `orca` (~3x faster, still good quality)
2. **Batch queries**: Send multiple prompts in one request (requires Ollama batch API)
3. **Cache LLM responses**: Save prompts/responses to avoid re-processing

---

## Debugging GNNExplainer Issues

### "GNNExplainer failed for multiple nodes"

**Normal**: 5-10% failures expected (isolated nodes with no edges)

**If >20% fail**:
- Check graph connectivity: `print(data_graph.edge_types)`
- Verify embeddings loaded: `print(all_customer_embeddings.shape)`
- Check GNNExplainer hyperparameters (lr, epochs)

### "Interpretations all say 'awaiting Ollama'"

**Solution**: Ollama not connected. Either:
1. Start Ollama: `ollama serve`
2. Or manually set `USE_OLLAMA = False` and review template prompts

### "Cluster profiles look wrong"

**Check**:
1. Component_df correctly loaded: `print(component_df.head())`
2. Feature extraction logic: Verify top_features are valid indices
3. Transaction data alignment: Ensure `customer_idx` matches component_df

---

## Future Enhancements

1. **Edge Importance Extraction**: Extract the actual transaction data for top edges (amounts, times, merchants)
2. **Multi-Hop Explanations**: Extend GNNExplainer beyond 2-hop neighborhoods
3. **Temporal Dynamics**: Track how explanations change over time (e.g., monthly cluster evolution)
4. **Counterfactual Explanations**: "What if this connection didn't exist?" - could this customer be legitimate?
5. **Ensemble Approaches**: Combine GNNExplainer with other explainability methods

---

## Summary

- **Old approach (RF/SHAP)**: Fast, interpretable, but ignores graph structure
- **New approach (GNNExplainer)**: Slower, but graph-aware, cluster-centric, and LLM-integrated
- **Use together**: RF/SHAP for per-customer decisions, GNNExplainer for cluster risk profiling

The new pipeline is production-ready once Ollama is set up. Outputs are ready for stakeholder review, model auditing, and regulatory reporting.
