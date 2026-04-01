Web app runtime bundle.

Place runtime artifacts in:
- webapp_resources/outputs
- webapp_resources/data

Recommended workflow:
1. Produce experiment artifacts in outputs/ from notebooks.
2. Run: python sync_webapp_resources.py
3. Commit tracked runtime files via Git LFS.
