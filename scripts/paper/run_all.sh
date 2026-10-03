#!/usr/bin/env bash
# MD-DGFLS Unified Research Reproduction Master Script (scripts/paper/run_all.sh)
# Executes all evaluation phases end-to-end and validates zero regression against baseline.

set -e

export PYTHONPATH=.

echo "========================================================================"
echo "MD-DGFLS: End-to-End Research Paper Reproduction Pipeline"
echo "========================================================================"

echo ""
echo "[Step 1/6] Running Phase 1 Metric & Handbook Audit..."
python3 scripts/paper/audit_handbook_metrics.py

echo ""
echo "[Step 2/6] Running Phase 2 Public Benchmark & Baseline Comparisons..."
python3 scripts/paper/generate_main_results.py

echo ""
echo "[Step 3/6] Running Phase 3 Router Calibration & Ablation Study..."
python3 src/analysis/router_eval.py
python3 src/analysis/cross_domain.py
python3 scripts/paper/generate_ablation_results.py

echo ""
echo "[Step 4/6] Running Phase 4 Robustness, XAI, and LLM Copilot Audits..."
python3 scripts/paper/generate_robustness_results.py
python3 src/analysis/xai_eval.py
python3 src/analysis/llm_audit.py

echo ""
echo "[Step 5/7] Generating IEEE Publication Figures..."
MPLCONFIGDIR=/tmp/mpl python3 scripts/paper/generate_figures.py

echo ""
echo "[Step 6/7] Executing Unit Test Suite (Original + New Evaluation Suites)..."
python3 -m unittest discover tests

echo ""
echo "[Step 7/7] Executing 13-Operation Multi-Domain Operational Verification..."
git restore outputs/ 2>/dev/null || true
python3 tests/verify_all_domains.py

echo ""
echo "========================================================================"
echo "SUCCESS: All evaluation experiments, tables, and regression gates passed!"
echo "Artifacts located in results/ and paper/"
echo "========================================================================"
