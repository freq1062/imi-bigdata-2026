"""
explanation_card.py — Reusable display components for the Streamlit UI.

Provides:
  - render_customer_card(): Full card combining graph heatmap, transaction
    heatmap, and explanation text. Used by both Model Output and Run Model pages.
  - build_heatmap_graph(): Graph with hub nodes coloured by GNN importance.
  - build_transaction_heatmap(): Dataframe with rows coloured by AE risk score.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from pyvis.network import Network

import streamlit.components.v1 as _stcomp

from lib.components import risk_color, risk_bg, _load_industry_lookup
from lib.resource_paths import OUTPUTS_DIR


# ── Colour helpers ──────────────────────────────────────────────────────────

def _ae_risk_color(score: float) -> str:
    """Map AE risk score [0,1] to a hex colour (green → yellow → red)."""
    if score <= 0:
        return "#22c55e"  # green
    t = min(1.0, score)
    if t < 0.5:
        # green → yellow
        r = int(255 * (t / 0.5))
        g = 255
        b = 0
    else:
        # yellow → red
        r = 255
        g = int(255 * (1 - (t - 0.5) / 0.5))
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


def _gnn_hub_color(importance: float) -> str:
    """Map GNN hub importance [0,1] to a hex colour (dim → bright purple)."""
    if importance <= 0:
        return "#64748b"  # slate
    t = min(1.0, importance)
    # interpolate slate → bright purple
    r = int(100 + 155 * t)
    g = int(116 + 52 * t)
    b = int(139 + 116 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Graph with GNN hub heatmap ─────────────────────────────────────────────

def build_heatmap_graph(
    customer_id: str,
    risk_score: float,
    cat_counts: dict[str, int] | list[str],
    city_counts: dict[str, int] | list[str],
    hub_importance: dict[str, float] | None = None,
    height: str = "380px",
) -> str:
    """Build a PyVis graph with hub nodes coloured by GNN importance.

    Parameters
    ----------
    customer_id : str
    risk_score : float
        Customer's risk score (used for the centre node colour).
    cat_counts, city_counts : dict or list
        Merchant categories and cities the customer is connected to.
    hub_importance : dict or None
        Mapping of hub name → importance score [0,1] from GNNExplainer.
        Hub names should match the keys in cat_counts / city_counts.
        If None or empty, all hubs get a default dim colour.
    height : str
        PyVis canvas height.

    Returns
    -------
    str
        HTML string with embedded PyVis graph.
    """
    if isinstance(cat_counts, list):
        cat_counts = {c: 1 for c in cat_counts}
    if isinstance(city_counts, list):
        city_counts = {c: 1 for c in city_counts}
    if hub_importance is None:
        hub_importance = {}

    G = nx.Graph()
    c_color = risk_color(risk_score)
    ind_lookup = _load_industry_lookup()

    total_txns = sum(cat_counts.values()) + sum(city_counts.values())
    cust_label = f"Customer\n{customer_id[:14]}"
    cust_tip = (
        f"CUSTOMER\n"
        f"ID: {customer_id}\n"
        f"Risk Score: {risk_score:.1%}\n"
        f"Merchant Categories: {len(cat_counts)}\n"
        f"Cities: {len(city_counts)}\n"
        f"Total Edges: {total_txns}"
    )
    G.add_node(
        customer_id,
        label=cust_label,
        color=c_color,
        size=35,
        title=cust_tip,
        shape="dot",
        font={"size": 12, "color": "white"},
    )

    for cat, tx_count in cat_counts.items():
        node_id = f"cat::{cat}"
        imp = hub_importance.get(cat, hub_importance.get(node_id, 0.0))
        node_color = _gnn_hub_color(imp)

        cat_name = ind_lookup.get(cat, cat)
        if cat_name != cat:
            short = f"{cat}\n{cat_name[:20] + chr(8230) if len(cat_name) > 20 else cat_name}"
            tip = (
                f"MERCHANT CATEGORY\n"
                f"Code: {cat}\n"
                f"Name: {cat_name}\n"
                f"Transactions: {tx_count}\n"
                f"GNN importance: {imp:.2%}"
            )
        else:
            short = cat[:18] + chr(8230) if len(cat) > 18 else cat
            tip = (
                f"MERCHANT CATEGORY\n"
                f"{cat}\n"
                f"Transactions: {tx_count}\n"
                f"GNN importance: {imp:.2%}"
            )

        node_size = max(14, min(32, 14 + tx_count * 2 + imp * 10))
        edge_width = max(1.5, min(8.0, 1.5 + tx_count * 0.6 + imp * 3))
        G.add_node(node_id, label=short, color=node_color, size=node_size, title=tip, shape="dot")
        G.add_edge(
            customer_id, node_id,
            color=node_color,
            width=edge_width,
            title=f"Transactions: {tx_count} | GNN importance: {imp:.2%}",
            label=str(tx_count) if tx_count > 1 else "",
        )

    for city, tx_count in city_counts.items():
        node_id = f"city::{city}"
        imp = hub_importance.get(city, hub_importance.get(node_id, 0.0))
        node_color = _gnn_hub_color(imp)

        short = city[:18] + chr(8230) if len(city) > 18 else city
        tip = (
            f"CITY\n"
            f"{city}\n"
            f"Transactions: {tx_count}\n"
            f"GNN importance: {imp:.2%}"
        )
        node_size = max(14, min(32, 14 + tx_count * 2 + imp * 10))
        edge_width = max(1.5, min(8.0, 1.5 + tx_count * 0.6 + imp * 3))
        G.add_node(node_id, label=short, color=node_color, size=node_size, title=tip, shape="dot")
        G.add_edge(
            customer_id, node_id,
            color=node_color,
            width=edge_width,
            title=f"Transactions: {tx_count} | GNN importance: {imp:.2%}",
            label=str(tx_count) if tx_count > 1 else "",
        )

    # Render
    net = Network(height=height, width="100%", bgcolor="#1e293b", font_color="#e2e8f0")
    net.from_nx(G)
    net.set_options("""
    {
        "physics": { "enabled": true, "stabilization": { "iterations": 50 },
                     "solver": "forceAtlas2Based",
                     "forceAtlas2Based": { "gravitationalConstant": -40,
                                            "springLength": 180,
                                            "springConstant": 0.01 } },
        "interaction": { "hover": true, "tooltipDelay": 200 },
        "edges": { "smooth": { "type": "continuous" } }
    }
    """)
    html = net.generate_html()

    # Inject GNN importance legend
    legend = (
        "<div style='display:flex;gap:18px;align-items:center;padding:4px 12px;"
        "font-size:0.75rem;color:#94a3b8;flex-wrap:wrap;margin-top:2px;'>"
        "<span><span style='display:inline-block;width:12px;height:12px;"
        "border-radius:50%;background:#64748b;margin-right:4px;vertical-align:middle;'></span>"
        "Low GNN importance</span>"
        "<span><span style='display:inline-block;width:12px;height:12px;"
        "border-radius:50%;background:#a855f7;margin-right:4px;vertical-align:middle;'></span>"
        "High GNN importance</span>"
        "<span style='color:#64748b;'>Hub nodes coloured by how influential they were in the model's decision</span>"
        "</div>"
    )
    return html.replace("</body>", legend + "</body>")


# ── Transaction heatmap ─────────────────────────────────────────────────────

def build_transaction_heatmap(
    txns_df: pd.DataFrame,
    ae_scores_lookup: dict[str, float] | None = None,
    max_rows: int = 50,
) -> pd.DataFrame:
    """Build a transaction dataframe with AE-risk-based styling.

    Parameters
    ----------
    txns_df : pd.DataFrame
        Must contain columns: transaction_id, amount, merchant_category,
        city, transaction_datetime, etc.
    ae_scores_lookup : dict or None
        Mapping of {transaction_id: ae_risk_score} to colour rows by.
    max_rows : int
        Maximum rows to return.

    Returns
    -------
    pd.DataFrame
        With an added 'ae_risk_score' and 'amount' columns. The caller
        should use `st.dataframe(styled_df.style.apply(...))` or iterate.
    """
    if txns_df.empty:
        return txns_df

    df = txns_df.copy()

    # Normalise columns
    if "amount" not in df.columns and "amount_cad" in df.columns:
        df["amount"] = pd.to_numeric(df["amount_cad"], errors="coerce").fillna(0.0)
    df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0.0)

    # Add AE risk score column
    if ae_scores_lookup:
        df["ae_risk_score"] = df["transaction_id"].map(
            lambda x: float(ae_scores_lookup.get(str(x), 0.0))
        )
    else:
        df["ae_risk_score"] = 0.0

    df["ae_risk_score"] = pd.to_numeric(df["ae_risk_score"], errors="coerce").fillna(0.0)
    df = df.sort_values("ae_risk_score", ascending=False).head(max_rows)

    return df


def render_transaction_heatmap(df: pd.DataFrame):
    """Render a transaction table with AE-risk colouring."""
    if df.empty:
        st.info("No transactions found.")
        return

    display_cols = [
        "transaction_id", "transaction_datetime", "amount",
        "merchant_category", "city", "ae_risk_score",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    show_df = df[display_cols].copy()

    # Format
    if "amount" in show_df.columns:
        show_df["amount"] = show_df["amount"].apply(lambda x: f"${x:,.2f}")
    if "ae_risk_score" in show_df.columns:
        show_df["ae_risk_score"] = show_df["ae_risk_score"].apply(lambda x: f"{x:.4f}")

    # Apply colour via HTML
    rows_html = ""
    for _, row in df.iterrows():
        ae_score = float(row.get("ae_risk_score", 0.0))
        bg_color = _ae_risk_color(ae_score)
        bg_opacity = f"{bg_color}22"  # 13% opacity
        row_style = f"background:{bg_opacity};"
        rows_html += "<tr style='" + row_style + "'>"
        for col in display_cols:
            val = row.get(col, "")
            if col == "amount":
                val = f"${float(val):,.2f}"
            elif col == "ae_risk_score":
                val = f"{float(val):.4f}"
                # Also show a mini colour dot
                val = f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{_ae_risk_color(ae_score)};margin-right:6px;vertical-align:middle;'></span>{val}"
            elif col == "transaction_datetime":
                val = str(pd.to_datetime(val, errors="coerce"))[:19] if pd.notna(val) else ""
            rows_html += f"<td style='padding:4px 8px;font-size:0.82rem;color:#e2e8f0;'>{val}</td>"
        rows_html += "</tr>"

    header_html = "".join(
        f"<th style='padding:6px 8px;color:#94a3b8;font-size:0.75rem;text-transform:uppercase;text-align:left;'>{c}</th>"
        for c in display_cols
    )

    st.markdown(
        f"""
        <style>
        .tx-heatmap {{ width:100%; border-collapse:collapse; }}
        .tx-heatmap tr:hover {{ filter: brightness(1.2); }}
        </style>
        <table class="tx-heatmap">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # Legend
    st.markdown(
        "<div style='display:flex;gap:12px;padding:4px 0;font-size:0.72rem;color:#94a3b8;'>"
        "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;margin-right:4px;vertical-align:middle;'></span>Normal</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:#ffcc00;margin-right:4px;vertical-align:middle;'></span>Suspicious</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:4px;vertical-align:middle;'></span>Highly anomalous</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Explanation text display ────────────────────────────────────────────────

def render_explanation_text(
    text: str,
    llm_status: str = "",
    risk_score: float = 0.0,
):
    """Render explanation text with LLM or deterministic styling.

    - LLM explanations get a green left border with 'AI EXPLANATION' label.
    - Deterministic fallback gets a risk-coloured left border with 'RULE-BASED' label.
    """
    is_llm = llm_status in ("ok",)
    border_color = "#22c55e" if is_llm else risk_color(risk_score)
    label = "🤖 AI EXPLANATION" if is_llm else "📋 RULE-BASED ANALYSIS"
    label_color = "#4ade80" if is_llm else "#94a3b8"

    st.markdown(
        f"""
        <div style="
            background:{'#0f2a1a' if is_llm else '#1e293b'};
            border-left:4px solid {border_color};
            border-radius:6px;
            padding:10px 14px;
            margin-bottom:10px;
            color:{'#bbf7d0' if is_llm else '#e2e8f0'};
            font-size:0.88rem;
            line-height:1.55;
        ">
            <div style="font-size:0.72rem;color:{label_color};margin-bottom:5px;font-weight:600;">
                {label}
            </div>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Full customer card ──────────────────────────────────────────────────────

def render_customer_card(
    customer_id: str,
    risk_score: float,
    risk_tier: str,
    cat_counts: dict[str, int] | list[str],
    city_counts: dict[str, int] | list[str],
    txns_df: pd.DataFrame,
    explain_text: str = "",
    llm_status: str = "",
    hub_importance: dict[str, float] | None = None,
    ae_scores_lookup: dict[str, float] | None = None,
    graph_height: str = "380px",
    show_heatmap_graph: bool = True,
    show_tx_heatmap: bool = True,
    show_explanation: bool = True,
):
    """Render the full customer card with graph heatmap + transaction heatmap + explanation.

    This is the reusable component used by both Model Output and Run Model pages.
    """
    border_color = risk_color(risk_score)

    # Header
    st.markdown(
        f"""
        <div style="
            border:1px solid {border_color}44;
            border-radius:12px;
            padding:6px 12px 2px 12px;
            background:#0f172a;
            margin-bottom:8px;
        ">
            <span style="font-size:0.75rem;color:#64748b;">
                Customer ID: <b style="color:#94a3b8;">{customer_id}</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk badge
    tier_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    icon = tier_icons.get(risk_tier, "⚪")
    bg = risk_bg(risk_score)
    color = risk_color(risk_score)
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border:2px solid {color};
            border-radius:12px;
            padding:16px 24px;
            display:flex;
            align-items:center;
            gap:24px;
            margin-bottom:12px;
        ">
            <div style="font-size:2.8rem;font-weight:700;color:{color};">
                {risk_score:.1%}
            </div>
            <div>
                <div style="font-size:1.1rem;color:#cbd5e1;">Risk Score</div>
                <div style="font-size:1.4rem;font-weight:600;color:{color};">
                    {icon} {risk_tier} RISK
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Side-by-side: graph (left) | explanation (right)
    col_graph, col_info = st.columns([3, 2], gap="medium")

    with col_graph:
        if show_heatmap_graph and (cat_counts or city_counts):
            n_cats = len(cat_counts)
            n_cities = len(city_counts)
            st.caption(
                f"Transaction connections: {n_cats} categor{'y' if n_cats == 1 else 'ies'}, "
                f"{n_cities} cit{'y' if n_cities == 1 else 'ies'}"
                f"{' · Hub nodes coloured by GNN importance' if hub_importance else ''}"
            )
            html = build_heatmap_graph(
                customer_id, risk_score, cat_counts, city_counts,
                hub_importance=hub_importance, height=graph_height,
            )
            _stcomp.html(html, height=int(graph_height.rstrip("px")), scrolling=False)
        else:
            st.caption("No graph neighborhood data available.")

    with col_info:
        if show_explanation and explain_text:
            render_explanation_text(explain_text, llm_status, risk_score)
        elif show_explanation:
            st.caption("No explanation available.")

    # Transaction heatmap
    if show_tx_heatmap and not txns_df.empty:
        with st.expander("Transaction History · Risk-Coloured", expanded=False):
            tx_heatmap_df = build_transaction_heatmap(txns_df, ae_scores_lookup)
            render_transaction_heatmap(tx_heatmap_df)
