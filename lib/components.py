"""
components.py — Modular AML display components.

Key design principles:
  * NarrativeEngine is an abstract base → swap RuleBasedNarrative for an LLM later
    by subclassing and overriding `generate()`.
  * CustomerInfoCard is the unified "box" that composes all sub-sections.
    Each section (risk badge, narrative, SHAP drivers, graph, KYC) is an
    independently renderable method so pages can mix and match.
  * GraphVisualizer builds PyVis HTML that can be embedded in Streamlit.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components
from pyvis.network import Network


def show_nav_logo(filename: str = "project_aegis.png", width: int = 40) -> bool:
    """Attempt to display a logo image in the Streamlit sidebar.

    Searches several likely locations under the repository root for the
    provided filename. If found, the image is displayed and the function
    returns True. If no file is found, a textual fallback is written to
    the sidebar and the function returns False.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base, filename),
        os.path.join(base, "assets", filename),
        os.path.join(base, "static", filename),
        os.path.join(base, "images", filename),
        os.path.join(base, "pages", filename),
        os.path.join(base, "lib", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                # Render as a sticky, centered element so it appears at the
                # top of the sidebar even when Streamlit inserts the page
                # navigation above normal sidebar content.
                with open(p, "rb") as f:
                    import base64

                    data = base64.b64encode(f.read()).decode("utf-8")

                img_html = (
                    "<div style='position:sticky;top:0;z-index:9999;"
                    "background:transparent;padding:8px 0;display:flex;justify-content:center;'>"
                    f"<img src=\"data:image/png;base64,{data}\" "
                    f"style=\"width:{width}px;max-width:100%;border-radius:8px;\"/>"
                    "</div>"
                )
                st.sidebar.markdown(img_html, unsafe_allow_html=True)
                return True
            except Exception:
                break

    # Fallback: simple centered title in sidebar if image isn't available
    st.sidebar.markdown("<div style='text-align:center;padding:8px 0;'><strong>Team 76 AML Detection</strong></div>", unsafe_allow_html=True)
    return False


def apply_sidebar_styles() -> None:
    """Inject shared sidebar nav CSS (purple text, light-purple highlight) into the page."""
    st.markdown("""
<style>
[data-testid="stSidebarNav"] a span {
    color: #7b68ee !important;
    font-size: 1.05rem !important;
}
[data-testid="stSidebarNav"] a[aria-selected="true"],
[data-testid="stSidebarNav"] li[aria-selected="true"] > a {
    background-color: #e8e0ff !important;
    border-left: 3px solid #7b68ee;
}
[data-testid="stSidebarNav"] a[aria-selected="true"] span,
[data-testid="stSidebarNav"] li[aria-selected="true"] > a span {
    color: #7b68ee !important;
}
[data-testid="stSidebarNav"] a:hover { background-color: #e8e0ff !important; }
[data-testid="stSidebarNav"] a:hover span { color: #7b68ee !important; }
</style>
""", unsafe_allow_html=True)


def header_with_logo(title: str, filename: str = "project_aegis.png", img_width: int = 220):
    """Render a purple page header title and apply shared sidebar styles."""
    apply_sidebar_styles()
    st.markdown(
        f"<h1 style='margin:0;padding:0;color:#7b68ee;font-size:2.4rem;'>{title}</h1>",
        unsafe_allow_html=True,
    )


@functools.lru_cache(maxsize=1)
def _load_industry_lookup() -> dict:
    """Return dict of industry_code (str) → industry_name (str)."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "kyc_industry_codes.csv.gz",
    )
    try:
        df = pd.read_csv(data_path, compression="gzip", dtype=str)
        return dict(zip(df["industry_code"], df["industry"]))
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Narrative Engines — swap for LLM by subclassing
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeEngine(ABC):
    """
    Abstract base for narrative generation.

    To plug in an LLM, subclass this and override `generate()`:

        class OpenAINarrative(NarrativeEngine):
            def generate(self, risk_tier, risk_score, drivers, context=None):
                prompt = build_prompt(risk_tier, risk_score, drivers, context)
                return openai_client.chat(prompt)

    Then pass your instance to CustomerInfoCard(narrative_engine=OpenAINarrative()).
    """

    @abstractmethod
    def generate(
        self,
        risk_tier: str,
        risk_score: float,
        drivers: list[dict],
        context: Optional[dict] = None,
    ) -> str:
        """
        Generate a human-readable narrative.

        Parameters
        ----------
        risk_tier : 'HIGH' | 'MEDIUM' | 'LOW'
        risk_score : float in [0, 1]
        drivers : list of dicts with keys 'description', 'shap_value'
        context : optional extra metadata (customer_id, kyc_data, etc.)

        Returns
        -------
        str  — narrative text (may contain markdown)
        """


class RuleBasedNarrative(NarrativeEngine):
    """
    Default, deterministic narrative matching model_output_explanations.csv format.
    No external dependencies.
    """

    # Geo-velocity threshold above which we flag geographic impossibility (cities/hr).
    # >3 cities/hr over a session is physically improbable in Canada.
    _GEO_VELOCITY_THRESHOLD = 3.0

    def generate(
        self,
        risk_tier: str,
        risk_score: float,
        drivers: list[dict],
        context: Optional[dict] = None,
    ) -> str:
        if risk_tier == "LOW":
            return (
                f"Customer presents a **LOW** risk profile (score {risk_score:.1%}). "
                "No significant behavioural anomalies detected."
            )

        # Detect geographic-impossibility pattern among the SHAP drivers
        geo_alert = ""
        for d in drivers:
            feat = d.get("feature", "")
            raw = d.get("raw_value")
            if feat == "geo_velocity" and raw is not None and float(raw) > self._GEO_VELOCITY_THRESHOLD:
                geo_alert = (
                    f"  \n🌍 **Geographic impossibility detected**: transactions span "
                    f"**{int(round(float(raw)))} cities/hr** —"
                    f" physically impossible without simultaneous card use."
                )
                break

        driver_texts = [d.get("description", d.get("feature", "")) for d in drivers]
        drivers_str = ", ".join(driver_texts) if driver_texts else "unspecified factors"

        if risk_tier == "HIGH":
            heading = f"⚠️ **HIGH RISK** — score {risk_score:.1%}"
            action = "Recommend **immediate case review** and enhanced due-diligence."
        else:
            heading = f"⚡ **MEDIUM RISK** — score {risk_score:.1%}"
            action = "Recommend **monitoring** and secondary review."

        return (
            f"{heading}.  \nPrimary behavioural drivers: **{drivers_str}**.{geo_alert}  \n{action}"
        )


class PrecomputedNarrative(NarrativeEngine):
    """
    Returns a pre-computed narrative string as-is (used by Model Output page
    where narratives are already stored in model_output_explanations.csv).
    """

    def generate(
        self,
        risk_tier: str,
        risk_score: float,
        drivers: list[dict],
        context: Optional[dict] = None,
    ) -> str:
        if context and "narrative" in context:
            return context["narrative"]
        return RuleBasedNarrative().generate(risk_tier, risk_score, drivers, context)


# ─────────────────────────────────────────────────────────────────────────────
# Risk colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def risk_color(risk_score: float) -> str:
    """Return a hex colour interpolated green→amber→red."""
    if risk_score < 0.40:
        return "#22c55e"   # green
    if risk_score < 0.70:
        return "#f59e0b"   # amber
    return "#ef4444"       # red


def risk_bg(risk_score: float) -> str:
    if risk_score < 0.40:
        return "#14532d"
    if risk_score < 0.70:
        return "#78350f"
    return "#7f1d1d"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class GraphVisualizer:
    """
    Builds an interactive PyVis graph for a customer profile.

    Two modes:
      * live_graph   — called from Run Model with a list of transaction dicts.
                       Nodes: Customer → (Category, City) connected by labeled edges.
      * driver_graph — called from Model Output with SHAP driver dicts.
                       Nodes: Customer → FeatureDriver nodes.
    """

    # ── Live graph (Run Model page) ───────────────────────────────────────────

    @staticmethod
    def build_live_graph(
        customer_id: str,
        risk_score: float,
        transactions: list[dict],
        height: str = "420px",
    ) -> str:
        """
        Build HTML for customer + transaction graph.

        Each transaction is an edge (labeled) from Customer → MerchantCategory
        and Customer → City.  Multiple transactions to the same hub merge into
        one hub node with aggregated tooltip.
        """
        G = nx.MultiDiGraph()
        c_color = risk_color(risk_score)
        cust_label = f"Customer\n{customer_id[:14]}"
        cust_tip = (
            f"CUSTOMER\nID: {customer_id}\n"
            f"Risk Score: {risk_score:.1%}"
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

        # Track hub aggregates
        cat_stats: dict[str, dict] = {}
        city_stats: dict[str, dict] = {}
        ind_lookup = _load_industry_lookup()

        for i, tx in enumerate(transactions):
            amount = tx.get("amount", 0.0)
            cat = str(tx.get("merchant_category", "Unknown"))
            city = str(tx.get("city", "Unknown"))
            tx_type = tx.get("txn_type", "")
            if not tx_type:
                is_cash = bool(tx.get("cash_indicator", False))
                is_ecom = bool(tx.get("ecommerce_ind", False))
                tx_type = "ATM" if is_cash else ("E-com" if is_ecom else "CARD")
            edge_label = f"${amount:,.0f}\n{tx_type}"

            # Accumulate hub stats
            for hub_name, stats_dict in [(cat, cat_stats), (city, city_stats)]:
                if hub_name not in stats_dict:
                    stats_dict[hub_name] = {"count": 0, "total": 0.0, "types": []}
                stats_dict[hub_name]["count"] += 1
                stats_dict[hub_name]["total"] += amount
                stats_dict[hub_name]["types"].append(tx_type)

            # Edge customer → category
            G.add_edge(
                customer_id,
                f"cat::{cat}",
                label=edge_label,
                color="#94a3b8",
                width=1.5,
            )
            # Edge customer → city
            G.add_edge(
                customer_id,
                f"city::{city}",
                label=edge_label,
                color="#64748b",
                width=1.5,
            )

        # Category hub nodes
        for cat, stats in cat_stats.items():
            node_id = f"cat::{cat}"
            # Resolve industry code → name; fall back to the code itself
            cat_name = ind_lookup.get(cat, cat)
            display = cat_name if cat_name != cat else cat
            tip = (
                f"MERCHANT CATEGORY\n"
                f"{cat} \u2013 {cat_name}\n"
                f"Transactions: {stats['count']}\n"
                f"Total: ${stats['total']:,.2f}"
            )
            # Label: code + shortened name
            if cat_name != cat:
                short = f"{cat}\n{cat_name[:20] + chr(8230) if len(cat_name) > 20 else cat_name}"
            else:
                short = cat[:18] + chr(8230) if len(cat) > 18 else cat
            G.add_node(
                node_id,
                label=short,
                color="#3b82f6",
                size=20,
                title=tip,
                shape="dot",
            )

        # City hub nodes
        for city, stats in city_stats.items():
            node_id = f"city::{city}"
            tip = (
                f"CITY: {city}\n"
                f"Transactions: {stats['count']}\n"
                f"Total: ${stats['total']:,.2f}"
            )
            short = city[:18] + chr(8230) if len(city) > 18 else city
            G.add_node(
                node_id,
                label=short,
                color="#f97316",
                size=20,
                title=tip,
                shape="dot",
            )

        return GraphVisualizer._render_pyvis(G, height=height)

    # ── Neighborhood graph (Model Output page) ───────────────────────────────
    # Shows the training-graph neighborhood of a customer: which merchant
    # category and city hub nodes they are connected to via their historical
    # transactions.  No per-transaction amounts (not stored in the output CSV)
    # — only the unique hubs are shown.

    @staticmethod
    def build_neighborhood_graph(
        customer_id: str,
        risk_score: float,
        categories: "dict[str, int] | list[str]",
        cities: "dict[str, int] | list[str]",
        height: str = "380px",
    ) -> str:
        """
        Build an HTML graph showing customer ↔ merchant-category and city hubs.

        `categories` / `cities` may be:
          - a dict  {name: transaction_count}  (preferred — shows richer info)
          - a list  [name, ...]                (treated as count=1 each)
        """
        # Normalise to dicts so the rest of the logic is uniform
        if isinstance(categories, list):
            cat_counts: dict[str, int] = {c: 1 for c in categories}
        else:
            cat_counts = dict(categories)
        if isinstance(cities, list):
            city_counts: dict[str, int] = {c: 1 for c in cities}
        else:
            city_counts = dict(cities)

        G = nx.Graph()
        c_color = risk_color(risk_score)
        ind_lookup = _load_industry_lookup()

        total_txns = sum(cat_counts.values())
        cust_label = f"Customer\n{customer_id[:14]}"
        cust_tip = (
            f"CUSTOMER\n"
            f"ID: {customer_id}\n"
            f"Risk Score: {risk_score:.1%}\n"
            f"Merchant Categories: {len(cat_counts)}\n"
            f"Cities: {len(city_counts)}\n"
            f"Total Transaction Edges: {total_txns}"
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
            cat_name = ind_lookup.get(cat, cat)
            if cat_name != cat:
                short = f"{cat}\n{cat_name[:20] + chr(8230) if len(cat_name) > 20 else cat_name}"
                tip = (
                    f"MERCHANT CATEGORY\n"
                    f"Code: {cat}\n"
                    f"Name: {cat_name}\n"
                    f"Transactions: {tx_count}"
                )
                edge_tip = f"Transactions: {tx_count}\nCategory: {cat} \u2013 {cat_name}"
            else:
                short = cat[:18] + chr(8230) if len(cat) > 18 else cat
                tip = (
                    f"MERCHANT CATEGORY\n"
                    f"{cat}\n"
                    f"Transactions: {tx_count}"
                )
                edge_tip = f"Transactions: {tx_count}\nCategory: {cat}"
            node_size = max(14, min(32, 14 + tx_count * 2))
            edge_width = max(1.5, min(8.0, 1.5 + tx_count * 0.6))
            G.add_node(node_id, label=short, color="#3b82f6", size=node_size, title=tip, shape="dot")
            G.add_edge(
                customer_id, node_id,
                color="#3b82f6",
                width=edge_width,
                title=edge_tip,
                label=str(tx_count) if tx_count > 1 else "",
            )

        for city, tx_count in city_counts.items():
            node_id = f"city::{city}"
            short = city[:18] + chr(8230) if len(city) > 18 else city
            tip = (
                f"CITY\n"
                f"{city}\n"
                f"Transactions: {tx_count}"
            )
            edge_tip = f"Transactions: {tx_count}\nCity: {city}"
            node_size = max(14, min(32, 14 + tx_count * 2))
            edge_width = max(1.5, min(8.0, 1.5 + tx_count * 0.6))
            G.add_node(node_id, label=short, color="#f97316", size=node_size, title=tip, shape="dot")
            G.add_edge(
                customer_id, node_id,
                color="#f97316",
                width=edge_width,
                title=edge_tip,
                label=str(tx_count) if tx_count > 1 else "",
            )

        return GraphVisualizer._render_pyvis(G, height=height)

    # ── Driver graph (Model Output page) ─────────────────────────────────────

    @staticmethod
    def build_driver_graph(
        customer_id: str,
        risk_score: float,
        drivers: list[dict],
        height: str = "340px",
    ) -> str:
        """
        Build HTML for customer + SHAP driver feature nodes.
        Node size = |SHAP value|; color = red (risk-raising) / blue (risk-reducing).
        """
        G = nx.Graph()
        c_color = risk_color(risk_score)
        cust_tip = (
            f"CUSTOMER\nID: {customer_id}\n"
            f"Risk Score: {risk_score:.1%}"
        )
        short_id = customer_id[:16] + chr(8230) if len(customer_id) > 16 else customer_id
        G.add_node(
            customer_id,
            label=f"Customer\n{short_id}",
            color=c_color,
            size=30,
            title=cust_tip,
        )

        for d in drivers:
            shap_val = float(d.get("shap_value", 0.0))
            desc = d.get("description", d.get("feature", ""))
            feat = d.get("feature", desc)
            raw = d.get("raw_value", None)

            node_color = "#ef4444" if shap_val > 0 else "#3b82f6"
            node_size = max(10, min(28, abs(shap_val) * 600))
            tip = (
                f"{desc}\n"
                f"SHAP: {shap_val:+.4f}\n"
                + (f"Value: {raw:.2f}" if raw is not None else "")
            )
            short_feat = desc[:20] + chr(8230) if len(desc) > 20 else desc
            G.add_node(
                feat,
                label=short_feat,
                color=node_color,
                size=node_size,
                title=tip,
            )
            G.add_edge(customer_id, feat, color="#475569", width=1)

        return GraphVisualizer._render_pyvis(G, height=height)

    # ── Internal renderer ─────────────────────────────────────────────────────

    @staticmethod
    def _render_pyvis(G: nx.Graph, height: str = "400px") -> str:
        net = Network(
            height=height,
            width="100%",
            bgcolor="#1e293b",
            font_color="white", # type: ignore
            notebook=False,
            cdn_resources="in_line",
        )
        net.from_nx(G)
        net.set_options(
            json.dumps(
                {
                    "interaction": {
                        "hover": True,
                        "tooltipDelay": 80,
                        "navigationButtons": False,
                        "keyboard": False,
                    },
                    "physics": {
                        "enabled": True,
                        "barnesHut": {
                            "gravitationalConstant": -18000,
                            "springLength": 120,
                            "springConstant": 0.04,
                            "damping": 0.15,
                        },
                        "minVelocity": 0.75,
                        "stabilization": {"enabled": True, "iterations": 80},
                    },
                    "edges": {
                        "smooth": {"enabled": True, "type": "dynamic"},
                        "font": {"size": 11, "color": "#cbd5e1", "strokeWidth": 2, "strokeColor": "#1e293b"},
                    },
                }
            )
        )
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as f:
            tmp_path = f.name
        net.save_graph(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp_path)
        # Disable physics after the graph stabilises so the simulation
        # doesn't keep burning CPU in every embedded iframe.
        disable_physics_js = (
            "<script>"
            "window.addEventListener('load', function() {"
            "  var checkNet = setInterval(function() {"
            "    if (typeof network !== 'undefined') {"
            "      clearInterval(checkNet);"
            "      network.once('stabilized', function() {"
            "        network.setOptions({ physics: { enabled: false } });"
            "      });"
            "    }"
            "  }, 50);"
            "});"
            "</script>"
        )
        html = html.replace("</body>", disable_physics_js + "</body>")
        return html


# ─────────────────────────────────────────────────────────────────────────────
# CustomerInfoCard — the primary display component
# ─────────────────────────────────────────────────────────────────────────────

class CustomerInfoCard:
    """
    Unified display box for a customer's AML risk profile.

    Usage
    -----
    # With default rule-based narrative:
    card = CustomerInfoCard()

    # With a future LLM narrative:
    card = CustomerInfoCard(narrative_engine=OpenAINarrative())

    # Render full card:
    card.render(customer_id=..., risk_score=..., risk_tier=...,
                drivers=[...], transactions=[...])

    # Or render individual sections:
    card.render_risk_badge(risk_score, risk_tier)
    card.render_narrative(risk_tier, risk_score, drivers)
    card.render_drivers(drivers)
    card.render_graph_live(customer_id, risk_score, transactions)
    card.render_kyc(kyc_dict)
    """

    def __init__(self, narrative_engine: Optional[NarrativeEngine] = None):
        self.narrative_engine: NarrativeEngine = narrative_engine or RuleBasedNarrative()

    # ── Individual section renderers ──────────────────────────────────────────

    def render_risk_badge(self, risk_score: float, risk_tier: str):
        """Prominent risk score metric with colour-coded tier."""
        color = risk_color(risk_score)
        bg = risk_bg(risk_score)
        tier_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        icon = tier_icons.get(risk_tier, "⚪")
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

    def render_narrative(
        self,
        risk_tier: str,
        risk_score: float,
        drivers: list[dict],
        context: Optional[dict] = None,
    ):
        """Narrative text block generated by the NarrativeEngine."""
        text = self.narrative_engine.generate(risk_tier, risk_score, drivers, context)
        st.markdown(
            f"""
            <div style="
                background:#1e293b;
                border-left:4px solid {risk_color(risk_score)};
                border-radius:6px;
                padding:14px 18px;
                margin-bottom:12px;
                color:#e2e8f0;
                font-size:0.95rem;
                line-height:1.6;
            ">
                {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_drivers(self, drivers: list[dict]):
        """SHAP driver table with a mini horizontal bar per driver."""
        if not drivers:
            return
        st.markdown("**Top Risk Drivers (SHAP)**")
        for d in drivers:
            feat = d.get("feature", "")
            desc = d.get("description", feat)
            shap_val = float(d.get("shap_value", 0.0))
            bar_pct = min(100, int(abs(shap_val) * 1200))
            bar_color = "#ef4444" if shap_val > 0 else "#3b82f6"
            raw = d.get("raw_value", None)
            raw_str = f" = {raw:.2f}" if raw is not None else ""

            # Flag geographic impossibility inline
            geo_badge = ""
            if feat == "geo_velocity" and raw is not None and float(raw) > RuleBasedNarrative._GEO_VELOCITY_THRESHOLD:
                geo_badge = (
                    f" &nbsp;<span style='background:#7f1d1d;color:#fca5a5;"
                    f"border-radius:4px;padding:1px 6px;font-size:0.75rem;'>🌍 geo-impossible</span>"
                )

            st.markdown(
                f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.85rem;color:#94a3b8;margin-bottom:3px;">
                        <span>{desc}{raw_str}{geo_badge}</span>
                        <span style="color:{bar_color};font-weight:600;">
                            {shap_val:+.4f}
                        </span>
                    </div>
                    <div style="background:#334155;border-radius:4px;height:6px;">
                        <div style="width:{bar_pct}%;background:{bar_color};
                                    height:6px;border-radius:4px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def render_graph_live(
        self,
        customer_id: str,
        risk_score: float,
        transactions: list[dict],
        height: str = "420px",
    ):
        """Interactive graph for the Run Model page (live transaction input)."""
        html = GraphVisualizer.build_live_graph(
            customer_id, risk_score, transactions, height=height
        )
        st_components.html(html, height=int(height.rstrip("px")), scrolling=False)

    def render_graph_drivers(
        self,
        customer_id: str,
        risk_score: float,
        drivers: list[dict],
        height: str = "340px",
    ):
        """Interactive graph for the Model Output page (SHAP driver nodes)."""
        html = GraphVisualizer.build_driver_graph(
            customer_id, risk_score, drivers, height=height
        )
        st_components.html(html, height=int(height.rstrip("px")), scrolling=False)

    def render_kyc(self, kyc: dict):
        """KYC summary table.

        For business accounts only account-level fields are shown (annual sales,
        employee count, account tenure). Personal fields (age, income) are not
        displayed because they are not collected / are masked for business KYC.
        """
        if not kyc:
            return

        is_biz = bool(kyc.get("is_biz") or kyc.get("is_business"))

        # Fields shown for ALL account types
        common_labels = {
            "is_biz": "Account Type",
            "is_business": "Account Type",
            "tenure": "Account Tenure (days)",
            "tenure_days": "Account Tenure (days)",
        }
        # Fields shown only for individuals
        individual_labels = {
            "age": "Age",
            "income": "Annual Income",
        }
        # Fields shown only for businesses
        business_labels = {
            "sales": "Annual Sales",
            "annual_sales": "Annual Sales",
            "emp_count": "Employees",
            "employee_count": "Employees",
        }

        if is_biz:
            labels = {**common_labels, **business_labels}
        else:
            labels = {**common_labels, **individual_labels}

        rows_html = ""
        seen_labels: set[str] = set()
        for k, label in labels.items():
            if k not in kyc:
                continue
            if label in seen_labels:
                continue
            seen_labels.add(label)
            val = kyc[k]
            if k in ("is_biz", "is_business"):
                val = "Business" if val else "Individual"
            elif k in ("income", "sales", "annual_sales"):
                val = f"${float(val):,.0f}"
            elif isinstance(val, float):
                val = f"{val:,.1f}"
            rows_html += (
                f"<tr>"
                f"<td style='color:#94a3b8;padding:4px 8px;'>{label}</td>"
                f"<td style='color:#e2e8f0;padding:4px 8px;font-weight:500;'>{val}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
                {rows_html}
            </table>
            """,
            unsafe_allow_html=True,
        )

    # ── Full card ─────────────────────────────────────────────────────────────

    def render(
        self,
        customer_id: str,
        risk_score: float,
        risk_tier: str,
        drivers: list[dict],
        transactions: Optional[list[dict]] = None,
        kyc: Optional[dict] = None,
        narrative_context: Optional[dict] = None,
        graph_mode: str = "live",   # "live" | "drivers" | "none"
        show_kyc: bool = True,
        show_drivers: bool = True,
        show_graph: bool = True,
        graph_height: str = "400px",
    ):
        """
        Render the full customer info card inside a bordered container.

        graph_mode:
          'live'    — uses transaction edges (Run Model page)
          'drivers' — uses SHAP driver nodes (Model Output page)
          'none'    — skip graph
        """
        border_color = risk_color(risk_score)
        with st.container():
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
            self.render_risk_badge(risk_score, risk_tier)

            col_left, col_right = st.columns([3, 2], gap="medium")
            with col_left:
                if show_graph and graph_mode != "none":
                    if graph_mode == "live":
                        self.render_graph_live(
                            customer_id, risk_score, transactions or [], height=graph_height
                        )
                    else:
                        self.render_graph_drivers(
                            customer_id, risk_score, drivers, height=graph_height
                        )
            with col_right:
                ctx = narrative_context or {}
                ctx.setdefault("customer_id", customer_id)
                self.render_narrative(risk_tier, risk_score, drivers, context=ctx)
                if show_drivers:
                    self.render_drivers(drivers)
                if show_kyc and kyc:
                    st.markdown("**KYC Profile**")
                    self.render_kyc(kyc)
