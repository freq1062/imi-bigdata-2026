"""
3_Model_Output.py — Review pre-computed AML scores for all KYC customers.

Features:
  ▸ Database stats: total customers, flagged, graph edges
  ▸ Top-K most suspicious customers with full info cards
  ▸ Human review: mark each customer as "Confirmed Fraud" or "Clear"
  ▸ Search by customer_id or transaction_id
  ▸ CustomerInfoCard (driver-graph mode) — no recompute needed,
    all data comes from model_output.csv + model_output_explanations.csv
"""

import sys
import os
import pickle

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd

from lib.components import CustomerInfoCard, PrecomputedNarrative, GraphVisualizer, _load_industry_lookup
import streamlit.components.v1 as st_components

st.set_page_config(
    page_title="Model Output",
    layout="wide",
    initial_sidebar_state="expanded",
)

# (sidebar logo removed — header displays logo on Home page)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading model output…")
def _load_output():
    df = pd.read_csv("model_output.csv")
    return df

@st.cache_data(show_spinner="Loading explanations…")
def _load_explanations():
    df = pd.read_csv("model_output_explanations.csv")
    return df

# Edge count: try to pull from sage artifacts, else fall back to a csv if available
@st.cache_data(show_spinner=False)
def _get_edge_count():
    try:
        with open("sage_artifacts.pkl", "rb") as f:
            arts = pickle.load(f)
        ei_cc = arts.get("edge_cust_cat")
        ei_ct = arts.get("edge_cust_city")
        n = 0
        if ei_cc is not None:
            n += ei_cc.shape[1]
        if ei_ct is not None:
            n += ei_ct.shape[1]
        return n
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_graph_data():
    """
    Load edge arrays and reverse maps from sage_artifacts so we can reconstruct
    each customer's merchant-category and city neighborhood for the graph view.
    Arrays are stored as plain numpy to keep the cache serialisable.
    """
    try:
        with open("sage_artifacts.pkl", "rb") as f:
            arts = pickle.load(f)
        return {
            "cust_map": arts["cust_map"],
            "rev_cat": {v: k for k, v in arts["cat_map"].items()},
            "rev_city": {v: k for k, v in arts["city_map"].items()},
            "ei_cc_cust": arts["edge_cust_cat"][0].numpy(),
            "ei_cc_cat":  arts["edge_cust_cat"][1].numpy(),
            "ei_ct_cust": arts["edge_cust_city"][0].numpy(),
            "ei_ct_city": arts["edge_cust_city"][1].numpy(),
        }
    except Exception:
        return None


def _get_customer_neighbors(cid: str, gdata: dict | None) -> tuple[dict[str, int], dict[str, int]]:
    """
    Return ({category: tx_count}, {city: tx_count}) for a customer from the training graph.
    Counts reflect the number of edges (transactions) in the heterogeneous graph.
    """
    if gdata is None:
        return {}, {}
    nidx = gdata["cust_map"].get(cid)
    if nidx is None:
        return {}, {}
    mask_cc = gdata["ei_cc_cust"] == nidx
    cat_counts: dict[str, int] = {}
    for cat_idx in gdata["ei_cc_cat"][mask_cc]:
        name = gdata["rev_cat"].get(int(cat_idx), str(cat_idx))
        cat_counts[name] = cat_counts.get(name, 0) + 1
    mask_ct = gdata["ei_ct_cust"] == nidx
    city_counts: dict[str, int] = {}
    for city_idx in gdata["ei_ct_city"][mask_ct]:
        name = gdata["rev_city"].get(int(city_idx), str(city_idx))
        city_counts[name] = city_counts.get(name, 0) + 1
    return cat_counts, city_counts


# ─────────────────────────────────────────────────────────────────────────────
# Standard Merchant Category Code (MCC) descriptions
# Complements kyc_industry_codes.csv (SIC/NAICS) for card transaction MCCs.
# ─────────────────────────────────────────────────────────────────────────────

_MCC_LOOKUP: dict[str, str] = {
    # Airlines
    "3000": "United Airlines", "3001": "American Airlines", "3002": "Air Canada",
    "3005": "British Airways", "3020": "Air France", "3075": "Singapore Airlines",
    "3099": "Other Airlines",
    # Car Rental
    "3351": "Enterprise Rent-A-Car", "3352": "National Car Rental",
    "3395": "Hertz", "3398": "Budget Rent-A-Car",
    # Hotels / Lodging
    "3501": "Holiday Inns", "3509": "Marriott Hotels", "3516": "Hilton Hotels",
    "3640": "Best Western Hotels", "3641": "Quality Inns", "3692": "Comfort Inns",
    "3722": "Sheraton Hotels", "3732": "Radisson Hotels", "3752": "Delta Hotels",
    # Transportation
    "4011": "Railroads", "4111": "Local Transit", "4112": "Passenger Railways",
    "4121": "Taxicabs and Limousines", "4131": "Bus Lines",
    "4214": "Motor Freight Carriers", "4411": "Cruise Lines", "4511": "Airlines",
    "4722": "Travel Agencies and Tour Operators", "4784": "Tolls / Bridge Fees",
    # Telecom / Internet
    "4812": "Telephone Equipment", "4813": "Telephone Services",
    "4814": "Telephone Services", "4816": "Computer Networking / Internet",
    "4829": "Wire Transfers / Money Orders",
    "4899": "Cable, Satellite and Pay TV",
    "4900": "Utilities – Electric, Gas, Water",
    # Building / Wholesale
    "5200": "Home Supply Warehouses", "5211": "Lumber Stores",
    "5231": "Glass / Paint / Wallpaper", "5251": "Hardware Stores",
    "5261": "Lawn and Garden Supply",
    "5300": "Wholesale Clubs", "5309": "Duty-Free Stores",
    "5310": "Discount Stores", "5311": "Department Stores",
    "5331": "Variety Stores", "5399": "Miscellaneous General Merchandise",
    # Food / Grocery
    "5411": "Grocery Stores / Supermarkets", "5422": "Meat and Fish Markets",
    "5441": "Candy and Confectionery", "5451": "Dairy Products",
    "5462": "Bakeries", "5499": "Miscellaneous Food Stores",
    # Automotive
    "5511": "Car Dealerships (New / Used)", "5521": "Used Car Dealerships",
    "5531": "Auto and Home Supply Stores", "5532": "Automotive Tire Stores",
    "5533": "Automotive Parts / Accessories", "5541": "Service Stations – Gasoline",
    "5542": "Automated Fuel Dispensers", "5551": "Boat Dealers",
    "5571": "Motorcycle Dealers", "5581": "Recreational Vehicle Dealers",
    "5599": "Automotive – Miscellaneous Dealers",
    # Clothing
    "5611": "Men's Clothing", "5621": "Women's Clothing",
    "5631": "Women's Accessories", "5641": "Children's Apparel",
    "5651": "Family Clothing Stores", "5655": "Sports Apparel",
    "5661": "Shoe Stores", "5681": "Furriers / Fur Shops",
    "5691": "Men's and Women's Clothing", "5699": "Miscellaneous Apparel",
    # Home Furnishings / Electronics
    "5712": "Furniture and Home Furnishings", "5713": "Floor Covering Stores",
    "5719": "Miscellaneous Home Furnishings",
    "5722": "Household Appliance Stores",
    "5731": "Electronics Stores", "5732": "Electronics Stores",
    "5733": "Music Stores", "5734": "Computer Software Stores",
    "5735": "Record / CD / Tape Stores",
    # Restaurants
    "5811": "Caterers", "5812": "Restaurants / Eating Places",
    "5813": "Bars / Taverns / Nightclubs", "5814": "Fast Food Restaurants",
    "5815": "Digital Goods – Media", "5816": "Digital Goods – Games",
    "5817": "Digital Goods – Apps", "5818": "Digital Goods – General",
    # Pharmacy / Health / Specialty Retail
    "5912": "Drug Stores and Pharmacies", "5921": "Liquor Stores",
    "5940": "Sporting Goods Stores", "5941": "Sporting Goods Stores",
    "5942": "Book Stores", "5943": "Office / School Supply Stores",
    "5944": "Jewelry Stores", "5945": "Toy and Game Shops",
    "5946": "Camera and Photographic Stores",
    "5947": "Gift / Card / Novelty / Souvenir",
    "5948": "Luggage and Leather Goods", "5977": "Cosmetics / Beauty Supplies",
    "5992": "Florists", "5993": "Cigar Stores", "5995": "Pet Shops / Supplies",
    "5960": "Direct Marketing – Insurance", "5961": "Catalog / Mail Order",
    "5964": "Direct Marketing – Catalog", "5965": "Direct Marketing – Combo Catalog",
    "5966": "Direct Marketing – Outbound Telemarketing",
    "5967": "Direct Marketing – Inbound Telemarketing",
    "5968": "Direct Marketing – Subscription",
    "5969": "Other Direct Marketing",
    "5999": "Miscellaneous Retail",
    # Finance / Banking
    "6010": "Financial Institutions – Cash",
    "6011": "Financial Institutions – ATM",
    "6051": "Currency Exchange / Travellers' Cheques",
    "6211": "Securities Brokers / Dealers",
    "6300": "Insurance Sales", "6381": "Insurance Premiums",
    "6399": "Insurance – Other", "6411": "Insurance Agents / Brokers",
    "6513": "Real Estate Agents and Managers", "6540": "POI Funding",
    # Hotels / Personal Services
    "7011": "Hotels and Motels", "7012": "Timeshares",
    "7032": "Sporting and Recreational Camps",
    "7033": "Trailer Parks / Campgrounds",
    "7210": "Laundry / Cleaning", "7211": "Laundry Services",
    "7216": "Dry Cleaners", "7217": "Carpet and Upholstery Cleaning",
    "7221": "Photographic Studios", "7230": "Barbers and Beauty Shops",
    "7251": "Shoe Repair", "7261": "Funeral Services",
    "7276": "Tax Preparation", "7277": "Counselling Services",
    "7297": "Massage Parlors", "7298": "Health and Beauty Spas",
    "7299": "Miscellaneous Personal Services",
    # Business Services
    "7311": "Advertising Services", "7321": "Consumer Credit Reporting",
    "7338": "Quick Copy / Reproduction", "7342": "Exterminating / Disinfecting",
    "7349": "Cleaning / Maintenance / Janitorial",
    "7361": "Temporary Labour and Staffing",
    "7372": "Computer Programming / Software",
    "7374": "Computer Processing / Data Preparation",
    "7375": "Computer Information Retrieval",
    "7379": "Computer Maintenance / Repair",
    "7392": "Management Consulting", "7393": "Detective / Protective Services",
    "7394": "Equipment Rental / Leasing", "7399": "Business Services – Other",
    # Automotive Services
    "7512": "Car Rental", "7513": "Truck and Trailer Rental",
    "7523": "Parking Lots / Garages", "7531": "Auto Body Repair",
    "7534": "Tire Retreading / Repair", "7538": "General Auto Repair",
    "7542": "Car Washes", "7549": "Towing Services",
    # Repair / Entertainment
    "7622": "Electronics Repair", "7629": "Small Appliance Repair",
    "7699": "Repair Shops – Other",
    "7832": "Movie Theatres", "7841": "Video Rental",
    "7911": "Dance Halls / Studios",
    "7922": "Theatrical Producers / Ticket Agencies",
    "7933": "Bowling Alleys", "7941": "Professional Sports",
    "7991": "Tourist Attractions", "7992": "Golf Courses",
    "7994": "Video Game Arcades", "7995": "Betting / Lottery Tickets",
    "7996": "Amusement Parks / Circuses", "7999": "Recreation Services",
    # Healthcare
    "8011": "Doctors / Physicians", "8021": "Dentists / Orthodontists",
    "8031": "Osteopaths", "8041": "Chiropractors",
    "8042": "Optometrists / Ophthalmologists", "8043": "Optical Goods",
    "8050": "Nursing / Personal Care Facilities", "8062": "Hospitals",
    "8071": "Medical / Dental Laboratories", "8099": "Health Practitioners",
    # Education
    "8211": "Elementary / Secondary Schools",
    "8220": "Colleges / Universities",
    "8249": "Vocational / Trade Schools", "8299": "Schools – Other",
    # Non-Profit / Government
    "8398": "Charitable / Social Service Organizations",
    "8641": "Civic / Social / Fraternal Associations",
    "8661": "Religious Organizations",
    "8699": "Membership Organizations",
    "8742": "Management Consulting", "8743": "Public Relations",
    "8911": "Architectural / Engineering Services",
    "8931": "Accounting / Bookkeeping", "8999": "Professional Services",
    "9222": "Fines", "9311": "Tax Payments",
    "9399": "Government Services", "9402": "Postal Services",
}


# ─────────────────────────────────────────────────────────────────────────────
# Transaction data loading (per customer, cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=50)
def _load_customer_transactions(cid: str) -> pd.DataFrame:
    """
    Scan all raw transaction CSV files in chunks and collect rows for this customer.
    Cached per customer_id (LRU, up to 50 entries).
    Returns a normalised DataFrame ready to display.
    """
    data_dir = os.path.join(_ROOT, "data")
    # Build a combined code→name lookup: MCC codes first, then kyc_industry_codes
    kyc_lookup = _load_industry_lookup()
    ind_lookup = {**kyc_lookup, **_MCC_LOOKUP}  # MCC names win on overlap

    # (filename, display type, has_location, has_ecommerce, has_cash)
    specs = [
        ("card.csv.gz",   "CARD",   True,  True,  False),
        ("abm.csv.gz",    "ATM",    True,  False, True),
        ("eft.csv.gz",    "EFT",    False, False, False),
        ("emt.csv.gz",    "EMT",    False, False, False),
        ("wire.csv.gz",   "WIRE",   False, False, False),
        ("cheque.csv.gz", "CHEQUE", False, False, False),
    ]

    dfs: list[pd.DataFrame] = []
    for fname, txn_type, has_loc, has_ecom, has_cash in specs:
        path = os.path.join(data_dir, fname)
        try:
            chunks = []
            for chunk in pd.read_csv(
                path, compression="gzip", chunksize=100_000, low_memory=False
            ):
                matched = chunk[chunk["customer_id"] == cid]
                if not matched.empty:
                    chunks.append(matched)
            if not chunks:
                continue
            df = pd.concat(chunks, ignore_index=True)

            df["txn_type"] = txn_type
            df["amount"] = pd.to_numeric(df.get("amount_cad"), errors="coerce")
            df["datetime"] = pd.to_datetime(
                df.get("transaction_datetime"), errors="coerce"
            )
            df["dr_cr"] = df.get("debit_credit", pd.Series(dtype=str)).str.upper().map(
                {"D": "Debit", "C": "Credit"}
            ).fillna("")

            # Merchant category: resolve code → human name via MCC + kyc lookup.
            # Card CSV numeric codes may be float dtype (e.g. 5812.0), so
            # normalise to bare int string before lookup.
            if "merchant_category" in df.columns:
                raw_cat = df["merchant_category"]
                if pd.api.types.is_numeric_dtype(raw_cat):
                    code_str = (
                        raw_cat.fillna(-1).astype(int).astype(str)
                        .replace("-1", "")
                    )
                else:
                    code_str = (
                        raw_cat.fillna("").astype(str).str.split(".").str[0]
                    )
                df["category"] = code_str.apply(
                    lambda c: ind_lookup[c] if c in ind_lookup else (c if c else "")
                )
            else:
                # Non-card types have no merchant category; txn_type carries the type
                df["category"] = ""

            df["city"] = (
                df["city"].fillna("").str.strip().str.title()
                if has_loc and "city" in df.columns
                else ""
            )
            df["province"] = (
                df["province"].fillna("").str.upper()
                if has_loc and "province" in df.columns
                else ""
            )
            df["ecommerce"] = (
                df["ecommerce_ind"].astype(str).map({"1": "Yes", "0": "", "1.0": "Yes", "0.0": ""})
                .fillna("")
                if has_ecom and "ecommerce_ind" in df.columns
                else ""
            )
            df["cash"] = (
                df["cash_indicator"].astype(str).map({"1": "Yes", "0": "", "1.0": "Yes", "0.0": ""})
                .fillna("")
                if has_cash and "cash_indicator" in df.columns
                else ""
            )

            keep = ["transaction_id", "datetime", "amount", "dr_cr",
                    "txn_type", "category", "city", "province", "ecommerce", "cash"]
            dfs.append(df[keep])
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("datetime", na_position="last").reset_index(drop=True)
    return result

output_df = _load_output()
expl_df = _load_explanations()
edge_count = _get_edge_count()
graph_data = _load_graph_data()

# Merge to include customer_id, predicted_label, risk_score, explanation
merged = pd.merge(
    output_df[["customer_id", "predicted_label", "risk_score"]],
    expl_df[["customer_id", "explanation"]],
    on="customer_id",
    how="inner"
)

# Add risk_tier column
merged["risk_tier"] = merged["risk_score"].apply(
    lambda x: "HIGH" if x > 0.7 else ("LOW" if x < 0.4 else "MEDIUM")
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state: human review decisions
# ─────────────────────────────────────────────────────────────────────────────

if "mo_reviews" not in st.session_state:
    st.session_state["mo_reviews"] = {}  # customer_id → "fraud" | "clear" | None

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build drivers list from explanation row
# ─────────────────────────────────────────────────────────────────────────────

def _drivers_from_row(row: pd.Series) -> list[dict]:
    drivers = []
    for i in (1, 2, 3):
        desc = row.get(f"driver_{i}", "")
        shap = row.get(f"driver_{i}_shap", 0.0)
        if desc and str(desc).strip():
            try:
                shap = float(shap)
            except (TypeError, ValueError):
                shap = 0.0
            drivers.append({"description": str(desc), "shap_value": shap})
    return drivers


# ─────────────────────────────────────────────────────────────────────────────
# Transaction table renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_transaction_table(
    txns_df: pd.DataFrame,
    cid: str,
    key_prefix: str,
) -> None:
    """
    Render a filterable transaction table for one customer.
    Filter values are read from / written to st.session_state so they persist
    across reruns and survive pill-button clicks.
    """
    if txns_df.empty:
        st.caption("No raw transactions found in the source files for this customer.")
        return

    sk_cats   = f"mo_txn_cats_{key_prefix}_{cid}"
    sk_cities = f"mo_txn_cities_{key_prefix}_{cid}"
    sk_types  = f"mo_txn_types_{key_prefix}_{cid}"
    sk_sort   = f"mo_txn_sort_{key_prefix}_{cid}"
    sk_asc    = f"mo_txn_asc_{key_prefix}_{cid}"

    # Only show non-empty merchant categories (card-type rows)
    all_cats   = sorted(c for c in txns_df["category"].dropna().unique().tolist() if c)
    all_cities = sorted(c for c in txns_df["city"].dropna().unique().tolist() if c)
    all_types  = sorted(txns_df["txn_type"].dropna().unique().tolist())
    min_dt = txns_df["datetime"].min()
    max_dt = txns_df["datetime"].max()

    # ── Filter row ──────────────────────────────────────────────────────────
    fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
    with fcol1:
        sel_cats = st.multiselect(
            "Merchant Category",
            options=all_cats,
            default=st.session_state.get(sk_cats, []),
            key=f"ms_cat_{key_prefix}_{cid}",
            placeholder="All categories",
        )
    with fcol2:
        sel_types = st.multiselect(
            "Transaction Type",
            options=all_types,
            default=st.session_state.get(sk_types, []),
            key=f"ms_type_{key_prefix}_{cid}",
            placeholder="All types",
        )
    with fcol3:
        sel_cities = st.multiselect(
            "City / Location",
            options=all_cities,
            default=st.session_state.get(sk_cities, []),
            key=f"ms_city_{key_prefix}_{cid}",
            placeholder="All locations",
        )

    fcol4, fcol5, fcol6, fcol7 = st.columns([3, 1, 1, 1])
    with fcol4:
        date_range = st.date_input(
            "Date Range",
            value=(
                min_dt.date() if pd.notna(min_dt) else None,
                max_dt.date() if pd.notna(max_dt) else None,
            ),
            key=f"dr_{key_prefix}_{cid}",
        )
    with fcol5:
        sort_by = st.selectbox(
            "Sort By",
            options=["datetime", "amount"],
            format_func=lambda x: "Date / Time" if x == "datetime" else "Amount",
            index=["datetime", "amount"].index(
                st.session_state.get(sk_sort, "datetime")
            ),
            key=f"sb_{key_prefix}_{cid}",
        )
    with fcol6:
        sort_asc = st.selectbox(
            "Order",
            options=[True, False],
            format_func=lambda x: "↑ Asc" if x else "↓ Desc",
            index=0 if st.session_state.get(sk_asc, True) else 1,
            key=f"sa_{key_prefix}_{cid}",
        )
    with fcol7:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Clear Filters", key=f"clf_{key_prefix}_{cid}", width="content"):
            for k in (sk_cats, sk_cities, sk_types, sk_sort, sk_asc):
                st.session_state.pop(k, None)
            st.rerun()

    # Persist selections
    st.session_state[sk_cats]  = sel_cats
    st.session_state[sk_cities] = sel_cities
    st.session_state[sk_types] = sel_types
    st.session_state[sk_sort]  = sort_by
    st.session_state[sk_asc]   = sort_asc

    # ── Apply filters ───────────────────────────────────────────────────────
    mask = pd.Series(True, index=txns_df.index)
    if sel_cats:
        mask &= txns_df["category"].isin(sel_cats)
    if sel_types:
        mask &= txns_df["txn_type"].isin(sel_types)
    if sel_cities:
        mask &= txns_df["city"].str.lower().isin([c.lower() for c in sel_cities])
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        d_start, d_end = date_range
        if d_start:
            mask &= txns_df["datetime"].dt.date >= d_start
        if d_end:
            mask &= txns_df["datetime"].dt.date <= d_end

    filtered = txns_df[mask].copy()

    # ── Sort ────────────────────────────────────────────────────────────────
    filtered = filtered.sort_values(sort_by, ascending=sort_asc, na_position="last")

    n_filt = len(filtered)
    total  = len(txns_df)

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_amt = filtered["amount"].sum()
    avg_amt   = filtered["amount"].mean() if n_filt > 0 else 0.0
    date_str  = (
        f"{filtered['datetime'].min().strftime('%b %d')} – "
        f"{filtered['datetime'].max().strftime('%b %d, %Y')}"
        if n_filt > 0 and pd.notna(filtered["datetime"].min())
        else "—"
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Showing",      f"{n_filt:,} / {total:,}",
               f"{n_filt - total:,}" if n_filt < total else None)
    mc2.metric("Total Amount", f"${total_amt:,.2f}")
    mc3.metric("Avg Amount",   f"${avg_amt:,.2f}")
    mc4.metric("Period",       date_str)

    # ── Display table ───────────────────────────────────────────────────────
    display = filtered.copy()
    display["datetime"] = display["datetime"].dt.strftime("%Y-%m-%d %H:%M")
    display["amount"]   = display["amount"].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else ""
    )
    display = display.rename(columns={
        "transaction_id": "Transaction ID",
        "datetime":       "Date / Time",
        "amount":         "Amount (CAD)",
        "dr_cr":          "Dr/Cr",
        "txn_type":       "Type",
        "category":       "Merchant Category",
        "city":           "City",
        "province":       "Prov.",
        "ecommerce":      "E-Com",
        "cash":           "ATM Cash",
    })
    st.dataframe(
        display,
        width="content",
        hide_index=True,
        column_config={
            "Amount (CAD)": st.column_config.TextColumn(width="small"),
            "Dr/Cr":        st.column_config.TextColumn(width="small"),
            "Type":         st.column_config.TextColumn(width="small"),
            "Prov.":        st.column_config.TextColumn(width="small"),
            "E-Com":        st.column_config.TextColumn(width="small"),
            "ATM Cash":     st.column_config.TextColumn(width="small"),
        },
    )


def _render_customer_card(row: pd.Series, key_prefix: str = ""):
    """Render a full CustomerInfoCard for one explanation row."""
    from lib.components import risk_color
    cid = str(row["customer_id"])
    risk_score = float(row["risk_score"])
    risk_tier = str(row.get("risk_tier", "LOW"))
    narrative = str(row.get("explanation", ""))
    drivers = _drivers_from_row(row)
    review_state = st.session_state["mo_reviews"].get(cid)

    # Resolve actual graph neighborhood (with per-category/city transaction counts)
    cat_counts, city_counts = _get_customer_neighbors(cid, graph_data)

    card = CustomerInfoCard(narrative_engine=PrecomputedNarrative())

    # ── Full-width header + risk badge ────────────────────────────────────────
    border_color = risk_color(risk_score)
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
                Customer ID: <b style="color:#94a3b8;">{cid}</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card.render_risk_badge(risk_score, risk_tier)

    # ── Side-by-side: graph (left) | narrative + drivers (right) ─────────────
    col_graph, col_info = st.columns([3, 2], gap="medium")
    with col_graph:
        if cat_counts or city_counts:
            n_cats = len(cat_counts)
            n_cities = len(city_counts)
            total_txns = sum(cat_counts.values())
            st.caption(
                f"Training-graph neighborhood: {n_cats} merchant categor{'y' if n_cats == 1 else 'ies'}, "
                f"{n_cities} cit{'y' if n_cities == 1 else 'ies'}, "
                f"{total_txns} transaction edge{'s' if total_txns != 1 else ''}"
            )
            # Use fixed graph height (380px) and make the narrative column
            # scrollable if it exceeds this height.
            graph_h_px = 380
            nb_html = GraphVisualizer.build_neighborhood_graph(
                cid, risk_score, cat_counts, city_counts, height=f"{graph_h_px}px"
            )
            st_components.html(nb_html, height=graph_h_px, scrolling=False)
        else:
            st.caption("Customer not found in training graph — no neighborhood to display.")
    with col_info:
        # Render narrative + drivers into one scrollable HTML block so that
        # the right column gets a scrollbar if it exceeds the graph height.
        # Narrative comes from `narrative` (precomputed by PrecomputedNarrative).
        narr_html = f"""
        <div style="max-height:{graph_h_px + 40}px; overflow:auto; padding-right:8px;">
        """

        # Narrative block (reuse styles from CustomerInfoCard.render_narrative)
        narr_html += (
            f"<div style='background:#1e293b;border-left:4px solid {risk_color(risk_score)};"
            "border-radius:6px;padding:14px 18px;margin-bottom:12px;color:#e2e8f0;"
            "font-size:0.95rem;line-height:1.6;'>"
            + (narrative or "")
            + "</div>"
        )

        # Drivers block (replicating CustomerInfoCard.render_drivers styling)
        if drivers:
            narr_html += "<div><strong>Top Risk Drivers (SHAP)</strong></div>"
            for d in drivers:
                feat = d.get("feature", "")
                desc = d.get("description", feat)
                try:
                    shap_val = float(d.get("shap_value", 0.0))
                except Exception:
                    shap_val = 0.0
                bar_pct = min(100, int(abs(shap_val) * 1200))
                bar_color = "#ef4444" if shap_val > 0 else "#3b82f6"
                raw = d.get("raw_value", None)
                raw_str = f" = {raw:.2f}" if raw is not None else ""

                geo_badge = ""
                if feat == "geo_velocity" and raw is not None:
                    try:
                        if float(raw) > 3.0:
                            geo_badge = (
                                " &nbsp;<span style='background:#7f1d1d;color:#fca5a5;"
                                "border-radius:4px;padding:1px 6px;font-size:0.75rem;'>🌍 geo-impossible</span>"
                            )
                    except Exception:
                        pass

                narr_html += (
                    "<div style='margin-bottom:8px;'>"
                    "<div style='display:flex;justify-content:space-between;"
                    "font-size:0.85rem;color:#94a3b8;margin-bottom:3px;'>"
                    f"<span>{desc}{raw_str}{geo_badge}</span>"
                    f"<span style='color:{bar_color};font-weight:600;'>{shap_val:+.4f}</span>"
                    "</div>"
                    "<div style='background:#334155;border-radius:4px;height:6px;'>"
                    f"<div style='width:{bar_pct}%;background:{bar_color};height:6px;border-radius:4px;'></div>"
                    "</div></div>"
                )

        narr_html += "</div>"
        st.markdown(narr_html, unsafe_allow_html=True)

    # ── Transactions section ─────────────────────────────────────────────────
    txns_df = _load_customer_transactions(cid)
    n_txns  = len(txns_df)
    amt_sum = txns_df["amount"].sum() if n_txns > 0 else 0.0

    sk_cats   = f"mo_txn_cats_{key_prefix}_{cid}"
    sk_cities = f"mo_txn_cities_{key_prefix}_{cid}"

    # Quick-filter pills: top graph nodes as one-click filter shortcuts.
    # Clicking a pill sets the corresponding filter in session state and reruns,
    # which causes the Transactions expander to auto-open.
    if (cat_counts or city_counts) and n_txns > 0:
        top_items = (
            [(name, "cat",  n) for name, n in sorted(cat_counts.items(),  key=lambda x: -x[1])[:5]]
          + [(name, "city", n) for name, n in sorted(city_counts.items(), key=lambda x: -x[1])[:3]]
        )
        st.caption(
            "🔵 **Category** · 🟠 **City** — click a node to filter the transaction table:"
        )
        pcols = st.columns(min(len(top_items), 8), gap="small")
        for i, (name, ntype, count) in enumerate(top_items):
            icon  = "🔵" if ntype == "cat" else "🟠"
            short = (name[:13] + "…") if len(name) > 13 else name
            with pcols[i]:
                if st.button(
                    f"{icon} {short}  ×{count}",
                    key=f"pill_{key_prefix}_{cid}_{ntype}_{i}",
                    width="content",
                ):
                    if ntype == "cat":
                        st.session_state[sk_cats]   = [name]
                        st.session_state.pop(sk_cities, None)
                    else:
                        st.session_state[sk_cities] = [name]
                        st.session_state.pop(sk_cats, None)
                    st.rerun()

    has_filter = bool(
        st.session_state.get(sk_cats) or st.session_state.get(sk_cities)
    )
    expander_label = (
        f"Transactions  ·  {n_txns:,} records  ·  ${amt_sum:,.2f} CAD total"
        if n_txns > 0
        else "Transactions (none found in source files)"
    )
    with st.expander(expander_label, expanded=has_filter):
        _render_transaction_table(txns_df, cid, key_prefix)

    st.divider()

    # Human review buttons
    col_a, col_b, col_status = st.columns([1, 1, 2])
    with col_a:
        if st.button(
            "Confirm Fraud",
            key=f"{key_prefix}_fraud_{cid}",
            width="content",
            type="primary" if review_state != "fraud" else "secondary",
        ):
            st.session_state["mo_reviews"][cid] = (
                None if review_state == "fraud" else "fraud"
            )
            st.rerun()
    with col_b:
        if st.button(
            "Clear / Not Fraud",
            key=f"{key_prefix}_clear_{cid}",
            width="content",
        ):
            st.session_state["mo_reviews"][cid] = (
                None if review_state == "clear" else "clear"
            )
            st.rerun()
    with col_status:
        if review_state == "fraud":
            st.error("Marked as **Confirmed Fraud** by investigator")
        elif review_state == "clear":
            st.success("Marked as **Cleared** by investigator")
        else:
            st.caption("Awaiting human review")

    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Page header + database stats
# ─────────────────────────────────────────────────────────────────────────────

st.title("Model Output & Investigator Review")
st.caption("Pre-computed risk scores from the last full-graph GraphSAGE inference run.")

# Stats row
n_customers = len(merged)
n_flagged = int((merged["predicted_label"] == 1).sum()) if "predicted_label" in merged.columns else int((merged["risk_score"] > 0.5).sum())
n_high = int((merged["risk_tier"] == "HIGH").sum()) if "risk_tier" in merged.columns else 0
n_reviewed = len(st.session_state["mo_reviews"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers", f"{n_customers:,}")
c2.metric("Flagged (>50%)", f"{n_flagged:,}", help="Predicted label = 1")
c3.metric("High Risk (>70%)", f"{n_high:,}")
c4.metric(
    "Graph Edges",
    f"{edge_count:,}" if edge_count else "N/A",
    help="Total edges in the heterogeneous transaction graph",
)
c5.metric("Reviewed", f"{n_reviewed:,}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs: Top K | Search
# ─────────────────────────────────────────────────────────────────────────────

tab_topk, tab_search = st.tabs(["Top Suspicious Customers", "Search"])

# ── Tab 1: Top K suspicious ───────────────────────────────────────────────────
with tab_topk:
    col_k, col_tier, col_exclude = st.columns([1, 1, 2])
    with col_k:
        top_k = st.slider("Show Top K customers", min_value=1, max_value=50, value=5)
    with col_tier:
        tier_filter = st.multiselect(
            "Risk Tier Filter",
            options=["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"],
        )
    with col_exclude:
        exclude_cleared = st.toggle("Hide customers already cleared", value=True)

    # Apply filters
    filtered = merged.copy()
    if "risk_tier" in filtered.columns and tier_filter:
        filtered = filtered[filtered["risk_tier"].isin(tier_filter)]
    if exclude_cleared:
        cleared_ids = {
            cid for cid, v in st.session_state["mo_reviews"].items() if v == "clear"
        }
        filtered = filtered[~filtered["customer_id"].astype(str).isin(cleared_ids)]

    top_df = filtered.sort_values("risk_score", ascending=False).head(top_k)

    if top_df.empty:
        st.info("No customers match the current filters.")
    else:
        # Summary leaderboard table
        st.markdown(f"**{len(top_df)} most suspicious customers** (sorted by risk score)")
        leaderboard_cols = ["customer_id", "risk_score", "risk_tier",
                            "driver_1", "driver_2", "narrative"]
        available = [c for c in leaderboard_cols if c in top_df.columns]
        lb = top_df[available].copy()
        if "risk_score" in lb.columns:
            lb["risk_score"] = lb["risk_score"].map("{:.2%}".format)
        if "narrative" in lb.columns:
            lb["narrative"] = lb["narrative"].str[:80] + "…"
        st.dataframe(lb, width="content", hide_index=True)
        st.divider()

        # Expandable cards
        for _, row in top_df.iterrows():
            cid = str(row["customer_id"])
            tier = str(row.get("risk_tier", "LOW"))
            score = float(row["risk_score"])
            review = st.session_state["mo_reviews"].get(cid, "")
            review_badge = " [FRAUD]" if review == "fraud" else (" [CLEAR]" if review == "clear" else "")

            with st.expander(
                f"{tier} {cid}  —  {score:.2%}{review_badge}",
                expanded=(tier == "HIGH" and not review),
            ):
                _render_customer_card(row, key_prefix="topk")

# ── Tab 2: Search ─────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("Search by **Customer ID** or **Transaction ID** (pulls up the parent customer's profile).")

    search_col, _ = st.columns([2, 3])
    with search_col:
        query = st.text_input(
            "Search",
            placeholder="e.g. SYNID0100000167 or TX_123456",
            key="mo_search_input",
        )

    if query:
        q = query.strip()

        # Customer ID search
        match = merged[merged["customer_id"].astype(str).str.contains(q, case=False, na=False)]

        if match.empty:
            st.warning(f"No customer found for '{q}'.")
        else:
            st.success(f"{len(match)} customer(s) found.")
            for _, row in match.iterrows():
                _render_customer_card(row, key_prefix="search")
    else:
        st.info("Enter a customer ID above to look up their risk profile.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: review summary
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Review Summary")
    reviews = st.session_state["mo_reviews"]
    confirmed = [cid for cid, v in reviews.items() if v == "fraud"]
    cleared = [cid for cid, v in reviews.items() if v == "clear"]

    st.metric("Confirmed Fraud", len(confirmed))
    st.metric("Cleared", len(cleared))

    if confirmed:
        st.markdown("**Confirmed fraud:**")
        for cid in confirmed:
            st.markdown(f"- `{cid}`")

    if cleared:
        st.markdown("**Cleared:**")
        for cid in cleared:
            st.markdown(f"- `{cid}`")

    if reviews:
        st.divider()
        if st.button("Export Review Decisions", width="content"):
            review_df = pd.DataFrame(
                [{"customer_id": k, "decision": v} for k, v in reviews.items()]
            )
            csv_bytes = review_df.to_csv(index=False).encode()
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="investigator_reviews.csv",
                mime="text/csv",
                width="content",
            )
        if st.button("Clear All Reviews", width="content"):
            st.session_state["mo_reviews"] = {}
            st.rerun()