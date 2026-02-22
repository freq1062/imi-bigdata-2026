import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("AML Knowledge Intelligence Library")

section = st.sidebar.radio(
    "Navigate Intelligence Modules",
    [
        "Core Intelligence & Risk Patterns",
        "Red Flag Indicator Library",
        "Risk Clusters for Modelling",
        "LLM & NLP Opportunities",
        "Source Index"
    ]
)

if section == "Core Intelligence & Risk Patterns":

    st.header("Section 1: Core Intelligence & Risk Patterns")
    st.markdown("Contextual intelligence underlying detection logic.")

    with st.expander("Project Athena – Underground Banking Model"):

        st.subheader("Logic")
        st.write("""
        Professional money launderers (PMLs) facilitate circumvention of currency controls
        (e.g., China/Hong Kong) to move funds into Canada/US financial systems.
        """)

        st.subheader("Mechanism")
        st.write("""
        • Funds transferred to 'money mule' accounts (students, homemakers).
        • Mules purchase assets (real estate, vehicles, securities).
        • Funds integrated into formal economy.
        """)

        st.subheader("Key Sectors Exploited")
        st.write("Real Estate | Automotive | Securities | Legal Professionals")

        st.subheader("Model Detection Mapping")
        st.write("""
        - Sudden inbound foreign transfers
        - Student account high-value asset purchases
        - Trust account usage linked to property closings
        - Cross-border transaction clustering
        """)

        st.write("""Source: https://fintrac-canafe.canada.ca/intel/operation/ml-rec-eng""")

    with st.expander("Project Guardian – Synthetic Opioid Trafficking"):

        st.subheader("Production Phase")
        st.write("""
        • Import of precursor chemicals and pill presses from China.
        • Payments via virtual currency or wire transfers.
        • Intermediary jurisdictions: Singapore, Hong Kong.
        """)

        st.subheader("Distribution Phase")
        st.write("""
        • Drug corridors (Vancouver → Calgary/Toronto).
        • Use of legitimate freight/logistics channels.
        """)

        st.subheader("Laundering Mechanisms")
        st.write("""
        • Cash smurfing
        • Online gambling platforms
        • Front companies (pharma/supplement labs)
        """)

        st.subheader("Model Detection Mapping")
        st.write("""
        - VC-to-fiat inflows
        - Logistics industry transaction concentration
        - Pharma lab accounts with no payroll
        - Structured cash deposits
        """)

        st.write("""Source: https://fintrac-canafe.canada.ca/intel/operation/iso-osi-eng""")

    with st.expander("Project Protect – Human Trafficking for Sexual Exploitation"):

        st.subheader("Business Models")
        st.write("""
        • Short-stay hotels
        • Private apartments
        • Illicit storefront massage businesses
        """)

        st.subheader("Financial Profile")
        st.write("""
        • Funnel accounts (victims)
        • High-volume incoming EMTs
        • Immediate depletion of funds
        """)

        st.subheader("Lavish Lifestyle Indicators")
        st.write("""
        • Luxury retail
        • Cosmetic surgery clinics
        • High-end restaurants
        """)

        st.subheader("Model Detection Mapping")
        st.write("""
        - EMT velocity anomaly
        - Account drain ratio
        - Shared hotel address clustering
        - Lifestyle inconsistency scoring
        """)

        st.write("""Source: https://fintrac-canafe.canada.ca/intel/operation/oai-hts-2021-eng""")

if section == "Red Flag Indicator Library":

    st.header("Section 2: Red Flag Indicator Library")

    st.subheader("Transactional & Behavioral Patterns")

    st.table({
        "Indicator": [
            "Flow-Through Activity",
            "Atypical Velocity",
            "Musty/Dirty Currency",
            "Threshold Avoidance"
        ],
        "Description": [
            "Funds exit within hours of entering",
            "Sudden spike in card or account activity",
            "Deposits of degraded physical currency",
            "Structuring across branches or individuals"
        ],
        "Model Feature Mapping": [
            "Inflow-Outflow Time Delta",
            "Transaction Velocity Change %",
            "Branch Deposit Pattern Scoring",
            "Threshold Proximity Index"
        ]
    })

    st.write("""Source: https://fintrac-canafe.canada.ca/guidance-directives/transaction-operation/indicators-indicateurs/fin_mltf-eng""")

    # --- Organized Crime & Industry-Specific Flags ---

    st.subheader("Organized Crime & Industry-Specific Indicators")

    data = [
        {
            "Category": "Opioid Production",
            "Red Flag / Indicator": "Payments for rental storage lockers or warehouses not commensurate with client profile.",
            "Source Context": "FINTRAC Operational Alert (2025) – Synthetic Opioids",
            "Source URL": "https://fintrac-canafe.canada.ca/intel/operation/iso-osi-eng"
        },
        {
            "Category": "Opioid Distribution",
            "Red Flag / Indicator": "Frequent purchases of mailing/packing supplies from post offices not in line with occupation.",
            "Source Context": "FINTRAC Operational Alert (2025) – Synthetic Opioids",
            "Source URL": "https://fintrac-canafe.canada.ca/intel/operation/iso-osi-eng"
        },
        {
            "Category": "Oil Smuggling",
            "Red Flag / Indicator": "Small US companies selling 'West Texas Intermediate' crude at steep discount to market rates.",
            "Source Context": "FinCEN Alert (2025) – Oil Smuggling Schemes",
            "Source URL": "https://www.fincen.gov/system/files/shared/FinCEN-Alert-Oil-Smuggling-FINAL-508C.pdf"
        },
        {
            "Category": "Oil Smuggling",
            "Red Flag / Indicator": "Shipments mislabeled as 'waste oil' or 'hazardous materials' to avoid border scrutiny.",
            "Source Context": "FinCEN Alert (2025) – Oil Smuggling Schemes",
            "Source URL": "https://www.fincen.gov/system/files/shared/FinCEN-Alert-Oil-Smuggling-FINAL-508C.pdf"
        },
        {
            "Category": "Underground Banking",
            "Red Flag / Indicator": "Use of 'straw buyers' (e.g., students/homemakers) to purchase high-end real estate or luxury vehicles.",
            "Source Context": "FINTRAC Operational Alert (2023) – Project Athena",
            "Source URL": "https://fintrac-canafe.canada.ca/intel/operation/ml-rec-eng"
        },
        {
            "Category": "Human Trafficking",
            "Red Flag / Indicator": "Frequent low-value payments for parking and same-day food delivery orders.",
            "Source Context": "FINTRAC Operational Alert (2021) – Project Protect",
            "Source URL": "https://fintrac-canafe.canada.ca/intel/operation/oai-hts-2021-eng"
        },
        {
            "Category": "Trade-Based ML",
            "Red Flag / Indicator": "'Phantom shipments' where funds are transferred for goods that are never received.",
            "Source Context": "FINTRAC Operational Alert (2018) – Trade-Based ML",
            "Source URL": "https://fintrac-canafe.canada.ca/intel/operation/oai-ml-eng"
        }
    ]

    df_flags = pd.DataFrame(data)

    st.dataframe(
    df_flags,
    column_config={
        "Source URL": st.column_config.LinkColumn(
            "Source URL",
            display_text= "Click Here"
        ),
    },
    hide_index=True,
    use_container_width=True
)

    # --- Optional Category Filter ---
    # category_filter = st.multiselect(
    #     "Filter by Category",
    #     df_flags["Category"].unique()
    # )

    # if category_filter:
    #     filtered_df = df_flags[df_flags["Category"].isin(category_filter)]
    # else:
    #     filtered_df = df_flags

    # st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Virtual Currency Indicators")

    st.table({
        "Indicator": [
            "Darknet Exposure",
            "Mixing Services",
            "VC-to-Fiat Imbalance",
            "Crypto-Postage Services"
        ],
        "Description": [
            "Interaction with darknet marketplaces",
            "Use of tumblers to obscure trails",
            "Large exchange inflows without outbound activity",
            "Anonymous mailing service transactions"
        ],
        "Detection Logic": [
            "Blockchain Address Risk Scoring",
            "Mixer Interaction Frequency",
            "Exchange Concentration Ratio",
            "Merchant Risk Classification"
        ]
    })

    st.write("""Source: https://fintrac-canafe.canada.ca/intel/operation/iso-osi-eng""")


if section == "Risk Clusters for Modelling":

    st.header("Section 3: Risk Clusters for Multi-Factor Detection")

    with st.expander("Cluster A – Front Company Scenario"):

        st.write("""
        Business account receives sudden inflows from unrelated parties,
        lacks payroll or tax activity,
        wires funds internationally to high-risk precursor jurisdictions.
        """)

        st.markdown("**Composite Detection Logic:**")
        st.write("""
        - Inflow anomaly score
        - Payroll absence flag
        - High-risk country wire flag (China/India)
        - Industry-code mismatch scoring
        """)

        st.write("""Source: https://fintrac-canafe.canada.ca/intel/operation/iso-osi-eng""")

    with st.expander("Cluster B – Repatriation Scenario"):

        st.write("""
        Bulk cash sent via Armored Car Services to US FI.
        Rapid wiring back to Mexico.
        Purchase of goods for resale.
        """)

        st.markdown("**Composite Detection Logic:**")
        st.write("""
        - ACS deposit flag
        - Circular fund movement detection
        - Rapid outbound international wire
        """)

        st.write("""Source: https://www.fincen.gov/system/files/shared/BCS-Alert-FINAL-508C.pdf""")

    with st.expander("Cluster C – Gatekeeper Scenario"):

        st.write("""
        Legal trust account used to purchase short-term GICs.
        Immediate redemption despite penalties.
        """)

        st.markdown("**Composite Detection Logic:**")
        st.write("""
        - Trust account investment flag
        - Early redemption penalty tolerance score
        - Legal professional intermediary tagging
        """)

        st.write("""https://fintrac-canafe.canada.ca/intel/operation/ml-rec-eng""")

if section == "LLM & NLP Opportunities":

    st.header("Section 4: LLM & NLP Intelligence Automation")

    st.markdown("### 1. Named Entity Recognition (NER)")
    st.write("""
    Extract chemical names (NPP/ANPP), jurisdictions (Hebei),
    and organizational identifiers from law enforcement PDFs.
    """)

    st.write("""Source: https://fintrac-canafe.canada.ca/intel/operation/iso-osi-eng""")

    st.markdown("### 2. Typology Classification")
    st.write("""
    Use LLMs to classify SAR narratives into:
    Project Athena | Guardian | Protect
    based on semantic patterns.
    """)

    st.markdown("### 3. Red Flag Extraction")
    st.write("""
    Automatically extract indicator sentences from regulatory updates
    and map to structured model features.
    """)

    st.markdown("### 4. Relationship Mapping")
    st.write("""
    Link shared phone numbers, hotel addresses,
    or payment descriptors to identify criminal cells.
    """)

    st.write("""Source: https://fintrac-canafe.canada.ca/guidance-directives/transaction-operation/indicators-indicateurs/fin_mltf-eng""")

if section == "Source Index":

    st.header("Section 5: Source Index for Traceability")

    st.markdown("""
    • [FINTRAC Operational Alert (2025): Laundering proceeds of illicit synthetic opioids](https://fintrac-canafe.canada.ca/intel/operation/iso-osi-eng)

    • [FINTRAC Operational Alert (2023): Underground banking schemes (Project Athena Update)](https://fintrac-canafe.canada.ca/intel/operation/ml-rec-eng)

    • [FINTRAC Operational Alert (2021): Human trafficking for sexual exploitation (Project Protect Update)](https://fintrac-canafe.canada.ca/intel/operation/oai-hts-2021-eng)

    • [FINTRAC Operational Alert (2018): Professional money laundering through trade and MSBs](https://fintrac-canafe.canada.ca/intel/operation/oai-ml-eng)

    • [FINTRAC Guidance: Money laundering and terrorist financing indicators](https://fintrac-canafe.canada.ca/guidance-directives/transaction-operation/indicators-indicateurs/fin_mltf-eng)

    • [FinCEN Alert (2025): Bulk Cash Smuggling by Mexico-based TCOs](https://www.fincen.gov/system/files/shared/BCS-Alert-FINAL-508C.pdf)

    • [FinCEN Alert (2025): Oil Smuggling Schemes on the U.S. Southwest Border](https://www.fincen.gov/system/files/shared/FinCEN-Alert-Oil-Smuggling-FINAL-508C.pdf)
    """)