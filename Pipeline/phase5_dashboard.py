# =============================================================================
# PHASE 5 — STREAMLIT DASHBOARD
# Airline Loyalty Program — Behavioral Intelligence Project
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Loyalty Intelligence Dashboard",
    page_icon   = "✈️",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Outputs"

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "Champions"    : "#2196F3",
    "At-Risk Stars": "#F44336",
    "Loyalists"    : "#4CAF50",
    "Sleeping/Lost": "#FF9800",
    "Dormant"      : "#9E9E9E",
}
RISK_COLORS = {
    "Low Risk"   : "#4CAF50",
    "Medium Risk": "#FF9800",
    "High Risk"  : "#F44336",
    "Dormant"    : "#9E9E9E",
}

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    members  = pd.read_csv(f"{OUTPUT_DIR}/segmented_members.csv")
    profiles = pd.read_csv(f"{OUTPUT_DIR}/segment_profiles.csv")
    shap_imp = pd.read_csv(f"{OUTPUT_DIR}/shap_feature_importance.csv")
    features = pd.read_csv(f"{OUTPUT_DIR}/model_ready_features.csv")

    # Merge flight-window columns into members for Member Lookup page
    # segmented_members.csv only has 18m totals; lookup needs 3m/6m/12m breakdown
    flight_cols = [
        "loyalty_number",
        "flights_3m",       "flights_6m",       "flights_12m",
        "distance_3m",      "distance_6m",      "distance_12m",
        "pts_earned_3m",    "pts_earned_6m",    "pts_earned_12m",
        "active_months_3m", "active_months_6m", "active_months_12m",
    ]
    flight_cols = [c for c in flight_cols if c in features.columns]
    members = members.merge(features[flight_cols], on="loyalty_number", how="left")

    return members, profiles, shap_imp, features
st.write("Dashboard file:", __file__)
st.write("Output path:", OUTPUT_DIR)
st.write("Exists:", OUTPUT_DIR.exists())
members, profiles, shap_imp, features = load_data()

# Fix cluster name duplicates — remap based on behaviour profile
def remap_segment(row):
    """More descriptive behavioural names based on cluster data."""
    c = row.get("behaviour_cluster", -1)
    if c == -1 or pd.isna(c):
        return "Dormant"
    mapping = {
        # Will be overridden dynamically below
    }
    return row.get("segment_name", "Unknown")

# Build dynamic cluster name map from data
cluster_stats = (members[members["behaviour_cluster"] >= 0]
                 .groupby("behaviour_cluster")
                 .agg(avg_flights=("flights_18m","mean"),
                      avg_months=("active_months_18m","mean"),
                      avg_recency=("months_since_last_flight","mean"),
                      avg_redeem=("redemption_ratio","mean"))
                 .round(2))

def smart_cluster_name(row):
    if row["avg_recency"] >= 8:
        return "Fading Flyers"
    elif row["avg_flights"] >= 30 and row["avg_months"] >= 10:
        return "Elite Road Warriors"
    elif row["avg_flights"] >= 15 and row["avg_months"] >= 7:
        return "Frequent Travellers"
    elif row["avg_flights"] >= 15 and row["avg_months"] < 5:
        return "Burst Travellers"
    elif row["avg_flights"] < 5:
        return "Occasional Flyers"
    else:
        return "Regular Flyers"

cluster_stats["smart_name"] = cluster_stats.apply(smart_cluster_name, axis=1)
smart_name_map = cluster_stats["smart_name"].to_dict()
smart_name_map[-1] = "Dormant"
members["behaviour_name"] = members["behaviour_cluster"].map(smart_name_map)

# Derived metrics
total_members    = len(members)
at_risk_stars    = members[members["final_segment"] == "At-Risk Stars"]
clv_at_risk      = at_risk_stars["clv"].sum()
high_risk_count  = (members["risk_tier"] == "High Risk").sum()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.image("https://img.icons8.com/color/96/airplane-mode-on.png", width=60)
st.sidebar.title("✈️ Loyalty Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview",
     "🚨 At-Risk Members",
     "👥 Segments",
     "🔍 Member Lookup",
     "📈 Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

# Global filters
selected_segments = st.sidebar.multiselect(
    "Segment",
    options=members["final_segment"].unique().tolist(),
    default=members["final_segment"].unique().tolist()
)
selected_tiers = st.sidebar.multiselect(
    "Loyalty Card",
    options=sorted(members["loyalty_card"].dropna().unique().tolist()),
    default=sorted(members["loyalty_card"].dropna().unique().tolist())
)

# Apply filters
filtered = members[
    members["final_segment"].isin(selected_segments) &
    members["loyalty_card"].isin(selected_tiers)
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(filtered):,}** members selected")

# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================
if page == "📊 Overview":
    st.title("📊 Loyalty Programme Overview")
    st.caption("Real-time view of member health across 16,737 Canadian loyalty members (2017–2018)")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Members",      f"{total_members:,}")
    c2.metric("At-Risk Stars",      f"{len(at_risk_stars):,}",
              delta=f"${clv_at_risk/1e6:.1f}M CLV at risk", delta_color="inverse")
    c3.metric("High Risk Members",  f"{high_risk_count:,}",
              delta=f"{high_risk_count/total_members*100:.1f}% of base",
              delta_color="inverse")
    c4.metric("Champions",
              f"{(members['final_segment']=='Champions').sum():,}",
              delta="38.9% of base")
    c5.metric("Dormant Members",
              f"{(members['final_segment']=='Dormant').sum():,}",
              delta="Activation opportunity")

    st.markdown("---")

    # ── Row 1: Segment distribution + Risk tier ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Member Segments")
        seg_counts = (filtered["final_segment"]
                      .value_counts()
                      .reset_index()
                      .rename(columns={"final_segment": "Segment",
                                       "count": "Members"}))
        fig = px.pie(seg_counts, values="Members", names="Segment",
                     color="Segment",
                     color_discrete_map=COLORS,
                     hole=0.45)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                          height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Tier Distribution")
        risk_counts = (filtered["risk_tier"]
                       .value_counts()
                       .reset_index()
                       .rename(columns={"risk_tier": "Risk Tier",
                                        "count": "Members"}))
        tier_order = ["Low Risk", "Medium Risk", "High Risk", "Dormant"]
        risk_counts["Risk Tier"] = pd.Categorical(
            risk_counts["Risk Tier"], categories=tier_order, ordered=True
        )
        risk_counts = risk_counts.sort_values("Risk Tier")
        fig = px.bar(risk_counts, x="Risk Tier", y="Members",
                     color="Risk Tier",
                     color_discrete_map=RISK_COLORS,
                     text="Members")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, height=320,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: CLV × Churn Score Scatter ─────────────────────────────────────
    st.subheader("Strategic Quadrant Map — CLV × Churn Risk")
    st.caption("Each dot is a member. Use this to identify where intervention is most valuable.")

    sample = filtered.sample(min(3000, len(filtered)), random_state=42)
    fig = px.scatter(
        sample,
        x="churn_score", y="clv",
        color="final_segment",
        color_discrete_map=COLORS,
        opacity=0.5,
        labels={"churn_score": "Churn Score", "clv": "CLV ($)",
                "final_segment": "Segment"},
        hover_data={"loyalty_number": True, "loyalty_card": True,
                    "churn_score": ":.3f", "clv": ":,.0f"},
        range_y=[0, 35000]
    )
    fig.add_vline(x=0.4, line_dash="dash", line_color="black", opacity=0.4)
    fig.add_hline(y=5780, line_dash="dash", line_color="black", opacity=0.4)
    fig.add_annotation(x=0.2,  y=33000, text="Champions",     showarrow=False,
                        font=dict(color="#2196F3", size=12, family="Arial Black"))
    fig.add_annotation(x=0.75, y=33000, text="At-Risk Stars", showarrow=False,
                        font=dict(color="#F44336", size=12, family="Arial Black"))
    fig.add_annotation(x=0.2,  y=500,   text="Loyalists",     showarrow=False,
                        font=dict(color="#4CAF50", size=12, family="Arial Black"))
    fig.add_annotation(x=0.75, y=500,   text="Sleeping/Lost", showarrow=False,
                        font=dict(color="#FF9800", size=12, family="Arial Black"))
    fig.update_layout(height=430, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Segment CLV comparison ────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Average CLV by Segment")
        seg_clv = (filtered.groupby("final_segment")["clv"]
                   .mean().reset_index()
                   .rename(columns={"clv": "Avg CLV", "final_segment": "Segment"}))
        seg_clv["color"] = seg_clv["Segment"].map(COLORS)
        fig = px.bar(seg_clv.sort_values("Avg CLV", ascending=True),
                     x="Avg CLV", y="Segment", orientation="h",
                     color="Segment", color_discrete_map=COLORS,
                     text="Avg CLV")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, height=300,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Avg Flights (18m) by Segment")
        seg_fl = (filtered.groupby("final_segment")["flights_18m"]
                  .mean().reset_index()
                  .rename(columns={"flights_18m": "Avg Flights",
                                   "final_segment": "Segment"}))
        fig = px.bar(seg_fl.sort_values("Avg Flights", ascending=True),
                     x="Avg Flights", y="Segment", orientation="h",
                     color="Segment", color_discrete_map=COLORS,
                     text="Avg Flights")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False, height=300,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 2 — AT-RISK MEMBERS
# =============================================================================
elif page == "🚨 At-Risk Members":
    st.title("🚨 At-Risk Member Action Centre")
    st.caption("Members who need attention — sorted by churn score × CLV impact")

    # ── Priority filters ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    min_score  = col1.slider("Min churn score",  0.0, 1.0, 0.5, 0.05)
    min_clv    = col2.number_input("Min CLV ($)", 0, 100000, 0, 1000)
    seg_filter = col3.multiselect(
        "Segment filter",
        ["At-Risk Stars", "Sleeping/Lost", "Dormant"],
        default=["At-Risk Stars", "Sleeping/Lost"]
    )

    at_risk = members[
        (members["churn_score"]    >= min_score) &
        (members["clv"]            >= min_clv)   &
        (members["final_segment"].isin(seg_filter))
    ].copy()

    at_risk["clv_at_risk"] = (at_risk["churn_score"] * at_risk["clv"]).round(0)
    at_risk = at_risk.sort_values("clv_at_risk", ascending=False)

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Members Flagged",    f"{len(at_risk):,}")
    k2.metric("Total CLV at Risk",  f"${at_risk['clv'].sum()/1e6:.2f}M")
    k3.metric("Avg Churn Score",    f"{at_risk['churn_score'].mean():.2f}")
    k4.metric("Avg CLV",            f"${at_risk['clv'].mean():,.0f}")

    st.markdown("---")

    # ── Recommended action per member ────────────────────────────────────────
    ACTION_MAP = {
        "At-Risk Stars" : "📞 Personal call + 2× points offer",
        "Sleeping/Lost" : "📧 'We miss you' email + fare discount",
        "Dormant"       : "📧 First-flight activation email",
    }
    at_risk["recommended_action"] = at_risk["final_segment"].map(ACTION_MAP)

    display_cols = {
        "loyalty_number"      : "Member ID",
        "loyalty_card"        : "Card",
        "clv"                 : "CLV ($)",
        "churn_score"         : "Churn Score",
        "months_since_last_flight" : "Months Inactive",
        "flights_18m"         : "Flights (18m)",
        "final_segment"       : "Segment",
        "recommended_action"  : "Recommended Action",
    }
    display_df = (at_risk[list(display_cols.keys())]
                  .rename(columns=display_cols)
                  .head(200))
    display_df["CLV ($)"]     = display_df["CLV ($)"].apply(lambda x: f"${x:,.0f}")
    display_df["Churn Score"] = display_df["Churn Score"].apply(lambda x: f"{x:.3f}")

    st.subheader(f"Action List ({len(at_risk):,} members)")
    st.dataframe(display_df, use_container_width=True, height=420)

    # Download button
    csv = at_risk[list(display_cols.keys())].to_csv(index=False)
    st.download_button(
        "⬇️ Download Action List (CSV)",
        data=csv,
        file_name="at_risk_action_list.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # ── CLV at risk by card tier ──────────────────────────────────────────────
    st.subheader("CLV at Risk by Card Tier")
    clv_by_tier = (at_risk.groupby("loyalty_card")["clv"]
                   .agg(["sum", "count", "mean"])
                   .reset_index()
                   .rename(columns={"loyalty_card": "Card",
                                    "sum": "Total CLV at Risk",
                                    "count": "Members",
                                    "mean": "Avg CLV"}))
    fig = px.bar(clv_by_tier, x="Card", y="Total CLV at Risk",
                 color="Card", text="Members",
                 color_discrete_sequence=["#F44336", "#FF9800", "#2196F3"])
    fig.update_traces(texttemplate="%{text} members", textposition="outside")
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 3 — SEGMENTS
# =============================================================================
elif page == "👥 Segments":
    st.title("👥 Customer Segments")
    st.caption("Understand who your members are and what drives their behaviour")

    seg_select = st.selectbox(
        "Select a segment to explore",
        ["Champions", "At-Risk Stars", "Loyalists", "Sleeping/Lost", "Dormant"]
    )

    seg_data = members[members["final_segment"] == seg_select]

    SEGMENT_DESCRIPTIONS = {
        "Champions"    : "Your most valuable, most engaged members. High CLV, low churn risk. "
                         "They fly frequently and consistently. Priority: keep them happy.",
        "At-Risk Stars": "High-value members showing signs of disengagement. "
                         "They used to fly often but have gone quiet. "
                         "Each lost member represents significant CLV. Act urgently.",
        "Loyalists"    : "Consistent, low-risk flyers with growth potential. "
                         "They are loyal but haven't reached their full value. "
                         "The right incentive can move them up a tier.",
        "Sleeping/Lost": "Low-value members with high churn probability. "
                         "Use low-cost re-engagement tactics. "
                         "Don't overspend — ROI is limited.",
        "Dormant"      : "Members who enrolled but never flew. "
                         "Surprisingly high CLV (from partner/credit card activity). "
                         "A first-flight incentive could unlock significant value.",
    }
    SEGMENT_ACTIONS = {
        "Champions"    : "🎯 **Action:** Quarterly personalised route offer + early access to new routes. "
                         "Send before their typical booking season.",
        "At-Risk Stars": "🚨 **Action:** Phone call from loyalty team within 48 hours of score crossing 0.40. "
                         "Offer status match + 2× points on next 3 bookings.",
        "Loyalists"    : "📈 **Action:** 60-day tier upgrade challenge — earn X points to reach next tier. "
                         "Trigger at 6-month enrollment anniversary.",
        "Sleeping/Lost": "📧 **Action:** 'We miss you' email with discounted fare on their most-flown route. "
                         "30-day expiry. Email only (keep cost low).",
        "Dormant"      : "✈️ **Action:** 3-email sequence over 30 days. "
                         "Double points + waived booking fee on first flight within 90 days.",
    }

    # Segment header
    color = COLORS[seg_select]
    st.markdown(
        f"<div style='background:{color}22; border-left:4px solid {color}; "
        f"padding:12px; border-radius:4px; margin-bottom:16px'>"
        f"<b style='color:{color}'>{seg_select}</b> — {len(seg_data):,} members "
        f"({len(seg_data)/total_members*100:.1f}% of base)<br>"
        f"{SEGMENT_DESCRIPTIONS[seg_select]}</div>",
        unsafe_allow_html=True
    )
    st.info(SEGMENT_ACTIONS[seg_select])

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Members",          f"{len(seg_data):,}")
    k2.metric("Avg CLV",          f"${seg_data['clv'].mean():,.0f}")
    k3.metric("Avg Churn Score",  f"{seg_data['churn_score'].mean():.2f}")
    k4.metric("Avg Flights (18m)",f"{seg_data['flights_18m'].mean():.1f}")
    k5.metric("Avg Months Inactive",
              f"{seg_data['months_since_last_flight'].mean():.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CLV Distribution")
        fig = px.histogram(seg_data, x="clv", nbins=40,
                           color_discrete_sequence=[color])
        fig.update_layout(height=280, margin=dict(t=10, b=10),
                          xaxis_title="CLV ($)", yaxis_title="Members")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Score Distribution")
        fig = px.histogram(seg_data, x="churn_score", nbins=30,
                           color_discrete_sequence=[color])
        fig.update_layout(height=280, margin=dict(t=10, b=10),
                          xaxis_title="Churn Score", yaxis_title="Members")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Card Tier Breakdown")
        tier_counts = seg_data["loyalty_card"].value_counts().reset_index()
        fig = px.pie(tier_counts, values="count", names="loyalty_card",
                     color_discrete_sequence=["#2196F3","#FF9800","#4CAF50"],
                     hole=0.4)
        fig.update_layout(height=260, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Province Breakdown (Top 8)")
        prov = (seg_data["province"].value_counts()
                .head(8).reset_index()
                .rename(columns={"province":"Province","count":"Members"}))
        fig = px.bar(prov, x="Members", y="Province", orientation="h",
                     color_discrete_sequence=[color])
        fig.update_layout(height=260, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 4 — MEMBER LOOKUP
# =============================================================================
elif page == "🔍 Member Lookup":
    st.title("🔍 Member Deep-Dive")
    st.caption("Look up any member to see their profile, churn risk, and recommended action")

    member_id = st.number_input(
        "Enter Member ID (Loyalty Number)",
        min_value=int(members["loyalty_number"].min()),
        max_value=int(members["loyalty_number"].max()),
        value=int(members[members["final_segment"]=="At-Risk Stars"]
                  .sort_values("churn_score", ascending=False)
                  .iloc[0]["loyalty_number"])
    )

    row = members[members["loyalty_number"] == member_id]
    if len(row) == 0:
        st.error("Member not found.")
    else:
        row = row.iloc[0]
        seg   = row["final_segment"]
        color = COLORS.get(seg, "#888")

        # Header
        st.markdown(
            f"<div style='background:{color}22; border-left:4px solid {color}; "
            f"padding:16px; border-radius:6px'>"
            f"<h3 style='margin:0; color:{color}'>Member #{int(row['loyalty_number'])}</h3>"
            f"<span style='color:#666'>{row['loyalty_card']} Card  |  "
            f"{seg}  |  Risk: {row['risk_tier']}</span></div>",
            unsafe_allow_html=True
        )
        st.markdown(" ")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CLV",             f"${row['clv']:,.0f}")
        c2.metric("Churn Score",     f"{row['churn_score']:.3f}")
        c3.metric("Flights (18m)",   f"{int(row['flights_18m'])}")
        c4.metric("Months Inactive", f"{row['months_since_last_flight']:.0f}")
        c5.metric("Tenure",          f"{row['enrollment_tenure_months']:.0f} months")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Flight Activity Profile")
            metric_names  = ["Flights 3m", "Flights 6m", "Flights 12m", "Flights 18m"]
            metric_values = [
                row.get("flights_3m",  0),
                row.get("flights_6m",  0),
                row.get("flights_12m", 0),
                row.get("flights_18m", 0),
            ]
            # Cumulative bar — each window includes prior windows
            fig = go.Figure(go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=color,
                text=[f"{int(v)}" for v in metric_values],
                textposition="outside"
            ))
            fig.update_layout(height=280, margin=dict(t=10, b=10),
                               yaxis_title="Total Flights in Window")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Member Profile")
            profile_data = {
                "Province"          : row.get("province", "—"),
                "Gender"            : row.get("gender", "—"),
                "Education"         : row.get("education", "—"),
                "Marital Status"    : row.get("marital_status", "—"),
                "Enrollment Year"   : int(row.get("enrollment_year", 0)),
                "Seasons Flown"     : f"{row['seasons_flown']:.0f} / 4",
                "Redemption Ratio"  : f"{row['redemption_ratio']*100:.1f}%",
                "Flight Trend"      : f"{row['flight_trend']:+.1f}",
                "Behaviour Cluster" : row.get("behaviour_name", "—"),
            }
            for k, v in profile_data.items():
                st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        st.subheader("🎯 Recommended Action")
        actions = {
            "Champions"    : ("Proactive quarterly personalised route offer + early access to new routes. "
                              "Send before their typical booking season.",
                              "Email + App notification", "Low"),
            "At-Risk Stars": ("Phone call from loyalty team within 48 hours. "
                              "Offer: status match + 2× points on next 3 bookings.",
                              "Phone + Email", "Urgent"),
            "Loyalists"    : ("60-day tier upgrade challenge — earn X points to reach next tier. "
                              "Trigger at 6-month enrollment anniversary.",
                              "Email + In-App", "Medium"),
            "Sleeping/Lost": ("'We miss you' email with discounted fare on most-flown route. "
                              "30-day expiry. Keep cost low.",
                              "Email only", "Low"),
            "Dormant"      : ("3-email sequence over 30 days. "
                              "Double points + waived booking fee on first flight.",
                              "Email", "Low"),
        }
        action_text, channel, urgency = actions.get(
            seg, ("No specific action recommended.", "—", "—")
        )
        urgency_color = {"Urgent":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}.get(urgency,"⚪")
        st.markdown(f"**Action:** {action_text}")
        st.markdown(f"**Channel:** {channel}  |  **Urgency:** {urgency_color} {urgency}")

# =============================================================================
# PAGE 5 — MODEL INSIGHTS
# =============================================================================
elif page == "📈 Model Insights":
    st.title("📈 Model & Feature Insights")
    st.caption("What drives churn — and what the model found")

    # ── Model performance ─────────────────────────────────────────────────────
    st.subheader("Model Performance Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC",     "0.921", "↑ target: 0.75")
    m2.metric("AUC-PR",      "0.775", "↑ target: 0.30")
    m3.metric("Lift @ Top 20%", "4.2×", "↑ target: 2×")
    m4.metric("Churners in Top 20%", "~83%", "of all churners found")

    st.markdown("---")

    # ── SHAP importance ───────────────────────────────────────────────────────
    st.subheader("Top Features Driving Churn (SHAP Values)")
    st.caption("Higher SHAP = bigger impact on whether a member churns")

    top_shap = shap_imp.head(15).sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(
        x=top_shap["mean_abs_shap"],
        y=top_shap["feature"],
        orientation="h",
        marker_color="steelblue",
        text=top_shap["mean_abs_shap"].round(3),
        textposition="outside"
    ))
    fig.update_layout(
        height=480, margin=dict(t=10, b=10),
        xaxis_title="Mean |SHAP| Value (impact on churn prediction)",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Business interpretation ───────────────────────────────────────────────
    st.subheader("What This Means for the Business")
    insights = {
        "months_since_last_flight"  :
            "**Recency is #1.** A member who hasn't flown in 6+ months is very likely to churn. "
            "Trigger interventions at the 3-month mark, not the 6-month mark.",
        "active_months_3m"          :
            "**Recent consistency matters.** Members who flew in at least 2 of the last 3 months "
            "are much less likely to churn than those with a single recent flight.",
        "enrollment_tenure_months"  :
            "**Newer members are more vulnerable.** Members enrolled in 2017–2018 churn at higher "
            "rates. Early engagement programmes in months 1–6 would reduce this.",
        "flight_trend"              :
            "**Trajectory predicts departure.** A declining flight trend (flying less than 3 months "
            "ago) is a leading indicator — it precedes full churn by 2–4 months.",
        "clv"                       :
            "**CLV is a churn signal, not just a value metric.** High-CLV members churn less — "
            "but when they do, the impact is disproportionate. Watch your top quartile closely.",
    }
    for feat, insight in insights.items():
        shap_val = shap_imp[shap_imp["feature"]==feat]["mean_abs_shap"].values
        shap_str = f"(SHAP: {shap_val[0]:.3f})" if len(shap_val) > 0 else ""
        with st.expander(f"📌 `{feat}` {shap_str}"):
            st.markdown(insight)

    st.markdown("---")
    st.subheader("Churn Rate by Recency")
    recency_data = pd.DataFrame({
        "Months Since Last Flight": ["0–1m", "1–3m", "3–6m", "6–12m", "12–18m"],
        "Churn Rate": [0.011, 0.060, 0.339, 0.935, 1.000],
        "Members":    [12509, 1696, 433, 325, 204]
    })
    fig = px.bar(recency_data, x="Months Since Last Flight", y="Churn Rate",
                 text="Churn Rate", color="Churn Rate",
                 color_continuous_scale=["#4CAF50","#FF9800","#F44336"])
    fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
    fig.update_layout(height=320, margin=dict(t=10,b=10),
                      yaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
