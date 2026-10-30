import io
import os
import re
import zipfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def asset(name):
    """Path to an image stored next to app.py (works locally and on Streamlit Cloud)."""
    return os.path.join(BASE_DIR, name)

GREEN = "#008751"
DARK_GREEN = "#1b5e20"
RED = "#d32f2f"
STATUS_OK = "No Anomaly Flag"
STATUS_FLAG = "Flagged for Review"

DEVELOPERS = [
    {"name": "A. O. Juwon", "address": "Ejuku, Yagba East LGA", "state": "Kogi State",
     "education": "NCE, B.Sc., M.Sc. (In View)", "role": "Team Lead and Project Problem Analyzer",
     "marital_status": "Married", "image": "developer 1.jpeg"},
    {"name": "ADEYEMI MARY MARVELOUS", "address": "ODO-ERE, YAGBA WEST LG", "state": "KOGI STATE",
     "education": "B. Sc", "role": "DATA COLLECTION TEAM", "marital_status": "MARRIED", "image": "developer 2.jpeg"},
    {"name": "IBITOYE MAYOMIKUN JONATHAN", "address": "ISANLU, YAGBA EAST LG", "state": "KOGI STATE",
     "education": "B. Eng. Ilorin", "role": "DATA COLLECTION TEAM", "marital_status": "Single", "image": "developer 3.jpeg"},
    {"name": "OLUKAIYE OLUGBENGA TIMOTHY", "address": "BOLORUNDURO KABBA, KABBA BUNU LG", "state": "KOGI STATE",
     "education": "B. Eng, CUSTECH Osara", "role": "PROBLEM SOLVING TEAM", "marital_status": "Single", "image": "developer 4.jpeg"},
    {"name": "KOMOLAFE BABATUNDE RICHARD", "address": "AYETORO GBEDE IJUMU LG", "state": "KOGI STATE",
     "education": "B. Sc.", "role": "DEMONSTRATION TEAM", "marital_status": "Single", "image": "developer 5.jpeg"},
    {"name": "ADEWUMI BISIRIYU TUNDE", "address": "IFE_OLUKOTUN, YAGBA EAST LG", "state": "KOGI STATE",
     "education": "B. Sc, CNA, NBA (in view)", "role": "DATA COLLECTION TEAM", "marital_status": "MARRIED", "image": "developer 6.jpeg"},
    {"name": "BABALOLA MARVELOUS ITUNUOLUWA", "address": "EGBE, YAGBA WEST", "state": "KOGI STATE",
     "education": "B. Sc (NURSING) FED.UNI LOKOJA", "role": "DEMONSTRATION TEAM MEMBER", "marital_status": "Single", "image": "developer 7.jpeg"},
    {"name": "AKANKU TAIYE OLUWAPEMISIRE", "address": "ITEDO ISANLU, YAGBA EAST LG", "state": "KOGI STATE",
     "education": "B Sc. (Econs)", "role": "DEMONSTRATION TEAM MEMBER", "marital_status": "MARRIED", "image": "developer 8.jpeg"},
    {"name": "VICTOR IWALAYE", "address": "IYARA, IJUMU LG", "state": "KOGI STATE",
     "education": "NATIONAL DEPLOMA", "role": "AI TEAM", "marital_status": "Single", "image": "developer 9.jpeg"},
    {"name": "RACHEAL OLORUNMAIYE", "address": "KABBA BUNNU, LG", "state": "KOGI STATE",
     "education": "B.Sc", "role": "DATA COLLECTION TEAM", "marital_status": "N/A", "image": "developer 10.jpeg"},
    {"name": "ISREAL ADEOLA OLUMAIYEGUN", "address": "ODO ERE, YAGBA WEST LG", "state": "KOGI STATE",
     "education": "B. Sc. (Econs) KSU, kabba", "role": "DEMONSTRATION TEAM MEMBER", "marital_status": "Single", "image": "developer 11.jpeg"},
    {"name": "Team Member 12", "address": "Address 12", "state": "State 12",
     "education": "Education 12", "role": "Role 12", "marital_status": "Single", "image": "developer12.jpeg"},
]

STEP_NAMES = {
    1: "1. Authorization",
    2: "2. Data Input",
    3: "3. AI Analysis",
    4: "4. Visualizations",
    5: "5. Q&A & Kudos",
    6: "6. Developers",
}


# =====================================================================
# DATA HELPERS
# =====================================================================
COLUMN_ALIASES = {
    "net": "netsalary", "netpay": "netsalary", "netsal": "netsalary",
    "basic": "basicsalary", "basicpay": "basicsalary", "basicsal": "basicsalary",
    "allowance": "allowances", "deduction": "deductions",
    "localgovernment": "lga", "localgovernmentarea": "lga",
}
ID_LIKE = {"id", "staffid", "employeeid", "recordnumber", "bankaccount",
           "accountnumber", "accountno", "phone", "phonenumber", "bvn", "nin",
           "serial", "sn", "sno", "index"}


def is_id_col(name):
    n = str(name).lower()
    return n in ID_LIKE or n.endswith("id") or n.endswith("number")


def clean_columns(df):
    """Lower-case column names, strip spaces/underscores, map common aliases."""
    df = df.copy()
    new_cols = []
    for c in df.columns:
        k = re.sub(r"[\s_\-\.]+", "", str(c).strip().lower())
        new_cols.append(COLUMN_ALIASES.get(k, k))
    df.columns = new_cols
    df = df.loc[:, ~df.columns.duplicated()]
    # Convert text columns like "85,000" or "₦85000" to numbers when they are mostly numeric
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and not is_id_col(c):
            conv = pd.to_numeric(
                df[c].astype(str).str.replace(r"[₦,\s]", "", regex=True),
                errors="coerce",
            )
            if conv.notna().mean() > 0.9:
                df[c] = conv
    return df.reset_index(drop=True)


def generate_sample(num_emp=500):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "employeeid": [f"EMP{i:03d}" for i in range(1, num_emp + 1)],
        "name": [f"Employee_{i:03d}" for i in range(1, num_emp + 1)],
        "state": rng.choice(["Kogi State", "Kwara State", "Ekiti State"], num_emp),
        "lga": rng.choice(["Okene", "Kabba", "Lokoja", "Ado-Ekiti", "Ijumu", "Yagba East"], num_emp),
        "basicsalary": np.maximum(rng.normal(85000, 15000, num_emp), 30000).round(2),
        "allowances": np.maximum(rng.normal(18000, 5000, num_emp), 0).round(2),
        "deductions": np.maximum(rng.normal(6000, 1800, num_emp), 0).round(2),
    })
    # Inject deliberate anomalies for testing
    df.loc[10, "basicsalary"] = 450000.00
    df.loc[25, "allowances"] = 300000.00
    df.loc[50, "deductions"] = 0.00
    df["netsalary"] = (df["basicsalary"] + df["allowances"] - df["deductions"]).round(2)
    return df


def run_analysis(df, contamination):
    """Scale numeric features, fit Isolation Forest, project with PCA.
    Returns (results_df, features_used, error_message)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if not is_id_col(c)]

    X = (df[feature_cols].copy()
         .replace([np.inf, -np.inf], np.nan)
         .dropna(axis=1, how="all")
         .fillna(0))
    X = X.loc[:, X.nunique() > 1]

    if X.shape[1] == 0:
        return None, [], "No usable numeric columns found (need at least one numeric column with varying values)."
    if len(X) < 10:
        return None, [], "At least 10 records are required for analysis."

    X_train, _ = train_test_split(X, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(scaler.transform(X_train))

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    results = df.copy().reset_index(drop=True)
    results["flagged"] = preds == -1
    results["anomalystatus"] = np.where(results["flagged"], STATUS_FLAG, STATUS_OK)
    results["anomalyscore"] = model.decision_function(X_scaled).round(4)

    n_comp = min(2, X.shape[1])
    pcs = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)
    results["pc1"] = pcs[:, 0]
    results["pc2"] = pcs[:, 1] if n_comp == 2 else 0.0
    return results, list(X.columns), None


# =====================================================================
# CHARTS
# =====================================================================
def chart_pca(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    n, f = d[~d["flagged"]], d[d["flagged"]]
    ax.scatter(n["pc1"], n["pc2"], s=12, c=GREEN, label="Normal", alpha=0.7)
    ax.scatter(f["pc1"], f["pc2"], s=35, c=RED, marker="x", label="Flagged")
    ax.set_title("Payroll Records (PCA View)", fontweight="bold", fontsize=10)
    ax.set_xlabel("Principal Component 1", fontsize=9)
    ax.set_ylabel("Principal Component 2", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def chart_lga(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Flagged Staff per Local Government", fontweight="bold", fontsize=10)
    if "lga" in d.columns:
        c = d[d["flagged"]]["lga"].astype(str).value_counts().sort_values()
        if not c.empty:
            ax.barh(c.index, c.values, color=DARK_GREEN)
            ax.set_xlabel("Flagged Count", fontsize=9)
            ax.set_ylabel("LGA", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No LGA Flagged Records", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "LGA Column Missing", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def chart_status_bar(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = d["anomalystatus"].value_counts().reindex([STATUS_OK, STATUS_FLAG], fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=[GREEN, RED])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Anomaly Status Count (Bar Chart)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Records", fontsize=9)
    fig.tight_layout()
    return fig


def chart_status_pie(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = d["anomalystatus"].value_counts().reindex([STATUS_OK, STATUS_FLAG], fill_value=0)
    if counts.sum() == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.pie(counts.values, labels=counts.index, colors=[GREEN, RED],
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title("Anomaly Share (%) (Pie Chart)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig


def chart_salary_dist(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    if "netsalary" in d.columns and pd.api.types.is_numeric_dtype(d["netsalary"]):
        ax.hist(d["netsalary"].dropna(), bins=15, color=GREEN, edgecolor="white", alpha=0.85)
        ax.set_title("Net Salary Distribution Frequency", fontweight="bold", fontsize=10)
        ax.set_xlabel("Net Salary (₦)", fontsize=9)
        ax.set_ylabel("Employee Count", fontsize=9)
    else:
        ax.text(0.5, 0.5, "NetSalary Column Missing", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def chart_allowance_scatter(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    ok = all(c in d.columns and pd.api.types.is_numeric_dtype(d[c]) for c in ["allowances", "netsalary"])
    if ok:
        n, f = d[~d["flagged"]], d[d["flagged"]]
        ax.scatter(n["allowances"], n["netsalary"], color=GREEN, label="Normal", alpha=0.6, s=25)
        ax.scatter(f["allowances"], f["netsalary"], color=RED, label="Flagged", marker="X", s=70)
        ax.set_title("Allowances vs Net Salary Outlier Mapping", fontweight="bold", fontsize=10)
        ax.set_xlabel("Allowances (₦)", fontsize=9)
        ax.set_ylabel("Net Salary (₦)", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Required Financial Columns Missing", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


CHARTS = [
    ("pca_view", chart_pca),
    ("flagged_per_lga", chart_lga),
    ("status_bar", chart_status_bar),
    ("status_pie", chart_status_pie),
    ("salary_distribution", chart_salary_dist),
    ("allowance_vs_netsalary", chart_allowance_scatter),
]


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================
# AUDIT PACKAGE
# =====================================================================
def build_audit_zip(results, flagged, investigator, features):
    buf = io.BytesIO()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    drop = ["pc1", "pc2"]
    summary = (
        "AI PAYROLL ANOMALY DETECTION - FORENSIC AUDIT SUMMARY\n"
        "=====================================================\n"
        f"Generated        : {stamp}\n"
        f"Investigator     : {investigator.get('name', '')}\n"
        f"Official Address : {investigator.get('address', '')}\n"
        f"Contact Number   : {investigator.get('contact', '')}\n\n"
        f"Records analyzed : {len(results)}\n"
        f"Records flagged  : {len(flagged)}\n"
        f"Flag rate        : {len(flagged) / max(len(results), 1) * 100:.2f}%\n"
        f"Features used    : {', '.join(features)}\n"
        "Model            : Isolation Forest (200 trees, trained on 80% of records), StandardScaler, PCA(2) for display\n\n"
        "NOTE: A flag is a statistical indicator for review, not proof of fraud.\n"
    )
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        summary_df = pd.DataFrame({
            "ReportItem": ["System", "DateTime", "Investigator", "Contact", "TotalRecords", "FlaggedRecords"],
            "Value": ["Federal Republic of Nigeria Payroll System", stamp, investigator.get("name", ""),
                      investigator.get("contact", ""), len(results), len(flagged)],
        })
        z.writestr("nigeria_payroll_summary.csv", summary_df.to_csv(index=False))
        z.writestr("audit_summary.txt", summary)
        z.writestr("flagged_records.csv", flagged.drop(columns=drop, errors="ignore").to_csv(index=False))
        z.writestr("full_results.csv", results.drop(columns=drop, errors="ignore").to_csv(index=False))
        for name, fn in CHARTS:
            fig = fn(results)
            img = io.BytesIO()
            fig.savefig(img, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            z.writestr(f"charts/{name}.png", img.getvalue())
    return buf.getvalue()


# =====================================================================
# UI HELPERS
# =====================================================================
def go(step):
    st.session_state.step = step
    st.rerun()


def init_state():
    ss = st.session_state
    ss.setdefault("auth_verified", False)
    ss.setdefault("full_name", "")
    ss.setdefault("address", "")
    ss.setdefault("contact", "")
    ss.setdefault("step", 1)
    ss.setdefault("dev_index", 0)
    ss.setdefault("questions", [])
    ss.setdefault("kudos", [])


def sync_auth():
    """Copy live widget values into persistent state BEFORE the nav menu is evaluated,
    so clicking a nav button right after typing never uses stale values."""
    ss = st.session_state
    if "auth_choice" in ss:
        ss.auth_verified = ss.auth_choice == "YES"
    if "in_name" in ss:
        ss.full_name = ss.in_name
    if "in_addr" in ss:
        ss.address = ss.in_addr
    if "in_contact" in ss:
        ss.contact = ss.in_contact


def authorized():
    ss = st.session_state
    return ss.auth_verified and ss.full_name.strip() != ""


def show_logo(candidates, width, fallback):
    for f in candidates:
        if os.path.exists(asset(f)):
            st.image(asset(f), width=width)
            return
    st.write(fallback)


def block(msg, back_step, key):
    st.error(msg)
    if st.button(f"⬅️ Go Back to Step {back_step}", key=key):
        go(back_step)
    st.stop()


def apply_style():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: #f7f9f8; }}
    .nigeria-banner {{ background-color: {GREEN}; height: 10px; border-radius: 4px; margin-bottom: 15px; }}
    .top-header {{ background-color: #ffffff; padding: 15px; border-radius: 12px; border: 2px solid {GREEN};
        box-shadow: 0px 4px 15px rgba(0,135,81,0.15); text-align: center; margin-bottom: 15px; }}
    .top-title {{ color: {GREEN}; font-weight: 900; font-size: 20px; text-transform: uppercase; margin: 5px 0 2px 0; }}
    .top-sub {{ color: {DARK_GREEN}; font-size: 12px; font-weight: 700; margin: 0; }}
    .stButton>button, .stDownloadButton>button, div[data-testid="stFormSubmitButton"]>button {{
        background-color: {GREEN} !important; color: white !important; font-weight: bold !important; width: 100%; }}
    .stButton>button:disabled {{ background-color: #9e9e9e !important; }}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    _, col_tiny = st.columns([19, 1])
    with col_tiny:
        show_logo(["efcc_logo.jpeg", "efcc_logo.jpg", "efcc_logo.png"], 25, "🦅")
    st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

    col_arm, col_title, col_school = st.columns([1, 4, 1])
    with col_arm:
        show_logo(["coat_of_arms.jpeg", "coat_of_arms.jpg", "coat_of_arms.png"], 80, "🇳🇬 Coat of Arms")
    with col_title:
        st.markdown("""
            <div class="top-header" style="margin-top: 5px; padding: 10px;">
                <p class="top-title">International Business School</p>
                <p class="top-sub">AI Payroll Anomaly Detection &amp; EFCC Forensic Audit Portal</p>
            </div>
        """, unsafe_allow_html=True)
    with col_school:
        show_logo(["school_logo.jpeg", "school_logo.jpg", "school_logo.png",
                   "logo.jpeg", "logo.jpg", "logo.png"], 80, "🏫 IBS Logo")
    st.markdown('<div class="nigeria-banner" style="margin-top: 10px;"></div>', unsafe_allow_html=True)


def render_nav():
    ss = st.session_state
    st.markdown("### 📌 Navigation Menu")
    cols = st.columns(6)
    for s_num, s_name in STEP_NAMES.items():
        with cols[s_num - 1]:
            label = f"📍 {s_num}" if ss.step != s_num else f"👉 [{s_num}]"
            if st.button(label, key=f"nav_btn_{s_num}"):
                if s_num > 1 and not authorized():
                    st.warning("⛔ ACCESS LOCKED: Please complete and confirm valid authorization on Step 1 first.")
                elif s_num > 2 and "df" not in ss:
                    st.warning("⛔ ACCESS LOCKED: Please load or generate a dataset on Step 2 first.")
                elif s_num > 3 and "results" not in ss:
                    st.warning("⛔ ACCESS LOCKED: Please run AI Model Analysis on Step 3 first.")
                else:
                    go(s_num)
    st.markdown(f"**Current Active Section:** `{STEP_NAMES[ss.step]}`")
    st.divider()


# =====================================================================
# PAGES
# =====================================================================
def page_authorization():
    ss = st.session_state
    st.header("🔑 Investigator Authorization & Registration")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Authorization Check")
        options = ["-- Please Select --", "YES", "NO"]
        st.radio("Do you have official authorization?", options,
                 index=1 if ss.auth_verified else 0, key="auth_choice")
        choice = ss.auth_choice
        ss.auth_verified = choice == "YES"
        if choice == "-- Please Select --":
            st.info("ℹ️ Please select your authorization status to proceed.")
        elif choice == "NO":
            st.error("⛔ ACCESS DENIED: Proper authorization is required.")
        else:
            st.success("✅ Authorization Confirmed.")

    with col2:
        st.subheader("Investigator Details")
        ss.full_name = st.text_input("Full Name:", value=ss.full_name, key="in_name")
        ss.address = st.text_input("Official Address:", value=ss.address, key="in_addr")
        ss.contact = st.text_input("Contact Number:", value=ss.contact, key="in_contact")

    if authorized():
        st.markdown(f"""
            <div style="background-color:#e8f5e9;padding:15px;border-radius:8px;border-left:5px solid {GREEN};margin-top:15px;">
                <h4 style="color:{GREEN};margin:0 0 5px 0;">Official Session Active</h4>
                <p style="margin:0;color:{DARK_GREEN};font-weight:600;">
                    Welcome, <b>{ss.full_name}</b>. Your credentials have been logged for this forensic session.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("Continue to Data Input & Cleaning ➡️"):
        if not ss.auth_verified:
            st.warning("⚠️ Please complete valid authorization first.")
        elif not ss.full_name.strip():
            st.warning("⚠️ Please enter your full name to proceed.")
        else:
            go(2)


def set_dataset(df):
    """Store dataset; drop stale results if the data changed."""
    ss = st.session_state
    old = ss.get("df")
    if old is None or not old.equals(df):
        ss.pop("results", None)
        ss.pop("flagged_records", None)
    ss.df = df


def clear_dataset():
    for k in ("df", "results", "flagged_records"):
        st.session_state.pop(k, None)


def page_data_input():
    if not authorized():
        block("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 to access this page.", 1, "back2_auth")

    st.header("📂 Payroll Data Input & Cleaning")
    choice = st.radio(
        "Select Dataset Source:",
        ["— Please Select —", "Generate Sample Payroll Dataset (500 Records)", "Upload External Payroll CSV File"],
        horizontal=True, index=0, key="dataset_radio_choice",
    )

    df = None
    if choice == "— Please Select —":
        # Keep any dataset loaded earlier so Back/Next doesn't wipe it.
        if "df" in st.session_state:
            df = st.session_state.df
            st.info("ℹ️ A dataset is already loaded. Pick a source above to replace it.")
        else:
            st.info("ℹ️ Please select a dataset source above to proceed.")
    elif choice.startswith("Generate"):
        df = generate_sample()
        st.success("Sample Nigerian payroll dataset generated with test anomalies.")
    else:
        up = st.file_uploader("Upload Payroll CSV file", type=["csv"])
        if up is not None:
            try:
                raw = pd.read_csv(up)
                df = clean_columns(raw)
                st.success(f"Successfully loaded CSV: {len(df)} rows, {len(df.columns)} columns.")
            except Exception as e:
                clear_dataset()
                st.error(f"Could not read that CSV: {e}")
        else:
            clear_dataset()
            st.info("Awaiting CSV upload...")

    if df is not None:
        set_dataset(df)
        st.subheader("Data Preview")
        st.dataframe(df.head(10))

    st.markdown("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("⬅️ Back to Authorization"):
            go(1)
    with b2:
        can = "df" in st.session_state
        if st.button("Continue to AI Model Analysis ➡️", disabled=not can):
            go(3)
        if not can:
            st.caption("⚠️ Please select or generate a dataset to continue.")


def page_analysis():
    ss = st.session_state
    if not authorized():
        block("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 first.", 1, "back3_auth")
    if "df" not in ss:
        block("⛔ ACCESS LOCKED: Please load or generate data on Step 2 first.", 2, "back3_data")

    st.header("🤖 AI Model Training & Anomaly Detection")
    pct = st.slider("Expected Anomaly Sensitivity Rate (%):", 1, 20, int(ss.get("contam_pct", 7)))
    ss.contam_pct = pct

    results, features, err = run_analysis(ss.df, pct / 100.0)
    if err:
        ss.pop("results", None)
        ss.pop("flagged_records", None)
        st.error(err)
    else:
        ss.results = results
        ss.features_used = features
        ss.flagged_records = results[results["flagged"]].sort_values("anomalyscore").copy()
        n_flag = len(ss.flagged_records)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Analyzed", len(results))
        m2.metric("Flagged", n_flag)
        m3.metric("Normal", len(results) - n_flag)
        m4.metric("Percentage", f"{n_flag / len(results) * 100:.2f}%")
        st.caption(f"Features used by the model: {', '.join(features)}")

        st.subheader("🚩 Flagged Payroll Anomalies")
        st.dataframe(ss.flagged_records.drop(columns=["pc1", "pc2"], errors="ignore"))

    st.markdown("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("⬅️ Back to Data Input"):
            go(2)
    with b2:
        if st.button("Continue to Visualizations & Reports ➡️", disabled=err is not None):
            go(4)


def page_visuals():
    ss = st.session_state
    if not authorized():
        block("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 first.", 1, "back4_auth")
    if "results" not in ss:
        block("⛔ ACCESS LOCKED: Please run AI Model Analysis on Step 3 first.", 3, "back4_ai")

    st.header("📊 Advanced Payroll Analytics & Forensic Graphs")
    results, flagged = ss.results, ss.flagged_records

    st.subheader("1. Dimensional Reduction & Regional Risk Analysis")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(chart_pca(results))
    with c2:
        show_fig(chart_lga(results))
    st.divider()

    st.subheader("2. Anomaly Counts & Proportion Share")
    c3, c4 = st.columns(2)
    with c3:
        show_fig(chart_status_bar(results))
    with c4:
        show_fig(chart_status_pie(results))
    st.divider()

    st.subheader("3. Distributional Depth & Financial Correlation Analysis")
    c5, c6 = st.columns(2)
    with c5:
        show_fig(chart_salary_dist(results))
    with c6:
        show_fig(chart_allowance_scatter(results))
    st.divider()

    st.subheader("📦 Download Audit Package")
    investigator = {"name": ss.full_name, "address": ss.address, "contact": ss.contact}
    zip_bytes = build_audit_zip(results, flagged, investigator, ss.get("features_used", []))
    st.download_button(
        "📥 Download Official Audit ZIP Package",
        data=zip_bytes,
        file_name=f"Nigeria_Payroll_Audit_{datetime.now().strftime('%Y%m%d')}.zip",
        mime="application/zip",
    )
    st.caption("A flag is a statistical indicator for further review, not proof of fraud.")

    st.markdown("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("⬅️ Back to AI Analysis"):
            go(3)
    with b2:
        if st.button("Continue to Q&A & Kudos ➡️"):
            go(5)


def previous_steps_done():
    return authorized() and "results" in st.session_state


def page_qa():
    ss = st.session_state
    if not previous_steps_done():
        block("⛔ ACCESS LOCKED: Please complete previous mandatory steps first.", 1, "back5_lock")

    st.header("❓ Project Q & A Assistant & User Kudos")
    q_input = st.text_input("Ask a question about the system:")
    if q_input:
        q = q_input.lower()
        ans = "I do not have a prepared answer for that question."
        if "algorithm" in q or "isolation" in q:
            ans = "The system uses Isolation Forest, an unsupervised machine learning algorithm."
        elif "training" in q:
            ans = "Training uses 80% of the dataset to learn standard payroll patterns."
        elif "report" in q:
            ans = "The report contains summary statistics, full data results, and flagged anomaly tables packaged into a ZIP file."
        st.info("Assistant Response: " + ans)

    st.divider()
    st.subheader("🌟 Leave Your Kudos")
    with st.form("kudos_form"):
        k_name = st.text_input("Your Name or Department:", value=ss.full_name)
        k_rating = st.select_slider("Rate System Performance:",
                                    options=["Poor", "Fair", "Good", "Very Good", "Excellent"], value="Excellent")
        k_comment = st.text_area("Your Kudos or Comments:", value="Great application!")
        if st.form_submit_button("Submit Kudos"):
            ss.kudos.append({"by": k_name, "rating": k_rating, "comment": k_comment})
            st.success("🎉 Thank you! Your kudos have been logged successfully.")

    st.markdown("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("⬅️ Back to Visualizations"):
            go(4)
    with b2:
        if st.button("Continue to Developers ➡️"):
            go(6)


def page_developers():
    ss = st.session_state
    if not previous_steps_done():
        block("⛔ ACCESS LOCKED: Please complete previous mandatory steps first.", 1, "back6_lock")

    st.header("👥 Meet the Development Team (12 Members)")
    st.markdown("Browse through the team members who built this forensic audit platform.")

    n = len(DEVELOPERS)
    col_prev, _, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Previous Developer"):
            ss.dev_index = (ss.dev_index - 1) % n
            st.rerun()
    with col_next:
        if st.button("Next Developer ➡️"):
            ss.dev_index = (ss.dev_index + 1) % n
            st.rerun()

    dev = DEVELOPERS[ss.dev_index % n]
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        img = dev["image"]
        alt = img.replace(".jpeg", ".jpg") if ".jpeg" in img else img.replace(".jpg", ".jpeg")
        if os.path.exists(asset(img)):
            st.image(asset(img), width=260)
        elif os.path.exists(asset(alt)):
            st.image(asset(alt), width=260)
        else:
            st.info(f"🖼️ [Image file '{img}' not found in folder. Please add your JPEG picture.]")
    with c2:
        st.subheader(dev["name"])
        st.markdown(f"📍 **Address:** {dev['address']}")
        st.markdown(f"🏛️ **State:** {dev['state']}")
        st.markdown(f"🎓 **Education:** {dev['education']}")
        st.markdown(f"💼 **Role:** {dev['role']}")
        st.markdown(f"💍 **Marital Status:** {dev['marital_status']}")
        st.caption(f"Developer {ss.dev_index % n + 1} of {n}")

    st.markdown("")
    if st.button("⬅️ Back to Q&A Assistant & Kudos"):
        go(5)


PAGES = {1: page_authorization, 2: page_data_input, 3: page_analysis,
         4: page_visuals, 5: page_qa, 6: page_developers}


def main():
    st.set_page_config(
        page_title="Federal Republic of Nigeria & IBS - Forensic Audit Portal",
        page_icon="🇳🇬",
        layout="wide",
    )
    init_state()
    sync_auth()
    apply_style()
    render_header()
    render_nav()
    PAGES[st.session_state.step]()
    st.markdown("---")
    st.caption("Federal Republic of Nigeria • Official Portal")


if __name__ == "__main__":
    main()
