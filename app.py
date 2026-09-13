"""
AI Complaint Assistant
-----------------------
An AI-powered university complaint management application, built around
three roles:

- Student (primary user): submits a complaint in natural language and
  expects it to be processed and eventually answered.
- University Administrator (secondary user): reviews AI results, manages
  complaint status, can correct AI routing, and replies to students.
- Responsible Department (supporting recipient): logs into its own portal
  and sees only the complaints the AI routed to it, so complaints reach an
  actual recipient rather than just a label.

An LLM (via Groq) classifies each complaint's category, determines
urgency/priority, recommends the responsible department, writes a short
summary, and suggests a next action. Students can look up their complaint
later (by ID + email) to see status and any reply.

Run with:
    streamlit run app.py

Environment variables:
    GROQ_API_KEY      - your Groq API key (or enter it in the sidebar)

Admin password: the first admin to open the Admin Dashboard is prompted to
create a password. That password is stored (hashed, salted) in the local
database, and any logged-in admin can change it later from the dashboard's
Account Settings section. Admins also create/reset department passwords
from the "Manage Department Accounts" section.
"""

import os
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DB_PATH = "complaints.db"
GROQ_MODEL = "llama-3.3-70b-versatile"

CATEGORIES = ["IT", "Finance", "Academic", "Hostel", "Administration", "Library", "Other"]
PRIORITIES = ["Low", "Medium", "High"]
STATUSES = ["Open", "In Progress", "Resolved"]

DEPARTMENT_MAP = {
    "IT": "IT Support Department",
    "Finance": "Finance & Accounts Office",
    "Academic": "Academic Affairs Office",
    "Hostel": "Hostel / Residence Life Office",
    "Administration": "General Administration",
    "Library": "Library Services",
    "Other": "Student Affairs Office",
}

st.set_page_config(page_title="AI Complaint Assistant", page_icon="🚀", layout="wide")

# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------

def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            student_name TEXT,
            student_email TEXT,
            complaint_text TEXT,
            category TEXT,
            priority TEXT,
            summary TEXT,
            department TEXT,
            suggested_action TEXT,
            status TEXT DEFAULT 'Open',
            admin_reply TEXT,
            replied_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_complaint(record: dict) -> int:
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO complaints
            (timestamp, student_name, student_email, complaint_text, category,
             priority, summary, department, suggested_action, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["timestamp"],
            record["student_name"],
            record["student_email"],
            record["complaint_text"],
            record["category"],
            record["priority"],
            record["summary"],
            record["department"],
            record["suggested_action"],
            record.get("status", "Open"),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def load_complaints() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM complaints ORDER BY id DESC", conn)
    conn.close()
    return df


def get_complaint_by_id_email(complaint_id: int, email: str):
    conn = get_conn()
    cur = conn.execute(
        "SELECT * FROM complaints WHERE id = ? AND LOWER(student_email) = LOWER(?)",
        (complaint_id, email),
    )
    cols = [c[0] for c in cur.description]
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return dict(zip(cols, row))


def update_status(complaint_id: int, new_status: str):
    conn = get_conn()
    conn.execute("UPDATE complaints SET status = ? WHERE id = ?", (new_status, complaint_id))
    conn.commit()
    conn.close()


def update_department(complaint_id: int, new_category: str):
    new_department = DEPARTMENT_MAP.get(new_category, new_category)
    conn = get_conn()
    conn.execute(
        "UPDATE complaints SET category = ?, department = ? WHERE id = ?",
        (new_category, new_department, complaint_id),
    )
    conn.commit()
    conn.close()


def save_reply(complaint_id: int, reply_text: str):
    conn = get_conn()
    conn.execute(
        "UPDATE complaints SET admin_reply = ?, replied_at = ? WHERE id = ?",
        (reply_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), complaint_id),
    )
    conn.commit()
    conn.close()


def get_setting(key: str):
    conn = get_conn()
    cur = conn.execute("SELECT value FROM admin_settings WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def set_setting(key: str, value: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO admin_settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Admin password management
# --------------------------------------------------------------------------

def hash_password(password: str, salt: str | None = None) -> str:
    """Return 'salt$hash' using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(hash_password(password, salt), stored)


def is_admin_password_set() -> bool:
    return get_setting("admin_password_hash") is not None


def set_admin_password(new_password: str):
    set_setting("admin_password_hash", hash_password(new_password))


def check_admin_password(password: str) -> bool:
    stored = get_setting("admin_password_hash")
    if stored is None:
        return False
    return verify_password(password, stored)


# --------------------------------------------------------------------------
# Department account management (Responsible Department role)
# --------------------------------------------------------------------------

def _dept_setting_key(dept_key: str) -> str:
    return f"dept_password_hash::{dept_key}"


def is_dept_password_set(dept_key: str) -> bool:
    return get_setting(_dept_setting_key(dept_key)) is not None


def set_dept_password(dept_key: str, new_password: str):
    set_setting(_dept_setting_key(dept_key), hash_password(new_password))


def check_dept_password(dept_key: str, password: str) -> bool:
    stored = get_setting(_dept_setting_key(dept_key))
    if stored is None:
        return False
    return verify_password(password, stored)


# --------------------------------------------------------------------------
# AI analysis (Groq)
# --------------------------------------------------------------------------

def get_groq_client():
    api_key = st.session_state.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def analyze_complaint(client: "Groq", complaint_text: str) -> dict:
    """Send the complaint to Groq and get back category, priority, summary,
    department, and a suggested next action."""

    system_prompt = f"""You are an assistant that triages university student complaints.
Given a complaint, respond ONLY with a JSON object (no markdown, no extra text) with these keys:
- "category": one of {CATEGORIES}
- "priority": one of {PRIORITIES}
- "summary": a short 1-2 sentence summary of the complaint
- "department": the specific department that should handle it (use the category to guide you)
- "suggested_action": one short, concrete next step staff should take to resolve this complaint

Base "priority" on urgency and impact (e.g. safety issues, exam-blocking issues, or financial deadlines are High).
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": complaint_text},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    category = data.get("category", "Other")
    if category not in CATEGORIES:
        category = "Other"
    priority = data.get("priority", "Medium")
    if priority not in PRIORITIES:
        priority = "Medium"
    summary = data.get("summary", complaint_text[:150])
    department = data.get("department") or DEPARTMENT_MAP.get(category, "Student Affairs Office")
    suggested_action = data.get("suggested_action", "Review the complaint and contact the student.")

    return {
        "category": category,
        "priority": priority,
        "summary": summary,
        "department": department,
        "suggested_action": suggested_action,
    }


# --------------------------------------------------------------------------
# UI: Sidebar / navigation
# --------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("🚀 AI Complaint Assistant")
    st.sidebar.markdown("A smarter way to triage and resolve student complaints.")

    page = st.sidebar.radio(
        "Go to",
        ["Submit Complaint", "Track My Complaint", "Admin Dashboard", "Department Portal"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Groq API Key")
    key_input = st.sidebar.text_input(
        "Enter your GROQ_API_KEY",
        type="password",
        value=st.session_state.get("groq_api_key", ""),
        help="You can also set this as an environment variable named GROQ_API_KEY instead.",
    )
    if key_input:
        st.session_state["groq_api_key"] = key_input

    if st.session_state.get("is_admin"):
        st.sidebar.markdown("---")
        st.sidebar.success("Logged in as Admin")
        if st.sidebar.button("Log out", key="admin_logout"):
            st.session_state["is_admin"] = False
            st.rerun()

    if st.session_state.get("dept_key"):
        st.sidebar.markdown("---")
        st.sidebar.success(f"Logged in as: {DEPARTMENT_MAP.get(st.session_state['dept_key'], st.session_state['dept_key'])}")
        if st.sidebar.button("Log out", key="dept_logout"):
            st.session_state["dept_key"] = None
            st.rerun()

    return page


# --------------------------------------------------------------------------
# UI: Student complaint submission page
# --------------------------------------------------------------------------

def render_submission_page():
    st.title("📝 Submit a Complaint")
    st.write("Describe your issue below. Our AI assistant will categorize, prioritize, and route it automatically.")

    with st.form("complaint_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)
        student_name = col_a.text_input("Your Name (optional)", placeholder="e.g. Ali Raza")
        student_email = col_b.text_input(
            "Your Email *",
            placeholder="e.g. ali.raza@university.edu",
            help="Needed so you can track your complaint and receive a reply.",
        )
        complaint_text = st.text_area(
            "Complaint Details",
            placeholder="Describe your complaint in detail...",
            height=180,
        )
        submitted = st.form_submit_button("Submit Complaint")

    if submitted:
        if not complaint_text.strip():
            st.error("Please enter your complaint before submitting.")
            return
        if not student_email.strip():
            st.error("Please enter your email so you can track this complaint later.")
            return

        client = get_groq_client()
        if client is None:
            st.error("No Groq API key found. Please add one in the sidebar or set GROQ_API_KEY.")
            return

        with st.spinner("Analyzing your complaint with AI..."):
            try:
                result = analyze_complaint(client, complaint_text)
            except Exception as e:
                st.error(f"AI analysis failed: {e}")
                return

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_name": student_name.strip() or "Anonymous",
            "student_email": student_email.strip(),
            "complaint_text": complaint_text.strip(),
            "category": result["category"],
            "priority": result["priority"],
            "summary": result["summary"],
            "department": result["department"],
            "suggested_action": result["suggested_action"],
            "status": "Open",
        }
        new_id = save_complaint(record)

        st.success(f"Your complaint has been submitted! Your tracking ID is **#{new_id}** — save it to check your status later.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Category", record["category"])
        col2.metric("Priority", record["priority"])
        col3.metric("Routed To", record["department"])
        st.info(f"**AI Summary:** {record['summary']}")
        st.write(f"**Suggested Action:** {record['suggested_action']}")


# --------------------------------------------------------------------------
# UI: Student self-service tracking page
# --------------------------------------------------------------------------

def render_tracking_page():
    st.title("🔍 Track My Complaint")
    st.write("Enter your complaint ID and the email you submitted it with to see its status.")

    col1, col2 = st.columns(2)
    complaint_id = col1.number_input("Complaint ID", min_value=1, step=1, value=1)
    email = col2.text_input("Email used at submission")

    if st.button("Check Status"):
        record = get_complaint_by_id_email(int(complaint_id), email.strip())
        if record is None:
            st.error("No matching complaint found. Double-check your ID and email.")
            return

        st.subheader(f"Complaint #{record['id']} — {record['status']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", record["category"])
        c2.metric("Priority", record["priority"])
        c3.metric("Department", record["department"])
        st.write(f"**Summary:** {record['summary']}")

        if record["admin_reply"]:
            st.success(f"**Reply from {record['department']}** ({record['replied_at']}):\n\n{record['admin_reply']}")
        else:
            st.info("No reply yet. Please check back later.")


# --------------------------------------------------------------------------
# UI: Admin login gate
# --------------------------------------------------------------------------

def render_admin_login():
    if not is_admin_password_set():
        st.title("🔐 Set Up Admin Access")
        st.write(
            "No admin password has been created yet. Set one now — you'll need it "
            "(and can change it later) to access the dashboard."
        )
        with st.form("admin_setup_form"):
            pw1 = st.text_input("New Admin Password", type="password")
            pw2 = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Password")

        if submitted:
            if len(pw1) < 6:
                st.error("Password must be at least 6 characters.")
            elif pw1 != pw2:
                st.error("Passwords do not match.")
            else:
                set_admin_password(pw1)
                st.session_state["is_admin"] = True
                st.success("Admin password created. Logging you in...")
                st.rerun()
        return

    st.title("🔐 Admin Login")
    st.write("This dashboard is restricted to authorized university staff.")

    with st.form("admin_login_form"):
        password = st.text_input("Admin Password", type="password")
        submitted = st.form_submit_button("Log In")

    if submitted:
        if check_admin_password(password):
            st.session_state["is_admin"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")


# --------------------------------------------------------------------------
# UI: Admin dashboard page
# --------------------------------------------------------------------------

def render_dashboard_page():
    if not st.session_state.get("is_admin"):
        render_admin_login()
        return

    st.title("📊 Admin Dashboard")

    with st.expander("⚙️ Account Settings — Change Admin Password"):
        with st.form("change_password_form"):
            current_pw = st.text_input("Current Password", type="password")
            new_pw1 = st.text_input("New Password", type="password")
            new_pw2 = st.text_input("Confirm New Password", type="password")
            change_submitted = st.form_submit_button("Update Password")

        if change_submitted:
            if not check_admin_password(current_pw):
                st.error("Current password is incorrect.")
            elif len(new_pw1) < 6:
                st.error("New password must be at least 6 characters.")
            elif new_pw1 != new_pw2:
                st.error("New passwords do not match.")
            else:
                set_admin_password(new_pw1)
                st.success("Password updated successfully.")

    with st.expander("🏢 Manage Department Accounts"):
        st.write(
            "Give each responsible department its own login so it can view and act on "
            "only the complaints routed to it. Setting a password here creates or resets that account."
        )
        with st.form("dept_account_form"):
            dept_choice = st.selectbox(
                "Department",
                CATEGORIES,
                format_func=lambda k: DEPARTMENT_MAP.get(k, k),
            )
            dept_pw1 = st.text_input("New Department Password", type="password")
            dept_pw2 = st.text_input("Confirm Department Password", type="password")
            dept_submitted = st.form_submit_button("Set / Reset Password")

        if dept_submitted:
            if len(dept_pw1) < 6:
                st.error("Password must be at least 6 characters.")
            elif dept_pw1 != dept_pw2:
                st.error("Passwords do not match.")
            else:
                set_dept_password(dept_choice, dept_pw1)
                st.success(f"Password set for {DEPARTMENT_MAP.get(dept_choice, dept_choice)}.")

        st.markdown("**Current account status:**")
        status_rows = [
            {"Department": DEPARTMENT_MAP.get(k, k), "Account Set Up": "✅ Yes" if is_dept_password_set(k) else "❌ No"}
            for k in CATEGORIES
        ]
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    df = load_complaints()

    if df.empty:
        st.info("No complaints have been submitted yet.")
        return

    # --- Top-level metrics ---
    total = len(df)
    high_priority = (df["priority"] == "High").sum()
    open_count = (df["status"] == "Open").sum()
    departments = df["department"].nunique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Complaints", total)
    m2.metric("High Priority", high_priority)
    m3.metric("Open Complaints", open_count)
    m4.metric("Departments Involved", departments)

    st.markdown("---")

    # --- Filters ---
    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        cat_filter = f1.multiselect("Category", sorted(df["category"].unique()))
        pri_filter = f2.multiselect("Priority", sorted(df["priority"].unique()))
        status_filter = f3.multiselect("Status", sorted(df["status"].unique()))

    filtered_df = df.copy()
    if cat_filter:
        filtered_df = filtered_df[filtered_df["category"].isin(cat_filter)]
    if pri_filter:
        filtered_df = filtered_df[filtered_df["priority"].isin(pri_filter)]
    if status_filter:
        filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]

    # --- Charts ---
    c1, c2 = st.columns(2)
    with c1:
        cat_counts = filtered_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.bar(cat_counts, x="Category", y="Count", title="Complaints by Category", color="Category")
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        pri_counts = filtered_df["priority"].value_counts().reset_index()
        pri_counts.columns = ["Priority", "Count"]
        color_map = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}
        fig_pri = px.pie(
            pri_counts, names="Priority", values="Count", title="Priority Breakdown",
            color="Priority", color_discrete_map=color_map,
        )
        st.plotly_chart(fig_pri, use_container_width=True)

    dept_counts = filtered_df["department"].value_counts().reset_index()
    dept_counts.columns = ["Department", "Count"]
    fig_dept = px.bar(
        dept_counts, x="Count", y="Department", orientation="h",
        title="Complaints Routed by Department",
    )
    st.plotly_chart(fig_dept, use_container_width=True)

    st.markdown("---")
    st.subheader("All Complaints")

    for _, row in filtered_df.iterrows():
        priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(row["priority"], "⚪")
        reply_flag = "💬" if row["admin_reply"] else ""
        with st.expander(
            f"{priority_emoji}{reply_flag} #{row['id']} — {row['category']} — "
            f"{row['student_name']} ({row['timestamp']})"
        ):
            st.write(f"**Student Email:** {row['student_email']}")
            st.write(f"**Full Complaint:** {row['complaint_text']}")
            st.write(f"**AI Summary:** {row['summary']}")
            st.write(f"**Department:** {row['department']}")
            st.write(f"**Priority:** {row['priority']}")
            st.write(f"**Suggested Action:** {row['suggested_action']}")

            current_status = row["status"] if row["status"] in STATUSES else "Open"
            new_status = st.selectbox(
                "Status",
                STATUSES,
                index=STATUSES.index(current_status),
                key=f"status_{row['id']}",
            )
            if new_status != row["status"]:
                update_status(row["id"], new_status)
                st.rerun()

            current_category = row["category"] if row["category"] in CATEGORIES else "Other"
            new_category = st.selectbox(
                "Reassign Department (if AI misrouted this)",
                CATEGORIES,
                index=CATEGORIES.index(current_category),
                format_func=lambda k: DEPARTMENT_MAP.get(k, k),
                key=f"category_{row['id']}",
            )
            if new_category != row["category"]:
                update_department(row["id"], new_category)
                st.rerun()

            st.markdown("**Reply to Student**")
            existing_reply = row["admin_reply"] or ""
            reply_text = st.text_area(
                "Message",
                value=existing_reply,
                key=f"reply_{row['id']}",
                height=100,
                label_visibility="collapsed",
                placeholder="Write a reply the student will see when they track their complaint...",
            )
            if st.button("Send Reply", key=f"send_{row['id']}"):
                if reply_text.strip():
                    save_reply(row["id"], reply_text.strip())
                    st.success("Reply saved.")
                    st.rerun()
                else:
                    st.warning("Reply cannot be empty.")

    st.download_button(
        "Download all complaints as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="complaints_export.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# UI: Department Portal (responsible department role)
# --------------------------------------------------------------------------

def render_department_login():
    st.title("🏢 Department Portal Login")
    st.write("Sign in as a responsible department to view and act on the complaints routed to you.")

    with st.form("dept_login_form"):
        dept_choice = st.selectbox(
            "Department",
            CATEGORIES,
            format_func=lambda k: DEPARTMENT_MAP.get(k, k),
        )
        password = st.text_input("Department Password", type="password")
        submitted = st.form_submit_button("Log In")

    if submitted:
        if not is_dept_password_set(dept_choice):
            st.error(
                "This department doesn't have an account yet. Ask the university "
                "administrator to set one up from the Admin Dashboard."
            )
        elif check_dept_password(dept_choice, password):
            st.session_state["dept_key"] = dept_choice
            st.rerun()
        else:
            st.error("Incorrect password.")


def render_department_dashboard(dept_key: str):
    dept_name = DEPARTMENT_MAP.get(dept_key, dept_key)
    st.title(f"🏢 {dept_name} — Complaint Queue")
    st.caption("You're seeing only the complaints the AI routed to your department.")

    df = load_complaints()
    dept_df = df[df["category"] == dept_key]

    if dept_df.empty:
        st.info("No complaints have been routed to your department yet.")
        return

    total = len(dept_df)
    high_priority = (dept_df["priority"] == "High").sum()
    open_count = (dept_df["status"] == "Open").sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Complaints", total)
    m2.metric("High Priority", high_priority)
    m3.metric("Open", open_count)

    st.markdown("---")

    pri_order = {"High": 0, "Medium": 1, "Low": 2}
    dept_df = dept_df.assign(_sort=dept_df["priority"].map(pri_order)).sort_values(
        ["_sort", "id"], ascending=[True, False]
    )

    for _, row in dept_df.iterrows():
        priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(row["priority"], "⚪")
        reply_flag = "💬" if row["admin_reply"] else ""
        with st.expander(
            f"{priority_emoji}{reply_flag} #{row['id']} — {row['student_name']} ({row['timestamp']}) — {row['status']}"
        ):
            st.write(f"**Full Complaint:** {row['complaint_text']}")
            st.write(f"**AI Summary:** {row['summary']}")
            st.write(f"**Suggested Action:** {row['suggested_action']}")

            current_status = row["status"] if row["status"] in STATUSES else "Open"
            new_status = st.selectbox(
                "Status",
                STATUSES,
                index=STATUSES.index(current_status),
                key=f"dept_status_{row['id']}",
            )
            if new_status != row["status"]:
                update_status(row["id"], new_status)
                st.rerun()

            st.markdown("**Reply to Student**")
            existing_reply = row["admin_reply"] or ""
            reply_text = st.text_area(
                "Message",
                value=existing_reply,
                key=f"dept_reply_{row['id']}",
                height=100,
                label_visibility="collapsed",
                placeholder="Write a reply the student will see when they track their complaint...",
            )
            if st.button("Send Reply", key=f"dept_send_{row['id']}"):
                if reply_text.strip():
                    save_reply(row["id"], reply_text.strip())
                    st.success("Reply saved.")
                    st.rerun()
                else:
                    st.warning("Reply cannot be empty.")

    st.download_button(
        "Download my department's complaints as CSV",
        data=dept_df.drop(columns=["_sort"]).to_csv(index=False).encode("utf-8"),
        file_name=f"{dept_key}_complaints_export.csv",
        mime="text/csv",
    )


def render_department_page():
    if not st.session_state.get("dept_key"):
        render_department_login()
    else:
        render_department_dashboard(st.session_state["dept_key"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    init_db()
    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False
    if "dept_key" not in st.session_state:
        st.session_state["dept_key"] = None

    page = render_sidebar()

    if page == "Submit Complaint":
        render_submission_page()
    elif page == "Track My Complaint":
        render_tracking_page()
    elif page == "Department Portal":
        render_department_page()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
