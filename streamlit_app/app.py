"""PhaseGuard banking login and customer session shell."""

from __future__ import annotations

import uuid

import streamlit as st

import api_client
from config import API_BASE_URL

st.set_page_config(page_title="PhaseGuard Banking Auth", page_icon="PG", layout="wide")


def _device_fingerprint() -> str:
    if "_device_fp" not in st.session_state:
        st.session_state["_device_fp"] = str(uuid.uuid4())
    return st.session_state["_device_fp"]


def _store_login(payload: dict) -> None:
    st.session_state["session_token"] = payload["session_token"]
    st.session_state["current_user"] = {
        "account_number": payload["account_number"],
        "full_name": payload["full_name"],
        "is_enrolled": payload.get("is_enrolled", False),
        "kyc_status": payload.get("kyc_status", "PENDING"),
    }


st.title("PhaseGuard Layer 2")
st.caption("Banking identity login for voice biometric enrollment and verification")

if st.session_state.get("session_token"):
    ok, profile = api_client.get_me()
    if ok:
        st.session_state["current_user"].update(profile)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customer", profile["full_name"])
        c2.metric("Account", profile["account_number"])
        c3.metric("KYC", profile["kyc_status"])
        c4.metric("Voice enrolled", "Yes" if profile["is_enrolled"] else "No")
        st.info("Use the sidebar pages for enrollment, verification, history, and risk review.")
        if st.button("Sign out"):
            for key in ("session_token", "current_user", "consent_granted", "enroll_session"):
                st.session_state.pop(key, None)
            st.rerun()
        st.stop()
    st.warning("Your saved session is no longer valid. Please sign in again.")
    st.session_state.pop("session_token", None)
    st.session_state.pop("current_user", None)

login_tab, register_tab = st.tabs(["Login", "Register"])

with login_tab:
    with st.form("bank_login"):
        account_number = st.text_input("Account Number", placeholder="UCO0012345678")
        mpin = st.text_input("6-digit MPIN", type="password", max_chars=6)
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        if not account_number or len(mpin) != 6 or not mpin.isdigit():
            st.error("Enter a valid account number and 6-digit MPIN.")
        else:
            ok, payload = api_client.login(account_number, mpin, _device_fingerprint())
            if ok:
                _store_login(payload)
                if payload.get("new_device_login"):
                    st.warning("New device sign-in detected and logged.")
                if payload.get("mpin_rotation_required"):
                    st.warning("MPIN rotation is required before continuing.")
                st.rerun()
            else:
                st.error(payload)

with register_tab:
    st.caption("Staff-assisted customer provisioning. Branch code is the registered home branch captured during KYC.")
    with st.form("bank_register"):
        c1, c2 = st.columns(2)
        payload = {
            "account_number": c1.text_input("Account Number", placeholder="UCO0012345678"),
            "mpin": c2.text_input("Initial 6-digit MPIN", type="password", max_chars=6),
            "full_name": c1.text_input("Full Name"),
            "phone_number": c2.text_input("Mobile Number"),
            "date_of_birth": str(c1.date_input("Date of Birth")),
            "branch_code": c2.text_input("Home Branch Code", placeholder="UCBA"),
            "account_type": c1.selectbox("Account Type", ["SAVINGS", "CURRENT", "NRE"]),
            "email": c2.text_input("Email (optional)") or None,
        }
        registered = st.form_submit_button("Create Customer", type="primary")

    if registered:
        ok, result = api_client.register_customer(payload)
        if ok:
            st.success(f"Customer created: {result['full_name']} ({result['account_number']})")
            st.info("The customer can now log in with account number and MPIN.")
        else:
            st.error(result)

st.divider()
st.caption(f"API: {API_BASE_URL}")
