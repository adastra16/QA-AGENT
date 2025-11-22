import streamlit as st
import requests
import json
import os

# example — replace this with your real ngrok URL (no angle brackets)
BACKEND_URL = "https://unasked-busiest-yahir.ngrok-free.dev"





# Local path to a saved generated script (for quick demo / optional)
LOCAL_SCRIPT_PATH = r"C:\Users\RUPSA NANDA\OneDrive\Desktop\QA-AGENT\uploaded_docs\selenium_.py"

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="QA-Agent Dashboard",
    layout="wide"
)

st.title("🤖 QA-Agent – Automated Testcase & Script Generator")
st.markdown("This UI lets you upload documents, build a knowledge base, generate testcases and auto-create Selenium scripts.")

st.sidebar.header("📁 Upload Documents")

# Keep uploaded paths in session
if "uploaded_paths" not in st.session_state:
    st.session_state["uploaded_paths"] = []

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.sidebar.file_uploader(
    "Upload support docs (HTML, MD, TXT, JSON, PDF)",
    accept_multiple_files=True
)

if uploaded_files:
    files_payload = []
    for f in uploaded_files:
        files_payload.append(("files", (f.name, f.getvalue(), f.type)))

    resp = requests.post(f"{BACKEND_URL}/upload_files/", files=files_payload)
    if resp.ok:
        saved = resp.json().get("saved", [])
        for s in saved:
            if s["path"] not in st.session_state["uploaded_paths"]:
                st.session_state["uploaded_paths"].append(s["path"])
        st.sidebar.success("Files uploaded to backend")
    else:
        st.sidebar.error("Upload failed")

# Show uploaded files
st.sidebar.subheader("Uploaded Files")
for p in st.session_state["uploaded_paths"]:
    st.sidebar.write(f"- {p}")

if st.sidebar.button("Clear uploaded"):
    st.session_state["uploaded_paths"] = []

# ---------------- BUILD KB ----------------
st.header("1️⃣ Build Knowledge Base")

if st.button("Build KB"):
    if not st.session_state["uploaded_paths"]:
        st.warning("Upload files first.")
    else:
        with st.spinner("Building knowledge base..."):
            payload = {
                "file_paths": st.session_state["uploaded_paths"],
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
            resp = requests.post(f"{BACKEND_URL}/build_kb/", json=payload)

            if resp.ok:
                st.success(resp.json())
            else:
                st.error(resp.text)

# ---------------- GENERATE TEST CASES ----------------
st.header("2️⃣ Generate Test Cases")

query = st.text_area("Enter your feature / requirement text:", height=80)
top_k = st.number_input("Top-K context retrieval:", 1, 10, 3)

if st.button("Generate Testcases"):
    if not query.strip():
        st.warning("Enter a requirement or feature.")
    else:
        with st.spinner("Generating test cases..."):
            resp = requests.post(
                f"{BACKEND_URL}/generate_testcases/",
                json={"query": query, "top_k": top_k}
            )
            if resp.ok:
                data = resp.json()
                st.session_state["generated"] = data.get("generated", [])
                st.success("Testcases generated!")
            else:
                st.error(resp.text)

# Display testcases
if st.session_state.get("generated"):
    st.subheader("Generated Test Cases")
    for item in st.session_state["generated"]:
        st.code(json.dumps(item["payload"], indent=2))


# ---------------- GENERATE SELENIUM SCRIPT ----------------
st.header("3️⃣ Generate Selenium Script")

if st.button("Refresh Testcases from Backend"):
    resp = requests.get(f"{BACKEND_URL}/list_testcases/")
    if resp.ok:
        st.session_state["backend_cases"] = resp.json().get("items", [])
        st.success("Fetched testcases!")
    else:
        st.error("Failed to fetch")

cases = st.session_state.get("backend_cases", [])

if cases:
    selected_id = st.selectbox(
        "Choose a testcase ID:",
        [c["id"] for c in cases]
    )

    if st.button("Generate Selenium Script"):
        with st.spinner("Generating Selenium script..."):
            resp = requests.post(
                f"{BACKEND_URL}/generate_selenium_script/",
                json={"testcase_id": selected_id}
            )
            if resp.ok:
                data = resp.json()
                # Accept either key: "selenium_script" (preferred) OR "script" (backend returns this)
                script_text = data.get("selenium_script") or data.get("script") or ""

                if script_text:
                    st.subheader("Generated Script")
                    st.code(script_text, language="python")

                    # Download button: allow saving directly from UI
                    st.download_button(
                        label="Download script (.py)",
                        data=script_text,
                        file_name=f"selenium_{selected_id}.py",
                        mime="text/x-python"
                    )

                    # Optional: show local path and link (if file already exists on disk)
                    if os.path.exists(LOCAL_SCRIPT_PATH):
                        st.markdown(f"Local script file detected: `{LOCAL_SCRIPT_PATH}`")
                        st.markdown(f"[Open local script](file:///{LOCAL_SCRIPT_PATH.replace(os.sep, '/')})")
                else:
                    st.error("Backend returned OK but no script text was found in the response.")
                    st.write(data)
            else:
                st.error(resp.text)
else:
    st.info("No testcases yet. Generate above.")

st.markdown("---")
st.caption(f"Backend running at: {BACKEND_URL}")
