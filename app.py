import streamlit as st
from pypdf import PdfReader
from google import genai                        # NEW SDK — uses stable v1 endpoint
from google.genai import types
from ddgs import DDGS
from datetime import datetime
from io import StringIO
import csv
import time

st.set_page_config(page_title="SupplyChainGPT", page_icon="📦", layout="wide")

# ====================== CONFIGURE GEMINI ONCE (new SDK) ======================
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API. Check your Streamlit secret: {e}")
    st.stop()

GEMINI_MODEL = "gemini-2.0-flash"   # Stable model on v1 endpoint

# ====================== KNOWLEDGE BASE ======================
knowledge_base = [
    {"title": "Incoterms 2020 Overview",       "category": "Incoterms",        "content": "Incoterms 2020 are 11 rules by the International Chamber of Commerce that define buyer/seller responsibilities for delivery, risk transfer, costs, and insurance in international trade."},
    {"title": "EXW - Ex Works",                "category": "Incoterms",        "content": "Seller makes goods available at their premises. Buyer handles loading, transport, insurance, export/import clearance, and all risks from seller's door."},
    {"title": "FOB - Free on Board",           "category": "Incoterms",        "content": "Seller delivers goods on board the vessel at the named port. Risk transfers when goods are on board. Seller handles export clearance."},
    {"title": "DDP - Delivered Duty Paid",     "category": "Incoterms",        "content": "Seller bears all costs and risks (including import duties, taxes, and customs clearance) until goods are delivered to buyer's premises."},
    {"title": "CIF - Cost, Insurance and Freight", "category": "Incoterms",   "content": "Seller pays for transport and insurance to the named port. Risk transfers when goods are on board the vessel."},
    {"title": "US Customs & CMMC Compliance",  "category": "Compliance",       "content": "US imports require accurate HTS codes, valuation, country of origin, and ISF filing. Defense suppliers must meet CMMC 2.0."},
    {"title": "EU Import Regulations",         "category": "Compliance",       "content": "EU requires CBAM reporting, TARIC codes, and proof of origin. Incoterms must align with Union Customs Code."},
    {"title": "Common Bill of Lading Errors",  "category": "Compliance",       "content": "Frequent issues: mismatched Incoterms, missing seals, incorrect consignee, wrong HS codes."},
    {"title": "Supply Chain Risk Management",  "category": "Risk Management",  "content": "Key risks include supplier bankruptcy, geopolitical events, port congestion, currency fluctuation."},
    {"title": "Geopolitical Disruptions 2026", "category": "Risk Management",  "content": "Red Sea/Suez issues, US-China tensions, and nearshoring trends are shifting global shipping routes."},
    {"title": "Sustainability & ESG",          "category": "Sustainability",   "content": "Buyers now demand Scope 3 carbon reporting. EU CSRD and US SEC climate rules are coming into force."},
    {"title": "Digital Tools & Traceability",  "category": "Logistics",        "content": "Blockchain, IoT sensors, and AI forecasting reduce documentation errors by 70%."},
]

# ====================== KEYWORD SEARCH (no embedding API needed) ======================
def _keyword_score(query: str, doc: str) -> float:
    query_words = set(query.lower().split())
    doc_lower = doc.lower()
    if not query_words:
        return 0.0
    matches = sum(1 for w in query_words if w in doc_lower)
    phrase_bonus = 0.3 if query.lower() in doc_lower else 0.0
    return min(matches / len(query_words) + phrase_bonus, 1.0)


def semantic_search(query, selected_categories, min_relevance, top_k=6):
    filtered_kb = knowledge_base
    if selected_categories and "All" not in selected_categories:
        filtered_kb = [i for i in knowledge_base if i["category"] in selected_categories]

    scored = sorted(
        [( item, _keyword_score(query, f"{item['title']} {item['content']}") ) for item in filtered_kb],
        key=lambda x: x[1], reverse=True
    )
    results = []
    for item, score in scored[:top_k]:
        relevance = round(score * 100, 1)
        if relevance >= min_relevance:
            r = item.copy()
            r["relevance"] = relevance
            results.append(r)
    return results


# ====================== GEMINI CALL (new SDK) with retry ======================
def call_gemini(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "resource" in err.lower():
                wait = 2 ** attempt
                st.warning(f"⏳ Rate limit hit. Retrying in {wait}s... ({attempt+1}/{retries})")
                time.sleep(wait)
            else:
                return f"❌ Gemini error: {e}"
    return "❌ Gemini quota exceeded. Wait ~1 minute and retry, or upgrade at https://ai.google.dev/pricing"


# ====================== AI FUNCTIONS ======================
def generate_ai_insights(query, results):
    if not results:
        return "No relevant information found."
    context = "\n\n---\n\n".join([
        f"**{r['title']}** ({r['category']}, {r['relevance']}% relevant)\n{r['content']}"
        for r in results
    ])
    return call_gemini(f"""User query: "{query}"
Relevant knowledge:
{context}
Create a professional AI analysis in clean Markdown:
- **Summary** (2-3 sentences)
- **Key Insights**
- **Potential Risks** (with severity)
- **Trends & Opportunities**
- **Recommended Actions** (3-5 practical steps)""")


def real_time_web_search(query, max_results=6):
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        st.error(f"❌ Web search failed: {e}")
        return []


def generate_web_insights(query, web_results):
    if not web_results:
        return "No recent web results found."
    context = "\n\n".join([f"**{r['title']}**\n{r['body']}\nSource: {r['href']}" for r in web_results])
    return call_gemini(f"""User query: "{query}"
Latest web results:
{context}
Summarize in clean Markdown:
- **Key Updates**
- **Impact on SMEs**
- **Recommended Actions**""")


def chat_with_memory(user_message):
    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
    return call_gemini(f"""You are SupplyChainGPT, a senior supply chain expert.
Previous conversation:
{history}
New question: {user_message}
Answer helpfully, accurately, and concisely.""")


def multi_document_audit(uploaded_files):
    texts = []
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            text = "".join(p.extract_text() or "" for p in reader.pages)
            texts.append(f"--- DOCUMENT: {file.name} ---\n{text[:6000]}")
        except Exception as e:
            st.warning(f"⚠️ Could not read {file.name}: {e}")
    if not texts:
        return "❌ No readable content found in the uploaded files."
    combined = "\n\n".join(texts)
    return call_gemini(f"""Analyze these {len(uploaded_files)} shipping documents:
{combined[:20000]}
Output ONLY clean Markdown with:
1. **Summary Table** (one row per document)
2. **Cross-Document Comparison**
3. **Overall Risk Rating**
4. **Recommended Fixes**""")


# ====================== MAIN APP ======================
st.title("🚢 SupplyChainGPT")
st.markdown("**Smart Search • Real-time News • Chat Memory • Multi-Doc Auditor** — Built for SMEs 2026")

with st.sidebar:
    st.header("📦 Tools")
    mode = st.radio("Choose tool:", [
        "🔍 Smart Search",
        "🌐 Real-time Web Search",
        "💬 Chat with Memory",
        "📄 Multi-Document Auditor"
    ], label_visibility="collapsed")
    st.divider()
    if mode == "🔍 Smart Search":
        advanced_mode = st.checkbox("Advanced Mode (AI Insights)", value=True)

# ====================== SMART SEARCH ======================
if mode == "🔍 Smart Search":
    st.subheader("🔍 Ask anything about supply chains, Incoterms, compliance, or logistics")
    st.caption("💡 Try these:")
    cols = st.columns(4)
    suggestions = [
        "Best Incoterm for sea freight to Europe?",
        "How to avoid customs delays on US imports?",
        "Current supply chain risks in 2026",
        "What does DDP actually mean for me?"
    ]
    for i, sug in enumerate(suggestions):
        if cols[i % 4].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.search_query = sug
            st.session_state.trigger_search = True
            st.rerun()

    query = st.text_input("Your supply chain question:",
                          placeholder="e.g. What are the risks of using EXW for international shipments?",
                          key="search_query")

    if st.button("🔍 Search", type="primary", use_container_width=True) or st.session_state.get("trigger_search", False):
        if "trigger_search" in st.session_state:
            del st.session_state.trigger_search
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching knowledge base..."):
                results = semantic_search(query, ["All"], 10)

            if "search_history" not in st.session_state:
                st.session_state.search_history = []
            st.session_state.search_history.append({"query": query})
            if len(st.session_state.search_history) > 10:
                st.session_state.search_history.pop(0)

            if results:
                st.success(f"Found {len(results)} relevant results")
                for r in results:
                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{r['title']}**")
                            st.caption(f"{r['category']} • {r['relevance']}% relevant")
                        with col2:
                            st.caption(f"⭐ {r['relevance']}")
                        st.markdown(r["content"])

                if advanced_mode:
                    with st.spinner("Generating AI insights..."):
                        insights = generate_ai_insights(query, results)
                    st.subheader("🤖 AI-Powered Insights & Analysis")
                    st.markdown(insights)

                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(["Title", "Category", "Relevance (%)", "Content"])
                for r in results:
                    writer.writerow([r["title"], r["category"], r["relevance"], r["content"]])
                st.download_button("📥 Export Results as CSV", data=output.getvalue(),
                                   file_name=f"supplychain_search_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv")
            else:
                st.info("No results found. Try keywords like 'DDP', 'customs', 'risk', 'FOB', etc.")

# ====================== REAL-TIME WEB SEARCH ======================
elif mode == "🌐 Real-time Web Search":
    st.subheader("🌐 Real-time Supply Chain News & Disruptions")
    query = st.text_input("What disruptions or news are you tracking?",
                          placeholder="e.g. Red Sea shipping disruptions March 2026",
                          key="web_query")
    if st.button("🔎 Search Web", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching the web..."):
                web_results = real_time_web_search(query)
            if web_results:
                for r in web_results:
                    with st.expander(f"📌 {r['title']}"):
                        st.write(r['body'])
                        st.markdown(f"[🔗 Open source]({r['href']})")
                with st.spinner("Generating AI summary..."):
                    insights = generate_web_insights(query, web_results)
                st.subheader("🤖 AI Analysis of Latest News")
                st.markdown(insights)
            else:
                st.info("No web results found. Try a different search term.")

# ====================== CHAT WITH MEMORY ======================
elif mode == "💬 Chat with Memory":
    st.subheader("💬 Chat with SupplyChainGPT")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about Incoterms, risks, compliance..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking..."):
            response_text = chat_with_memory(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

# ====================== MULTI-DOCUMENT AUDITOR ======================
else:
    st.subheader("📄 Multi-Document Auditor + Comparison")
    uploaded_files = st.file_uploader("Upload one or more PDF documents", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("🔍 Analyze & Compare All Documents", type="primary", use_container_width=True):
        with st.spinner("Analyzing documents..."):
            report = multi_document_audit(uploaded_files)
        st.markdown("### 📋 Multi-Document Audit & Comparison Report")
        st.markdown(report)
        st.download_button("📥 Download Full Report", data=report,
                           file_name=f"multi_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                           mime="text/markdown")

st.divider()
st.caption("SupplyChainGPT v3 • Real-time + Chat + Multi-Doc • Open-source • Built by Sid Vithal • 2026")
