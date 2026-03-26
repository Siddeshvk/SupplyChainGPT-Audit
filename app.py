import streamlit as st
from pypdf import PdfReader
from groq import Groq
from ddgs import DDGS
from datetime import datetime
from io import StringIO
import csv

st.set_page_config(page_title="SupplyChainGPT", page_icon="📦", layout="wide")

# ====================== GROQ CLIENT (free tier: 30 req/min, 14,400/day) ======================
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    GROQ_MODEL = "llama-3.1-8b-instant"  # Fast, free, generous limits
except Exception as e:
    st.error(f"❌ Failed to connect to Groq. Check your GROQ_API_KEY secret: {e}")
    st.stop()

# ====================== KNOWLEDGE BASE ======================
knowledge_base = [
    {"title": "Incoterms 2020 Overview",           "category": "Incoterms",       "content": "Incoterms 2020 are 11 rules by the International Chamber of Commerce that define buyer/seller responsibilities for delivery, risk transfer, costs, and insurance in international trade."},
    {"title": "EXW - Ex Works",                    "category": "Incoterms",       "content": "Seller makes goods available at their premises. Buyer handles loading, transport, insurance, export/import clearance, and all risks from seller's door."},
    {"title": "FCA - Free Carrier",                "category": "Incoterms",       "content": "Seller delivers goods to a named carrier or another nominated party. Risk passes to buyer when goods are handed to carrier. Suitable for all transport modes."},
    {"title": "FOB - Free on Board",               "category": "Incoterms",       "content": "Seller delivers goods on board the vessel at the named port. Risk transfers when goods are on board. Seller handles export clearance."},
    {"title": "DDP - Delivered Duty Paid",         "category": "Incoterms",       "content": "Seller bears all costs and risks (including import duties, taxes, and customs clearance) until goods are delivered to buyer's premises."},
    {"title": "CIF - Cost, Insurance and Freight", "category": "Incoterms",       "content": "Seller pays for transport and insurance to the named port. Risk transfers when goods are on board the vessel."},
    {"title": "DAP - Delivered at Place",          "category": "Incoterms",       "content": "Seller delivers goods to a named destination. Buyer handles import clearance and duties. Seller bears all risks during transport."},
    {"title": "US Customs & CMMC Compliance",      "category": "Compliance",      "content": "US imports require accurate HTS codes, valuation, country of origin, and ISF filing. Defense suppliers must meet CMMC 2.0."},
    {"title": "EU Import Regulations",             "category": "Compliance",      "content": "EU requires CBAM reporting, TARIC codes, and proof of origin. Incoterms must align with Union Customs Code."},
    {"title": "Common Bill of Lading Errors",      "category": "Compliance",      "content": "Frequent issues: mismatched Incoterms, missing seals, incorrect consignee, wrong HS codes, missing notify party."},
    {"title": "Letter of Credit (LC)",             "category": "Compliance",      "content": "A bank guarantee ensuring seller receives payment if documentary conditions are met. Requires strict document compliance — Incoterms must align exactly."},
    {"title": "Supply Chain Risk Management",      "category": "Risk Management", "content": "Key risks include supplier bankruptcy, geopolitical events, port congestion, currency fluctuation, and single-source dependencies."},
    {"title": "Geopolitical Disruptions 2026",     "category": "Risk Management", "content": "Red Sea/Suez issues, US-China tensions, and nearshoring trends are shifting global shipping routes and lead times."},
    {"title": "Dual Sourcing Strategy",            "category": "Risk Management", "content": "Using two or more suppliers for critical components reduces risk of supply disruption but increases procurement complexity."},
    {"title": "Sustainability & ESG",             "category": "Sustainability",   "content": "Buyers now demand Scope 3 carbon reporting. EU CSRD and US SEC climate disclosure rules require supply chain transparency."},
    {"title": "Digital Tools & Traceability",      "category": "Logistics",       "content": "Blockchain, IoT sensors, and AI forecasting reduce documentation errors by up to 70% and improve end-to-end visibility."},
    {"title": "Freight Modes Comparison",          "category": "Logistics",       "content": "Air freight: fast, expensive, low volume. Sea freight: slow, cheap, high volume. Rail (Asia-Europe): middle ground. Road: flexible for last mile."},
    {"title": "Customs Valuation Methods",         "category": "Compliance",      "content": "WTO values goods primarily on transaction value (invoice price). Adjustments made for royalties, assists, and related-party transactions."},
]

# ====================== KEYWORD SEARCH ======================
def keyword_score(query: str, doc: str) -> float:
    q_words = set(query.lower().split())
    doc_lower = doc.lower()
    if not q_words:
        return 0.0
    hits = sum(1 for w in q_words if w in doc_lower)
    bonus = 0.3 if query.lower() in doc_lower else 0.0
    return min(hits / len(q_words) + bonus, 1.0)


def smart_search(query: str, min_relevance: float = 10.0, top_k: int = 6):
    scored = sorted(
        [(item, keyword_score(query, f"{item['title']} {item['content']}")) for item in knowledge_base],
        key=lambda x: x[1], reverse=True
    )
    results = []
    for item, score in scored[:top_k]:
        rel = round(score * 100, 1)
        if rel >= min_relevance:
            r = item.copy()
            r["relevance"] = rel
            results.append(r)
    return results


# ====================== GROQ AI CALL ======================
def call_ai(prompt: str, system: str = "You are SupplyChainGPT, a senior supply chain expert. Be concise, accurate, and practical.") -> str:
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI error: {e}"


# ====================== WEB SEARCH ======================
def web_search(query: str, max_results: int = 6):
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        st.error(f"❌ Web search failed: {e}")
        return []


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
    st.caption("⚡ Powered by Groq (free tier)")

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
                results = smart_search(query)

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
                        context = "\n\n".join([f"[{r['title']} — {r['category']}]\n{r['content']}" for r in results])
                        insights = call_ai(f"""Query: "{query}"

Knowledge base matches:
{context}

Provide a professional supply chain analysis in clean Markdown covering:
- **Summary** (2-3 sentences)
- **Key Insights**
- **Potential Risks** (with severity: Low/Medium/High)
- **Recommended Actions** (3-5 practical steps)""")
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
                st.info("No results found. Try keywords like 'DDP', 'customs', 'risk', 'FOB', 'freight'.")

# ====================== REAL-TIME WEB SEARCH ======================
elif mode == "🌐 Real-time Web Search":
    st.subheader("🌐 Real-time Supply Chain News & Disruptions")
    query = st.text_input("What disruptions or news are you tracking?",
                          placeholder="e.g. Red Sea shipping disruptions 2026",
                          key="web_query")
    if st.button("🔎 Search Web", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching the web..."):
                results = web_search(query)
            if results:
                for r in results:
                    with st.expander(f"📌 {r['title']}"):
                        st.write(r["body"])
                        st.markdown(f"[🔗 Open source]({r['href']})")
                with st.spinner("Generating AI summary..."):
                    context = "\n\n".join([f"**{r['title']}**\n{r['body']}" for r in results])
                    summary = call_ai(f"""Query: "{query}"

Latest web results:
{context}

Summarize in clean Markdown:
- **Key Updates** (what's happening)
- **Impact on SMEs** (practical effects)
- **Recommended Actions** (3 concrete steps)""")
                st.subheader("🤖 AI Analysis of Latest News")
                st.markdown(summary)
            else:
                st.info("No results found. Try a different search term.")

# ====================== CHAT WITH MEMORY ======================
elif mode == "💬 Chat with Memory":
    st.subheader("💬 Chat with SupplyChainGPT")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Incoterms, risks, compliance..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build messages list for Groq with full history
        messages = [{"role": "system", "content": "You are SupplyChainGPT, a senior supply chain expert. Answer helpfully, accurately, and concisely. Use Markdown formatting where helpful."}]
        for msg in st.session_state.chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                response_text = f"❌ Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

# ====================== MULTI-DOCUMENT AUDITOR ======================
else:
    st.subheader("📄 Multi-Document Auditor + Comparison")
    st.caption("Upload shipping documents (Bills of Lading, Invoices, Packing Lists) for AI analysis")

    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.info(f"📂 {len(uploaded_files)} file(s) ready. Click below to analyze.")

        if st.button("🔍 Analyze & Compare All Documents", type="primary", use_container_width=True):
            texts = []
            with st.spinner("Reading documents..."):
                for file in uploaded_files:
                    try:
                        reader = PdfReader(file)
                        text = "".join(p.extract_text() or "" for p in reader.pages)
                        texts.append(f"=== DOCUMENT: {file.name} ===\n{text[:4000]}")
                        st.success(f"✅ Read: {file.name} ({len(text)} chars)")
                    except Exception as e:
                        st.warning(f"⚠️ Could not read {file.name}: {e}")

            if not texts:
                st.error("❌ No readable content found. Make sure your PDFs contain text (not just scanned images).")
            else:
                combined = "\n\n".join(texts)
                with st.spinner("AI is analyzing your documents..."):
                    report = call_ai(
                        prompt=f"""Analyze these {len(texts)} shipping documents and provide a detailed audit:

{combined[:12000]}

Output clean Markdown with these exact sections:

## 📊 Document Summary Table
| Document | Type | Key Parties | Incoterm | Value/Qty | Status |
|---|---|---|---|---|---|
(one row per document)

## 🔍 Cross-Document Analysis
Compare consistency across documents (e.g. do consignee names match? Do quantities align? Are Incoterms consistent?)

## ⚠️ Issues & Discrepancies Found
List any mismatches, missing fields, or compliance red flags with severity (🔴 High / 🟡 Medium / 🟢 Low)

## ✅ Overall Risk Rating
Give an overall risk score (Low/Medium/High/Critical) with justification.

## 🛠️ Recommended Fixes
List 3-5 specific actionable corrections needed before shipment.""",
                        system="You are an expert freight forwarder and customs compliance specialist. Analyze shipping documents thoroughly and flag all discrepancies."
                    )

                st.markdown("### 📋 Multi-Document Audit & Comparison Report")
                st.markdown(report)
                st.download_button("📥 Download Full Report", data=report,
                                   file_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                   mime="text/markdown")

st.divider()
st.caption("SupplyChainGPT v4 • Powered by Groq (Free Tier) • Built by Sid Vithal • 2026")
