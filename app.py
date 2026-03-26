import streamlit as st
import google.generativeai as genai
from datetime import datetime
from io import StringIO
import csv

st.set_page_config(page_title="SupplyChainGPT", page_icon="📦", layout="wide")

# ====================== KNOWLEDGE BASE (EASY TO MAINTAIN) ======================
# Add or edit entries here. No coding needed beyond copy-paste.
knowledge_base = [
    {
        "title": "Incoterms 2020 Overview",
        "category": "Incoterms",
        "content": "Incoterms 2020 are 11 rules by the International Chamber of Commerce that define buyer/seller responsibilities for delivery, risk transfer, costs, and insurance in international trade. Major updates from 2010: DAT became DPU, security obligations clarified."
    },
    {
        "title": "EXW - Ex Works",
        "category": "Incoterms",
        "content": "Seller makes goods available at their premises. Buyer handles loading, transport, insurance, export/import clearance, and all risks from seller's door. Lowest seller responsibility — best when buyer has strong logistics."
    },
    {
        "title": "FOB - Free on Board",
        "category": "Incoterms",
        "content": "Seller delivers goods on board the vessel at the named port. Risk transfers when goods are on board. Seller handles export clearance. Very common for sea freight."
    },
    {
        "title": "DDP - Delivered Duty Paid",
        "category": "Incoterms",
        "content": "Seller bears all costs and risks (including import duties, taxes, and customs clearance) until goods are delivered to buyer’s premises. Highest seller responsibility."
    },
    {
        "title": "CIF - Cost, Insurance and Freight",
        "category": "Incoterms",
        "content": "Seller pays for transport and insurance to the named port. Risk transfers when goods are on board the vessel. Buyer handles import clearance."
    },
    {
        "title": "US Customs & CMMC Compliance",
        "category": "Compliance",
        "content": "US imports require accurate HTS codes, valuation, country of origin, and ISF filing. Defense suppliers must meet CMMC 2.0 cybersecurity. Penalties for errors can exceed $50k+ per shipment plus delays."
    },
    {
        "title": "EU Import Regulations",
        "category": "Compliance",
        "content": "EU requires CBAM reporting for carbon-intensive goods, proper TARIC codes, and proof of origin. Incoterms must align with Union Customs Code. Multilingual documents must be legally consistent."
    },
    {
        "title": "Common Bill of Lading Errors",
        "category": "Compliance",
        "content": "Frequent issues: mismatched Incoterms, missing seals, incorrect consignee, wrong HS codes, or inconsistent descriptions. These cause 30-60 day delays and extra fees."
    },
    {
        "title": "Supply Chain Risk Management",
        "category": "Risk Management",
        "content": "Key risks include supplier bankruptcy, geopolitical events, port congestion, currency fluctuation, and cybersecurity. Best practice: maintain dual sourcing and monitor real-time visibility tools."
    },
    {
        "title": "Geopolitical Disruptions 2026",
        "category": "Risk Management",
        "content": "Red Sea/Suez issues, US-China tensions, and nearshoring trends are shifting routes. Many SMEs are moving 20-30% of production to Mexico/Vietnam to reduce risk."
    },
    {
        "title": "Sustainability & ESG in Supply Chains",
        "category": "Sustainability",
        "content": "Buyers now demand Scope 3 carbon reporting. EU CSRD and US SEC climate rules are coming. Using DPU/DDP with green carriers can improve ESG scores and win contracts."
    },
    {
        "title": "Digital Tools & Traceability",
        "category": "Logistics",
        "content": "Blockchain, IoT sensors, and AI forecasting are now affordable for SMEs. They reduce documentation errors by 70% and enable real-time compliance checks."
    },
]

# ====================== HELPER FUNCTIONS ======================
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

@st.cache_data(show_spinner=False)
def get_embeddings():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    embeddings = []
    for item in knowledge_base:
        result = genai.embed_content(
            model="models/embedding-001",
            content=f"{item['title']}. {item['content']}",
            task_type="retrieval_document"
        )
        embeddings.append(result["embedding"])
    return embeddings

def semantic_search(query, selected_categories, min_relevance, top_k=6):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    embeddings = get_embeddings()
    
    # Filter by category first
    if selected_categories and "All" not in selected_categories:
        filtered_indices = [i for i, item in enumerate(knowledge_base) if item["category"] in selected_categories]
    else:
        filtered_indices = list(range(len(knowledge_base)))
    
    filtered_kb = [knowledge_base[i] for i in filtered_indices]
    filtered_emb = [embeddings[i] for i in filtered_indices]
    
    # Embed query
    q_result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    q_emb = q_result["embedding"]
    
    # Score and rank
    scored = []
    for i, emb in enumerate(filtered_emb):
        score = cosine_similarity(q_emb, emb)
        scored.append((i, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in scored[:top_k]:
        if score * 100 >= min_relevance:
            item = filtered_kb[idx].copy()
            item["relevance"] = round(score * 100, 1)
            results.append(item)
    return results

def generate_ai_insights(query, results):
    if not results:
        return "No sufficiently relevant information found. Try a broader query or lower the relevance filter."
    
    context = "\n\n---\n\n".join([
        f"**{r['title']}** ({r['category']}, {r['relevance']}% relevant)\n{r['content']}"
        for r in results
    ])
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""You are a senior supply chain strategist.
User query: "{query}"

Relevant knowledge:
{context}

Create a professional AI analysis in clean Markdown:
- **Summary** (2-3 sentences)
- **Key Insights**
- **Potential Risks** (with severity)
- **Trends & Opportunities**
- **Recommended Actions** (3-5 practical steps for SMEs)

Be concise, actionable, and focused on real-world impact."""
    
    response = model.generate_content(prompt)
    return response.text

# ====================== MAIN APP ======================
st.title("🚢 SupplyChainGPT")
st.markdown("**Smart semantic search + Incoterms & Compliance Auditor** — Built for SMEs in 2026")

# Sidebar navigation
with st.sidebar:
    st.header("📦 Navigation")
    mode = st.radio("Choose tool:", ["🔍 Smart Search", "📄 Document Auditor"], label_visibility="collapsed")
    
    st.divider()
    
    advanced_mode = st.checkbox("Advanced Mode (Semantic Search + AI Insights)", value=True, help="Turn off for faster basic answers")
    
    if mode == "🔍 Smart Search":
        st.subheader("🔎 Filters")
        all_categories = ["All"] + sorted(set(item["category"] for item in knowledge_base))
        selected_categories = st.multiselect("Categories", all_categories, default=["All"])
        min_relevance = st.slider("Minimum Relevance %", 0, 100, 60)
        
        st.subheader("📜 Search History")
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        
        if st.session_state.search_history:
            for i, entry in enumerate(reversed(st.session_state.search_history[-8:])):
                if st.button(f"🔄 {entry['query'][:45]}...", key=f"hist_{i}"):
                    st.session_state.search_query = entry["query"]
                    st.session_state.trigger_search = True
                    st.rerun()
            if st.button("🗑️ Clear History"):
                st.session_state.search_history = []
                st.rerun()
        else:
            st.caption("No searches yet")

# ====================== SMART SEARCH TAB ======================
if mode == "🔍 Smart Search":
    st.subheader("🔍 Ask anything about supply chains, Incoterms, compliance, or logistics")
    
    # Suggested queries
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
    
    # Search input
    query = st.text_input("Your supply chain question:", placeholder="e.g. What are the risks of using EXW for international shipments?", key="search_query")
    
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True) or st.session_state.get("trigger_search", False)
    
    if search_clicked:
        if "trigger_search" in st.session_state:
            del st.session_state.trigger_search
        
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching knowledge base..."):
                results = semantic_search(query, selected_categories, min_relevance)
            
            # Save to history
            st.session_state.search_history.append({"query": query})
            if len(st.session_state.search_history) > 10:
                st.session_state.search_history.pop(0)
            
            if results:
                st.success(f"Found {len(results)} relevant results")
                
                # Display results as cards
                st.subheader("📋 Search Results")
                for r in results:
                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{r['title']}**")
                            st.caption(f"{r['category']} • {r['relevance']}% relevant")
                        with col2:
                            st.caption(f"⭐ {r['relevance']}")
                        st.markdown(r["content"])
                        st.markdown("---")
                
                # AI Analysis Layer
                if advanced_mode:
                    with st.spinner("Generating AI insights..."):
                        insights = generate_ai_insights(query, results)
                    st.subheader("🤖 AI-Powered Insights & Analysis")
                    st.markdown(insights)
                
                # Export
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(["Title", "Category", "Relevance (%)", "Content"])
                for r in results:
                    writer.writerow([r["title"], r["category"], r["relevance"], r["content"]])
                csv_data = output.getvalue()
                st.download_button(
                    label="📥 Export Results as CSV",
                    data=csv_data,
                    file_name=f"supplychain_search_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No results above the relevance threshold. Try lowering the filter or asking a broader question.")

# ====================== DOCUMENT AUDITOR TAB (improved) ======================
else:
    st.subheader("📄 Bill of Lading & Shipping Document Auditor")
    st.markdown("Upload any PDF or paste text to get instant Incoterms, multilingual, and regulatory checks.")
    
    uploaded_file = st.file_uploader("Upload Bill of Lading or shipping document (PDF)", type="pdf")
    text_input = st.text_area("OR paste text here", height=250)
    
    if uploaded_file or text_input:
        if uploaded_file:
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            text = "".join(page.extract_text() + "\n" for page in reader.pages)
        else:
            text = text_input
        
        if st.button("🔍 Analyze with AI (Gemini)", type="primary", use_container_width=True):
            with st.spinner("Analyzing document... (usually ~10 seconds)"):
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                prompt = f"""You are a world-class supply chain compliance auditor.
Document text:
{text[:15000]}

Perform these exact tasks and output ONLY in clean Markdown:
1. Extract every Incoterm mentioned.
2. Check consistency (responsibilities, transport mode, risk transfer, cost allocation). Flag contradictions.
3. If not in English, translate key sections and note any legal meaning changes.
4. Flag regulatory risks (US/EU customs, sanctions, CMMC implications).
5. Give severity rating (Low/Medium/High) + exact suggested fixes.

Use tables where helpful. Be extremely precise."""

                response = model.generate_content(prompt)
                
                st.markdown("### 📋 AI Audit Report")
                st.markdown(response.text)
                
                # Download button
                st.download_button(
                    label="📥 Download Full Report (Markdown)",
                    data=response.text,
                    file_name="supplychain_audit_report.md",
                    mime="text/markdown"
                )
                st.success("✅ Analysis complete!")

st.divider()
st.caption("SupplyChainGPT • Open-source • Built by Sid Vithal • 2026 • Semantic search + AI auditor")
