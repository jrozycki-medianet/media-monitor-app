import streamlit as st
import os
import json
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, grounding

# --- CLOUD SETUP: KEY GENERATION ---
# This block creates the key file from Streamlit Secrets
if "GCP_CREDENTIALS" in st.secrets:
    try:
        # Streamlit reads [GCP_CREDENTIALS] from secrets.toml as a Python dictionary automatically.
        # We don't need json.loads() if we use the table format.
        secrets_obj = st.secrets["GCP_CREDENTIALS"]
        
        # Convert it to a standard dict (in case it's a specific Streamlit object)
        cred_dict = dict(secrets_obj)
        
        # Write to file so Vertex AI can find it
        with open("service_account_key.json", "w") as f:
            json.dump(cred_dict, f)
            
    except Exception as e:
        st.error(f"Secrets Error: {e}")
        st.stop()

# --- CONFIGURATION ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account_key.json"

try:
    with open("service_account_key.json") as f:
        key_data = json.load(f)
        PROJECT_ID = key_data["project_id"]
    vertexai.init(project=PROJECT_ID, location="us-central1")
except Exception as e:
    st.error(f"Authentication Error: {e}")
    st.info("The app cannot read the Google Cloud Key. Please check the Secrets settings.")
    st.stop()

# --- MODEL SETUP ---
try:
    search_tool = Tool(google_search=grounding.GoogleSearch())
except Exception:
    search_tool = Tool.from_dict({"google_search": {}})

model = GenerativeModel("gemini-2.5-pro")

# --- HELPER FUNCTIONS ---
def get_categories_from_ai(business, industry):
    prompt = f"""
    You are a marketing strategist. 
    For a business named '{business}' in the '{industry}' industry, 
    list 5 high-level categories where a customer might encounter this brand during a search.
    Return ONLY a list of categories separated by commas. Do not number them.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        categories = [x.strip() for x in text.split(',')]
        return categories[:5]
    except Exception as e:
        st.error(f"AI Error: {e}")
        return []

def get_prompts_for_category(business, category):
    prompt = f"""
    Act as a potential customer.
    Write 5 specific Google search queries related to the category '{category}' 
    that might lead someone to find the business '{business}'.
    Return ONLY the 5 queries, one per line.
    """
    try:
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        clean_lines = [line.lstrip("1234567890.- ").strip() for line in lines if line.strip()]
        return clean_lines[:5]
    except Exception as e:
        return [f"Error generating prompts: {e}"]

def run_simulation(prompt_text):
    try:
        response = model.generate_content(prompt_text, tools=[search_tool])
        
        # Safe Dictionary Conversion
        try:
            response_dict = response.to_dict()
        except AttributeError:
            return response.text, "Error parsing citations"

        # Answer Extraction
        candidates = response_dict.get('candidates', [])
        if not candidates: return "No response", ""
        parts = candidates[0].get('content', {}).get('parts', [])
        answer = " ".join([p.get('text', '') for p in parts if 'text' in p])

        # Citation Extraction
        citations_list = []
        seen_urls = set()
        grounding_meta = candidates[0].get('grounding_metadata', {})
        
        # Navigation Links
        nav_links = grounding_meta.get('search_entry_point', {}).get('navigation_links', [])
        for link in nav_links:
            url = link.get('url')
            title = link.get('title', 'Web Result')
            if url and url not in seen_urls:
                citations_list.append({'url': url, 'title': title})
                seen_urls.add(url)

        # Grounding Chunks
        chunks = grounding_meta.get('grounding_chunks', [])
        for chunk in chunks:
            web = chunk.get('web', {})
            url = web.get('uri')
            title = web.get('title', 'Web Result')
            if url and url not in seen_urls:
                citations_list.append({'url': url, 'title': title})
                seen_urls.add(url)

        return answer, citations_list
        
    except Exception as e:
        return f"Error: {str(e)}", []

# --- APP INTERFACE ---
st.set_page_config(page_title="LLM Audit", layout="wide")
st.title("🤖 LLM Media Monitoring Pipeline")

if "step" not in st.session_state: st.session_state.step = 1
if "categories" not in st.session_state: st.session_state.categories = []
if "generated_prompts" not in st.session_state: st.session_state.generated_prompts = []
if "results" not in st.session_state: st.session_state.results = []

# Phase 1: Inputs
if st.session_state.step == 1:
    st.markdown("### Step 1: Context Setup")
    with st.form("setup_form"):
        col1, col2 = st.columns(2)
        with col1:
            business = st.text_input("Business Name")
        with col2:
            industry = st.text_input("Industry")
        if st.form_submit_button("Generate Categories") and business and industry:
            st.session_state.business_name = business
            st.session_state.industry = industry
            with st.spinner("Analyzing..."):
                st.session_state.categories = get_categories_from_ai(business, industry)
                st.session_state.step = 2
                st.rerun()

# Phase 2: Categories
if st.session_state.step == 2:
    st.markdown("### Step 2: Confirm Categories")
    edited = st.data_editor([{"Category": c} for c in st.session_state.categories], num_rows="dynamic", use_container_width=True)
    if st.button("Generate Prompts"):
        st.session_state.categories = [r["Category"] for r in edited if r["Category"]]
        st.session_state.generated_prompts = []
        bar = st.progress(0, "Generating...")
        for i, cat in enumerate(st.session_state.categories):
            bar.progress(i/len(st.session_state.categories))
            qs = get_prompts_for_category(st.session_state.business_name, cat)
            for q in qs: st.session_state.generated_prompts.append({"Category": cat, "Prompt": q})
        st.session_state.step = 3
        st.rerun()

# Phase 3: Prompts
if st.session_state.step == 3:
    st.markdown("### Step 3: Review Prompts")
    edited_prompts = st.data_editor(st.session_state.generated_prompts, num_rows="dynamic", use_container_width=True)
    iterations = st.slider("Runs per prompt", 1, 5, 1)
    if st.button("Run Simulation"):
        st.session_state.final_prompts = edited_prompts
        st.session_state.iterations = iterations
        st.session_state.step = 4
        st.rerun()

# Phase 4: Execution
if st.session_state.step == 4:
    st.markdown("### Step 4: Running...")
    data = []
    bar = st.progress(0, "Starting...")
    count = 0
    total = len(st.session_state.final_prompts) * st.session_state.iterations
    
    for item in st.session_state.final_prompts:
        for i in range(st.session_state.iterations):
            count += 1
            bar.progress(count/total)
            ans, cits = run_simulation(item['Prompt'])
            
            # Formatting
            csv_links = []
            html_links = []
            if cits:
                for c in cits:
                    csv_links.append(f'=HYPERLINK("{c["url"]}", "{c["title"]}")')
                    html_links.append(f'<a href="{c["url"]}" target="_blank">{c["title"]}</a>')
                csv_cell = "\n".join(csv_links)
                html_cell = "<br>".join(html_links)
            else:
                csv_cell = "No Citations"
                html_cell = "No Citations"

            data.append({
                "Category": item["Category"],
                "Prompt": item["Prompt"],
                "Iteration": i+1,
                "Answer": ans,
                "Citations": csv_cell,
                "Citations_HTML": html_cell
            })
    st.session_state.results = data
    st.session_state.step = 5
    st.rerun()

# Phase 5: Results
if st.session_state.step == 5:
    st.success("Complete!")
    df = pd.DataFrame(st.session_state.results)
    
    # HTML Table for display
    st.write(df[["Category", "Prompt", "Answer", "Citations_HTML"]].rename(columns={"Citations_HTML": "Citations"}).to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # CSV Download
    csv = df[["Category", "Prompt", "Answer", "Citations"]].to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "results.csv", "text/csv")
    
    if st.button("Restart"):
        st.session_state.clear()
        st.rerun()