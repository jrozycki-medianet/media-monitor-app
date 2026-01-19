import streamlit as st
import os
import json
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, grounding

# --- CLOUD SETUP: KEY GENERATION (NO PASSWORD) ---
# We force-write the key file every time the app loads to fix any corruption.
if "GCP_CREDENTIALS" in st.secrets:
    try:
        secret_data = st.secrets["GCP_CREDENTIALS"]
        
        # Determine if it's a JSON string or a Dictionary
        if isinstance(secret_data, str):
            cred_dict = json.loads(secret_data)
        elif isinstance(secret_data, dict):
            cred_dict = dict(secret_data)
        else:
            st.error("GCP_CREDENTIALS secret format is incorrect.")
            st.stop()
            
        # Overwrite the file on the server
        with open("service_account_key.json", "w") as f:
            json.dump(cred_dict, f)
            
    except Exception as e:
        st.error(f"Error reading secrets: {e}. Please check your Streamlit Secrets.")
        st.stop()
else:
    # If running locally without secrets.toml, we hope the file exists
    if not os.path.exists("service_account_key.json"):
        st.warning("Running in Cloud Mode but no 'GCP_CREDENTIALS' secret found.")

# --- CONFIGURATION ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account_key.json"

try:
    with open("service_account_key.json") as f:
        key_data = json.load(f)
        PROJECT_ID = key_data["project_id"]
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location="us-central1")
except Exception as e:
    # If this fails, it means the key file is still bad (e.g. empty)
    st.error(f"Authentication Failed: {e}")
    st.info("The app cannot read the Google Cloud Key. Please check the Secrets settings.")
    st.stop()

# --- DEFINE THE MODEL & SEARCH TOOL ---
try:
    # Modern Approach
    search_tool = Tool(google_search=grounding.GoogleSearch())
except Exception as e:
    # Fallback
    search_tool = Tool.from_dict({"google_search": {}})

# Load the Model
model = GenerativeModel("gemini-2.5-pro")

# --- HELPER FUNCTIONS ---
def get_categories_from_ai(business, industry):
    """Asks Gemini for 5 categories."""
    prompt = f"""
    You are a marketing strategist. 
    For a business named '{business}' in the '{industry}' industry, 
    list 5 high-level categories where a customer might encounter this brand during a search.
    Return ONLY a list of categories separated by commas. Do not number them. The categories should not include the actual '{business}' name.
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
    """Asks Gemini for search queries."""
    prompt = f"""
    Act as a potential customer.
    Write 5 specific Google search queries related to the category '{category}' 
    that might lead someone to find the business '{business}'.
    Return ONLY the 5 queries, one per line.
    The prompts should never contain the actual '{business}' name.
    """
    try:
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        clean_lines = [line.lstrip("1234567890.- ").strip() for line in lines if line.strip()]
        return clean_lines[:5]
    except Exception as e:
        return [f"Error generating prompts: {e}"]

def run_simulation(prompt_text):
    """
    Runs a single prompt with Search Grounding enabled.
    INTEGRATED FIX: Extracts Title + URL for named hyperlinks.
    """
    try:
        # We pass the 'tools' parameter here to enable Search
        response = model.generate_content(
            prompt_text, 
            tools=[search_tool]
        )
        
        # 1. Convert to Dictionary (Safer extraction method)
        try:
            response_dict = response.to_dict()
        except AttributeError:
            return response.text, "Error parsing citations"

        # 2. Extract Answer Text
        candidates = response_dict.get('candidates', [])
        if not candidates:
            return "No response from AI", ""
            
        parts = candidates[0].get('content', {}).get('parts', [])
        answer = " ".join([p.get('text', '') for p in parts if 'text' in p])

        # 3. Extract Citations (Title + URL)
        citations_list = []
        seen_urls = set()
        
        grounding_meta = candidates[0].get('grounding_metadata', {})
        
        # Method A: Check Search Entry Point (Navigation Links)
        search_entry = grounding_meta.get('search_entry_point', {})
        nav_links = search_entry.get('navigation_links', [])
        
        for link in nav_links:
            url = link.get('url')
            title = link.get('title', 'Web Result') 
            
            if url and url not in seen_urls:
                citations_list.append(f"{title} ({url})")
                seen_urls.add(url)

        # Method B: Check Grounding Chunks (Web Snippets)
        chunks = grounding_meta.get('grounding_chunks', [])
        for chunk in chunks:
            web = chunk.get('web', {})
            url = web.get('uri')
            title = web.get('title', 'Web Result')
            
            if url and url not in seen_urls:
                citations_list.append(f"{title} ({url})")
                seen_urls.add(url)
                
        if citations_list:
            url_text = "\n".join(citations_list)
        else:
            url_text = "No citations provided"
            
        return answer, url_text
        
    except Exception as e:
        return f"Error: {str(e)}", ""

# --- PAGE SETUP ---
st.set_page_config(page_title="LLM Audit", layout="wide")
st.title("🤖 LLM Media Monitoring Pipeline")

# --- SESSION STATE ---
if "step" not in st.session_state: st.session_state.step = 1
if "categories" not in st.session_state: st.session_state.categories = []
if "generated_prompts" not in st.session_state: st.session_state.generated_prompts = []
if "results" not in st.session_state: st.session_state.results = []

# --- PHASE 1: INPUTS ---
if st.session_state.step == 1:
    st.markdown("### Step 1: Context Setup")
    with st.form("setup_form"):
        col1, col2 = st.columns(2)
        with col1:
            business = st.text_input("Business Name", placeholder="Enter business name...")
        with col2:
            industry = st.text_input("Industry", placeholder="Enter industry...")
        submit = st.form_submit_button("Generate Categories")
        
    if submit:
        if business and industry:
            st.session_state.business_name = business
            st.session_state.industry = industry
            with st.spinner("Consulting Gemini Strategist..."):
                cats = get_categories_from_ai(business, industry)
                if cats:
                    st.session_state.categories = cats
                    st.session_state.step = 2
                    st.rerun()

# --- PHASE 2: CONFIRM CATEGORIES ---
if st.session_state.step == 2:
    st.markdown(f"### Step 2: Confirm Categories for **{st.session_state.business_name}**")
    edited_categories = st.data_editor([{"Category": c} for c in st.session_state.categories], num_rows="dynamic", use_container_width=True)

    if st.button("Confirm Categories & Generate Prompts"):
        final_cats = [row["Category"] for row in edited_categories if row["Category"]]
        st.session_state.categories = final_cats
        
        all_prompts = []
        progress_bar = st.progress(0, text="Generating prompts...")
        for i, cat in enumerate(final_cats):
            progress_bar.progress((i / len(final_cats)), text=f"Writing prompts for: {cat}...")
            queries = get_prompts_for_category(st.session_state.business_name, cat)
            for q in queries:
                all_prompts.append({"Category": cat, "Prompt": q})
        
        progress_bar.empty()
        st.session_state.generated_prompts = all_prompts
        st.session_state.step = 3
        st.rerun()

# --- PHASE 3: EDIT PROMPTS ---
if st.session_state.step == 3:
    st.markdown("### Step 3: Review Search Prompts")
    edited_prompts = st.data_editor(st.session_state.generated_prompts, num_rows="dynamic", use_container_width=True, height=400)
    
    st.divider()
    st.markdown("### Simulation Settings")
    iterations = st.slider("How many times to run each prompt?", min_value=1, max_value=10, value=2)
    st.warning(f"Total API Calls: {len(edited_prompts)} prompts × {iterations} runs = **{len(edited_prompts) * iterations} calls**")

    if st.button("Confirm Prompts & Run Simulation"):
        st.session_state.final_prompts = edited_prompts
        st.session_state.iterations = iterations
        st.session_state.step = 4
        st.rerun()

# --- PHASE 4: EXECUTION ---
if st.session_state.step == 4:
    st.markdown("### Step 4: Running Simulation")
    
    prompts = st.session_state.final_prompts
    runs = st.session_state.iterations
    total_ops = len(prompts) * runs
    
    my_bar = st.progress(0, text="Starting simulation...")
    status_text = st.empty()
    
    results_data = []
    
    counter = 0
    for p_idx, item in enumerate(prompts):
        prompt_text = item["Prompt"]
        category = item["Category"]
        
        for i in range(runs):
            counter += 1
            status_text.text(f"Processing ({counter}/{total_ops}): {prompt_text}")
            my_bar.progress(counter / total_ops)
            
            # Run the AI
            answer, citations = run_simulation(prompt_text)
            
            results_data.append({
                "Business": st.session_state.business_name,
                "Category": category,
                "Prompt": prompt_text,
                "Iteration": i + 1,
                "Answer": answer,
                "Citations": citations
            })
            
    st.session_state.results = results_data
    st.session_state.step = 5
    st.rerun()

# --- PHASE 5: OUTPUT & ANALYSIS (UPDATED) ---
if st.session_state.step == 5:
    st.success("Simulation Complete!")
    
    # Create the DataFrame once
    df = pd.DataFrame(st.session_state.results)

    # --- NEW: MENTIONS CALCULATOR ---
    st.markdown("---")
    st.header("📊 Mention Analysis")
    
    col_input, col_table = st.columns([1, 2])
    
    with col_input:
        st.markdown(f"**Primary Business:** {st.session_state.business_name}")
        extra_keywords = st.text_area(
            "Add Competitors / Keywords (comma separated):", 
            placeholder="e.g. Nike, Adidas, price, quality"
        )
    
    # Logic to calculate mentions
    # 1. Start with the main business name
    keywords_to_track = [st.session_state.business_name]
    
    # 2. Add user keywords if provided
    if extra_keywords:
        # Split by comma and remove whitespace
        cleaned_extras = [k.strip() for k in extra_keywords.split(',') if k.strip()]
        keywords_to_track.extend(cleaned_extras)
        
    # 3. Perform Counts
    analysis_stats = []
    for kw in keywords_to_track:
        kw_lower = kw.lower()
        # Count occurences (case insensitive)
        count_ans = df["Answer"].fillna("").astype(str).str.lower().str.count(kw_lower).sum()
        count_cit = df["Citations"].fillna("").astype(str).str.lower().str.count(kw_lower).sum()
        
        analysis_stats.append({
            "Keyword": kw,
            "Mentions in Answers": count_ans,
            "Mentions in Citations": count_cit,
            "Total Volume": count_ans + count_cit
        })
        
    # 4. Display Analysis Table
    with col_table:
        st.dataframe(
            pd.DataFrame(analysis_stats), 
            use_container_width=True, 
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### Raw Data Results")
    
    # Display Main Data
    st.dataframe(df)
    
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='llm_audit_results.csv',
        mime='text/csv',
    )
    
    if st.button("Start Over"):
        st.session_state.clear()
        st.rerun()