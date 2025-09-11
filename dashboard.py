# ==============================================================================
# Step 1: All Imports from your Notebook
# ==============================================================================
import streamlit as st
import requests
import pandas as pd
import easyocr
import cv2
import numpy as np
import re
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict

# ==============================================================================
# Step 2: All Constants and Setup from your Notebook
# ==============================================================================

# API URL to fetch product data
API_URL = "https://flipkart-backend-8z7t.onrender.com/api/v1/product/all-products"

# Initialize the OCR reader once
# NOTE: This might take a moment the first time it runs
try:
    reader = easyocr.Reader(["en"], gpu=False)
except Exception as e:
    st.error(f"Failed to initialize EasyOCR. Please check your installation. Error: {e}")
    st.stop()


# Layer 1 and Layer 2 compliance rules dictionaries
layer1 = {
    "Manufacturer/Importer Name & Address": False, "MRP": False, "Net Quantity": False,
    "Date of Manufacture/Expiry": False, "Country of Origin": False, "Consumer Care Details": False
}
layer2 = {
    "Grocery": {
        "FSSAI Number": False, "Expiry/Best Before Date": False, "Batch/Lot Number": False,
        "Ingredients List": False, "Allergen Info": False, "Storage Instructions": False,
        "Nutritional Info": False, "ISO/BRC/Organic Certification": False
    },
    # Add other categories from your notebook here if needed...
}


# ==============================================================================
# Step 3: All Classes and Functions from your Notebook
# (Copied directly from SIH.ipynb)
# ==============================================================================

@dataclass
class ComplianceRule:
    flag_name: str
    required_keywords: List[str] = None
    optional_keywords: List[str] = None
    forbidden_keywords: List[str] = None
    regex_patterns: List[str] = None
    min_length: int = None
    case_sensitive: bool = False

class ThreatLevelAnalyzer:
    def __init__(self):
        self.threat_levels = self._setup_threat_levels()

    def _setup_threat_levels(self):
        layer1_threats = {
            "Manufacturer/Importer Name & Address": "CRITICAL", "MRP": "HIGH", "Net Quantity": "HIGH",
            "Date of Manufacture/Expiry": "CRITICAL", "Country of Origin": "MEDIUM", "Consumer Care Details": "MEDIUM"
        }
        layer2_threats = {
            "Grocery": {
                "FSSAI Number": "CRITICAL", "Expiry/Best Before Date": "CRITICAL", "Batch/Lot Number": "HIGH",
                "Ingredients List": "HIGH", "Allergen Info": "CRITICAL", "Storage Instructions": "MEDIUM",
                "Nutritional Info": "MEDIUM", "ISO/BRC/Organic Certification": "LOW"
            }
        }
        return {"layer1": layer1_threats, "layer2": layer2_threats}

    def get_threat_level(self, flag_name: str, category: str) -> str:
        if flag_name in self.threat_levels["layer1"]:
            return self.threat_levels["layer1"][flag_name]
        if category in self.threat_levels["layer2"] and flag_name in self.threat_levels["layer2"][category]:
            return self.threat_levels["layer2"][category][flag_name]
        return "MEDIUM"

    def calculate_threat_score(self, missing_flags: Dict[str, str]) -> Dict:
        threat_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        threat_weights = {"CRITICAL": 10, "HIGH": 7, "MEDIUM": 4, "LOW": 1}
        for flag, threat_level in missing_flags.items():
            threat_counts[threat_level] += 1
        total_score = sum(threat_counts[level] * threat_weights[level] for level in threat_counts)
        max_possible_score = len(missing_flags) * threat_weights["CRITICAL"] if missing_flags else 1
        threat_percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        if threat_percentage >= 80 or threat_counts["CRITICAL"] > 0: overall_threat = "CRITICAL"
        elif threat_percentage >= 60 or threat_counts["HIGH"] > 2: overall_threat = "HIGH"
        elif threat_percentage >= 40 or threat_counts["MEDIUM"] > 3: overall_threat = "MEDIUM"
        else: overall_threat = "LOW"
        return {
            "overall_threat_level": overall_threat, "threat_score": total_score,
            "threat_percentage": round(threat_percentage, 1), "threat_breakdown": threat_counts,
            "total_missing_flags": len(missing_flags)
        }

class LocalComplianceModel:
    def __init__(self):
        self.rules = {}
        self.category_rules = defaultdict(dict)
        self._setup_rules()
    def _setup_rules(self):
        self.rules = {
            "Manufacturer/Importer Name & Address": ComplianceRule(flag_name="Manufacturer/Importer Name & Address", optional_keywords=["manufactured", "mfd", "importer", "marketed by", "address"], regex_patterns=[r'manufactured.*by', r'mfd.*by', r'marketed.*by', r'\d{6}'], min_length=10),
            "MRP": ComplianceRule(flag_name="MRP", optional_keywords=["mrp", "price", "rs"], regex_patterns=[r'mrp.*₹?\d+', r'₹\d+', r'rs\.?\s*\d+']),
            "Net Quantity": ComplianceRule(flag_name="Net Quantity", optional_keywords=["net", "quantity", "weight"], regex_patterns=[r'net.*[gkml]', r'\d+\s*[gkml]']),
            "Date of Manufacture/Expiry": ComplianceRule(flag_name="Date of Manufacture/Expiry", optional_keywords=["expiry", "exp", "best before", "mfd"], regex_patterns=[r'exp.*\d{1,2}[/-]\d{2,4}', r'best.*before']),
            "Country of Origin": ComplianceRule(flag_name="Country of Origin", optional_keywords=["made in", "country of origin"], regex_patterns=[r'made.*in.*\w+']),
            "Consumer Care Details": ComplianceRule(flag_name="Consumer Care Details", optional_keywords=["customer care", "helpline", "contact"], regex_patterns=[r'\d{10}', r'customer.*care'])
        }
        self.category_rules["Grocery"] = {
            "FSSAI Number": ComplianceRule(flag_name="FSSAI Number", optional_keywords=["fssai", "license"], regex_patterns=[r'fssai.*\d{14}', r'lic.*no.*\d+']),
            "Expiry/Best Before Date": ComplianceRule(flag_name="Expiry/Best Before Date", optional_keywords=["expiry", "best before", "exp"], regex_patterns=[r'exp.*\d{1,2}[/-]\d{2,4}']),
            "Ingredients List": ComplianceRule(flag_name="Ingredients List", optional_keywords=["ingredients"], regex_patterns=[r'ingredients:']),
            "Nutritional Info": ComplianceRule(flag_name="Nutritional Info", optional_keywords=["nutrition", "energy", "protein"], regex_patterns=[r'nutrition.*facts', r'per.*100g'])
        }
    def _check_text_against_rule(self, text: str, rule: ComplianceRule) -> bool:
        if not text: return False
        text_lower = text.lower()
        if rule.min_length and len(text.strip()) < rule.min_length: return False
        if rule.optional_keywords and not any(k.lower() in text_lower for k in rule.optional_keywords): return False
        if rule.regex_patterns and not any(re.search(p, text_lower, re.IGNORECASE) for p in rule.regex_patterns): return False
        return True
    def check_compliance(self, ocr_text: str, api_text: str, category: str, applicable_flags: List[str]) -> Dict[str, int]:
        combined_text = f"{ocr_text} {api_text}".strip()
        results = {}
        for flag in applicable_flags:
            rule = self.rules.get(flag) or self.category_rules.get(category, {}).get(flag)
            if rule:
                results[flag] = 1 if self._check_text_against_rule(combined_text, rule) else 0
            else:
                results[flag] = 0
        return results

def fetch_products():
    try:
        r = requests.get(API_URL)
        r.raise_for_status()
        return r.json().get("products", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return []

def run_ocr(url):
    try:
        resp = requests.get(url, timeout=10)
        img_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=0, paragraph=True)
        return " ".join(results) if results else "No text detected"
    except Exception as e:
        return f"OCR Error: {str(e)}"

def extract_api_text(product):
    desc = product.get("description", "")
    brand = product.get("brand", {}).get("name", "")
    highlights = " | ".join(product.get("highlights", []))
    mrp = "mrp: " + str(product.get("price", ""))
    specs_list = product.get("specifications", [])
    specs_texts = [f"{s.get('title', '')}: {s.get('description', '')}" for s in specs_list]
    specs_text = " | ".join(specs_texts)
    combined_text = f"Description: {desc} | Brand: {brand} | Highlights: {highlights} | Specs: {specs_text}"
    return {"description": desc, "brand": brand, "highlights": highlights, "specifications": specs_text, "price":mrp, "api_text": combined_text}

def add_threat_analysis(df_final_report):
    threat_analyzer = ThreatLevelAnalyzer()
    threat_data = []
    for idx, row in df_final_report.iterrows():
        category = row['category']
        flags = row['flags']
        missing_flags = {flag: threat_analyzer.get_threat_level(flag, category) for flag, value in flags.items() if value == 0}
        if missing_flags:
            threat_analysis = threat_analyzer.calculate_threat_score(missing_flags)
        else:
            threat_analysis = {"overall_threat_level": "NONE", "threat_score": 0, "threat_percentage": 0, "threat_breakdown": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}, "total_missing_flags": 0}
        threat_data.append({'missing_flags': missing_flags, **threat_analysis})
    threat_df = pd.DataFrame(threat_data)
    for col in threat_df.columns:
        if col not in ['missing_flags', 'threat_breakdown']:
             df_final_report[col] = threat_df[col]
    df_final_report['critical_missing'] = threat_df['threat_breakdown'].apply(lambda x: x['CRITICAL'])
    df_final_report['high_missing'] = threat_df['threat_breakdown'].apply(lambda x: x['HIGH'])
    df_final_report['medium_missing'] = threat_df['threat_breakdown'].apply(lambda x: x['MEDIUM'])
    df_final_report['low_missing'] = threat_df['threat_breakdown'].apply(lambda x: x['LOW'])
    df_final_report['missing_flags_display'] = threat_df['missing_flags'].apply(lambda x: list(x.keys()))
    return df_final_report

def merge_product_compliance(df_raw_report):
    merged_products = []
    for product_id, group in df_raw_report.groupby('product_id'):
        first_row = group.iloc[0]
        merged_flags = {}
        all_flags = set().union(*(row['flags'].keys() for _, row in group.iterrows()))
        for flag in all_flags:
            merged_flags[flag] = 1 if any(row['flags'].get(flag, 0) for _, row in group.iterrows()) else 0
        merged_products.append({'product_id': product_id, 'category': first_row['category'], 'product_name': first_row['product_name'], 'image_urls': list(group['image_url'].unique()), 'num_images': len(group['image_url'].unique()), 'flags': merged_flags})
    return pd.DataFrame(merged_products)

def run_complete_compliance_check_with_threats(df_final, layer1, layer2):
    model = LocalComplianceModel()
    raw_report = []
    progress_bar = st.progress(0, text="Analyzing products...")
    total_rows = len(df_final)

    for idx, row in df_final.iterrows():
        applicable_flags = list(layer1.keys()) + list(layer2.get(row["category"], {}).keys())
        full_api_text = " ".join([str(row.get(c, "")) for c in ["api_text", "description", "specifications", "price"]])
        flags_result = model.check_compliance(str(row.get("ocr_text", "")), full_api_text, row["category"], applicable_flags)
        raw_report.append({"product_id": row["product_id"], "category": row["category"], "product_name": row["product_name"], "image_url": row["image_url"], "flags": flags_result})
        progress_bar.progress((idx + 1) / total_rows, text=f"Analyzing: {row['product_name']}")

    progress_bar.empty()
    df_raw_report = pd.DataFrame(raw_report)
    df_merged = merge_product_compliance(df_raw_report)
    df_final_report = add_threat_analysis(df_merged)
    return df_final_report

# ==============================================================================
# Step 4: Streamlit UI Code
# ==============================================================================
st.set_page_config(page_title="Compliance Checker Dashboard", layout="wide")
st.title(" E-Commerce Compliance Dashboard")
st.markdown("An automated tool to scan and validate e-commerce listings against Legal Metrology requirements.")

@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_and_process_data():
    products_api = fetch_products()
    if not products_api:
        st.warning("Could not fetch any products from the API.")
        return pd.DataFrame()

    ocr_results, api_results = [], []
    for p in products_api:
        api_results.append({"product_id": p.get("_id"), "category": p.get("category"), "product_name": p.get("name"), **extract_api_text(p)})
        if "images" in p and p["images"]:
            for img in p["images"]:
                ocr_results.append({"product_id": p.get("_id"), "category": p.get("category"), "product_name": p.get("name"), "image_url": img.get("url"), "ocr_text": "Processing..."}) # Placeholder

    df_api = pd.DataFrame(api_results)
    df_ocr = pd.DataFrame(ocr_results)

    # Run OCR on unique image URLs to avoid re-processing
    unique_urls = df_ocr['image_url'].dropna().unique()
    ocr_text_map = {url: run_ocr(url) for url in unique_urls}
    df_ocr['ocr_text'] = df_ocr['image_url'].map(ocr_text_map)

    df_merged = pd.merge(df_ocr, df_api, on=["product_id", "category", "product_name"], how="left")
    
    # Run final analysis
    final_report = run_complete_compliance_check_with_threats(df_merged, layer1, layer2)
    return final_report

# --- Main Dashboard ---
df_report = load_and_process_data()

if not df_report.empty:
    st.header("Dashboard Overview")
    total_products = df_report.shape[0]
    critical_threats = (df_report['overall_threat_level'] == 'CRITICAL').sum()
    high_threats = (df_report['overall_threat_level'] == 'HIGH').sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products Analyzed", total_products)
    col2.metric("Products with CRITICAL Threats", critical_threats)
    col3.metric("Products with HIGH Threats", high_threats)

    st.subheader("Threat Level Distribution")
    threat_counts = df_report['overall_threat_level'].value_counts()
    st.bar_chart(threat_counts)

    st.header("Detailed Compliance Report")
    st.sidebar.header("Filter Options")
    
    # Filters
    categories = df_report['category'].unique()
    selected_category = st.sidebar.multiselect('Filter by Category:', options=categories, default=categories)
    
    threat_levels = df_report['overall_threat_level'].unique()
    selected_threat = st.sidebar.multiselect('Filter by Threat Level:', options=threat_levels, default=threat_levels)

    df_filtered = df_report[df_report['category'].isin(selected_category) & df_report['overall_threat_level'].isin(selected_threat)]
    
    # Display columns
    display_cols = ['product_name', 'category', 'overall_threat_level', 'threat_score', 'total_missing_flags', 'missing_flags_display']
    st.dataframe(df_filtered[display_cols].rename(columns={'missing_flags_display': 'Missing Flags'}), use_container_width=True)

else:
    st.info("Awaiting data to generate the report.")