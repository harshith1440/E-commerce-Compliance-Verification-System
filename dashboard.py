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

API_URL = "https://flipkart-backend-8z7t.onrender.com/api/v1/product/all-products"

@st.cache_resource
def get_ocr_reader():
    """Initializes and returns the EasyOCR reader, cached for performance."""
    try:
        return easyocr.Reader(["en"], gpu=False)
    except Exception as e:
        st.error(f"Fatal Error: Failed to initialize EasyOCR. The app cannot continue. Error: {e}")
        st.stop()

reader = get_ocr_reader()

# Layer 1 (common for all categories)
layer1 = {
    "Manufacturer/Importer Name & Address": False, "MRP": False, "Net Quantity": False,
    "Date of Manufacture/Expiry": False, "Country of Origin": False, "Consumer Care Details": False
}

# Layer 2 (COMPLETE category specific rules)
layer2 = {
    "Mobiles": {"SAR Value": False, "Battery Capacity": False, "Charger/Adapter Info": False, "BIS Certification": False, "Warranty Period": False, "Customer Care Helpline": False, "Importer/Distributor Details": False},
    "Electronics": {"BIS Certification": False, "Warranty Period": False, "Power Consumption/Voltage": False, "Customer Care Helpline": False, "Importer/Distributor Details": False},
    "Grocery": {"FSSAI Number": False, "Expiry/Best Before Date": False, "Batch/Lot Number": False, "Ingredients List": False, "Allergen Info": False, "Storage Instructions": False, "Nutritional Info": False, "ISO/BRC/Organic Certification": False},
    "Furniture": {"Material Type": False, "Finish/Polish Details": False, "Load Capacity": False, "Fire Resistance/IS Certification": False, "Assembly Instructions": False, "Warranty/Guarantee": False},
    "Lifestyle": {"Size/Measurement": False, "Fabric/Material Composition": False, "Care Instructions": False, "ISI/ISO Eco-Label": False, "Origin Label": False, "Batch Code": False},
    "Books": {"Publisher Info": False, "Author Name": False, "ISBN": False, "Edition/Year": False, "Country of Publication": False},
    "Beauty": {"Ingredients List": False, "Batch/Lot Number": False, "Expiry/Best Before Date": False, "Manufacturing License Number": False, "Usage Instructions": False},
    "Toys": {"Age Group Recommendation": False, "Safety Warning Labels": False, "ISI/CE Certification": False, "Material Type": False, "Battery Info (if applicable)": False},
    "Appliances": {"Power Rating (Wattage)": False, "Voltage/Frequency": False, "BIS Certification": False, "Warranty Period": False, "Customer Care Helpline": False},
    "Home": {"Material Type": False, "Dimensions": False, "Load Capacity": False, "Warranty/Guarantee": False}
}

@dataclass
class ComplianceRule:
    flag_name: str
    optional_keywords: List[str] = None
    regex_patterns: List[str] = None
    min_length: int = None

class ThreatLevelAnalyzer:
    def __init__(self):
        self.threat_levels = self._setup_threat_levels()

    def _setup_threat_levels(self):
        layer1_threats = {"Manufacturer/Importer Name & Address": "CRITICAL", "MRP": "HIGH", "Net Quantity": "HIGH", "Date of Manufacture/Expiry": "CRITICAL", "Country of Origin": "MEDIUM", "Consumer Care Details": "MEDIUM"}
        layer2_threats = {
            "Mobiles": {"SAR Value": "CRITICAL", "BIS Certification": "CRITICAL", "Battery Capacity": "HIGH", "Importer/Distributor Details": "HIGH", "Charger/Adapter Info": "MEDIUM", "Warranty Period": "MEDIUM", "Customer Care Helpline": "MEDIUM"},
            "Electronics": {"BIS Certification": "CRITICAL", "Power Consumption/Voltage": "HIGH", "Importer/Distributor Details": "HIGH", "Warranty Period": "MEDIUM", "Customer Care Helpline": "MEDIUM"},
            "Grocery": {"FSSAI Number": "CRITICAL", "Expiry/Best Before Date": "CRITICAL", "Allergen Info": "CRITICAL", "Ingredients List": "HIGH", "Batch/Lot Number": "HIGH", "Nutritional Info": "MEDIUM", "Storage Instructions": "MEDIUM", "ISO/BRC/Organic Certification": "LOW"},
            "Beauty": {"Manufacturing License Number": "CRITICAL", "Ingredients List": "CRITICAL", "Expiry/Best Before Date": "CRITICAL", "Usage Instructions": "HIGH", "Batch/Lot Number": "HIGH"},
            "Toys": {"Age Group Recommendation": "CRITICAL", "Safety Warning Labels": "CRITICAL", "ISI/CE Certification": "CRITICAL", "Material Type": "HIGH", "Battery Info (if applicable)": "MEDIUM"},
        }
        return {"layer1": layer1_threats, "layer2": layer2_threats}

    def get_threat_level(self, flag_name: str, category: str) -> str:
        if flag_name in self.threat_levels["layer1"]: return self.threat_levels["layer1"][flag_name]
        if category in self.threat_levels["layer2"] and flag_name in self.threat_levels["layer2"][category]: return self.threat_levels["layer2"][category][flag_name]
        return "MEDIUM" 

    def calculate_threat_score(self, missing_flags: Dict[str, str]) -> Dict:
        threat_counts = defaultdict(int)
        threat_weights = {"CRITICAL": 10, "HIGH": 7, "MEDIUM": 4, "LOW": 1}
        for threat_level in missing_flags.values(): threat_counts[threat_level] += 1
        total_score = sum(threat_counts[level] * threat_weights[level] for level in threat_counts)
        if threat_counts["CRITICAL"] > 0: overall_threat = "CRITICAL"
        elif threat_counts["HIGH"] > 1: overall_threat = "HIGH"
        elif total_score > 10: overall_threat = "MEDIUM"
        elif total_score > 0: overall_threat = "LOW"
        else: overall_threat = "NONE"
        return {"overall_threat_level": overall_threat, "threat_score": total_score, "threat_breakdown": dict(threat_counts), "total_missing_flags": len(missing_flags)}

class LocalComplianceModel:
    def __init__(self):
        self.rules = {}
        self.category_rules = defaultdict(dict)
        self._setup_rules()
    def _setup_rules(self):
        self.rules = {
            "Manufacturer/Importer Name & Address": ComplianceRule("Manufacturer/Importer Name & Address", optional_keywords=["manufactured", "mfd", "importer", "marketed by", "address"], regex_patterns=[r'manufactured.*by', r'mfd.*by', r'marketed.*by', r'\d{6}']),
            "MRP": ComplianceRule("MRP", optional_keywords=["mrp", "price", "rs"], regex_patterns=[r'mrp.*₹?\d+', r'₹\d+', r'rs\.?\s*\d+']),
            "Net Quantity": ComplianceRule("Net Quantity", optional_keywords=["net", "quantity", "weight", "net wt"], regex_patterns=[r'net.*[gkml]', r'\d+\s*[gkml]']),
            "Date of Manufacture/Expiry": ComplianceRule("Date of Manufacture/Expiry", optional_keywords=["expiry", "exp", "best before", "mfd", "mfg"], regex_patterns=[r'exp.*\d{1,2}[/-]\d{2,4}', r'best.*before', r'mfg.*date']),
            "Country of Origin": ComplianceRule("Country of Origin", optional_keywords=["made in", "country of origin"], regex_patterns=[r'made.*in.*\w+']),
            "Consumer Care Details": ComplianceRule("Consumer Care Details", optional_keywords=["customer care", "helpline", "contact", "email"], regex_patterns=[r'\d{10}', r'customer.*care'])
        }
        self.category_rules["Grocery"] = {
            "FSSAI Number": ComplianceRule("FSSAI Number", optional_keywords=["fssai", "license"], regex_patterns=[r'fssai.*\d{14}', r'lic.*no.*\d+']),
            "Expiry/Best Before Date": ComplianceRule("Expiry/Best Before Date", optional_keywords=["expiry", "best before", "exp"], regex_patterns=[r'exp.*\d{1,2}[/-]\d{2,4}']),
            "Ingredients List": ComplianceRule("Ingredients List", optional_keywords=["ingredients"]),
            "Nutritional Info": ComplianceRule("Nutritional Info", optional_keywords=["nutrition", "energy", "protein"], regex_patterns=[r'nutrition.*facts', r'per.*100g'])
        }
        self.category_rules["Mobiles"] = { "BIS Certification": ComplianceRule("BIS Certification", optional_keywords=["bis"]), "SAR Value": ComplianceRule("SAR Value", optional_keywords=["sar value"]) }
        self.category_rules["Electronics"] = { "BIS Certification": ComplianceRule("BIS Certification", optional_keywords=["bis"]), "Power Consumption/Voltage": ComplianceRule("Power Consumption/Voltage", optional_keywords=["voltage", "watt"]) }

    def _check_text_against_rule(self, text: str, rule: ComplianceRule) -> bool:
        if not text: return False
        text_lower = text.lower()
        keyword_match = any(k.lower() in text_lower for k in rule.optional_keywords) if rule.optional_keywords else False
        regex_match = any(re.search(p, text_lower, re.IGNORECASE) for p in rule.regex_patterns) if rule.regex_patterns else False
        return keyword_match or regex_match
    def check_compliance(self, combined_text: str, category: str, applicable_flags: List[str]) -> Dict[str, int]:
        results = {}
        for flag in applicable_flags:
            rule = self.rules.get(flag) or self.category_rules.get(category, {}).get(flag)
            results[flag] = 1 if rule and self._check_text_against_rule(combined_text, rule) else 0
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
        resp = requests.get(url, timeout=15)
        img_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=0, paragraph=True)
        return " ".join(results) if results else "No text detected"
    except Exception: return "OCR Error"

def extract_api_text(product):
    text_parts = [
        product.get("description", ""),
        product.get("brand", {}).get("name", ""),
        " | ".join(product.get("highlights", [])),
        "mrp: " + str(product.get("price", "")),
        " | ".join([f"{s.get('title', '')}: {s.get('description', '')}" for s in product.get("specifications", [])])
    ]
    return " | ".join(filter(None, text_parts))

def process_data(products_api):
    model = LocalComplianceModel()
    threat_analyzer = ThreatLevelAnalyzer()
    
    # Create a DataFrame with one row per product, aggregating all text and images
    all_products_data = []
    for p in products_api:
        image_urls = [img['url'] for img in p.get('images', []) if 'url' in img]
        all_products_data.append({
            "product_id": p.get("_id"),
            "category": p.get("category", "Uncategorized").capitalize(),
            "product_name": p.get("name", "Unknown Product"),
            "api_text": extract_api_text(p),
            "image_urls": image_urls
        })
    df = pd.DataFrame(all_products_data)

    # Run OCR and combine text
    ocr_texts = [" ".join([run_ocr(url) for url in url_list]) for url_list in df['image_urls']]
    df['ocr_text'] = ocr_texts
    df['combined_text'] = df['api_text'] + " " + df['ocr_text']
    
    # Run compliance and threat analysis
    analysis_results = []
    for _, row in df.iterrows():
        applicable_flags = list(layer1.keys()) + list(layer2.get(row["category"], {}).keys())
        flags = model.check_compliance(row['combined_text'], row['category'], applicable_flags)
        missing_flags = {flag: threat_analyzer.get_threat_level(flag, row['category']) for flag, v in flags.items() if v == 0}
        threat_analysis = threat_analyzer.calculate_threat_score(missing_flags)
        analysis_results.append({
            'flags': flags,
            'missing_flags_display': list(missing_flags.keys()),
            **threat_analysis
        })
    
    df_analysis = pd.DataFrame(analysis_results)
    return pd.concat([df.drop(columns=['api_text', 'ocr_text', 'combined_text']), df_analysis], axis=1)

# Streamlit App Layout
st.set_page_config(page_title="Compliance Checker Dashboard", layout="wide")
st.title(" E-Commerce Compliance Dashboard")
st.markdown("An automated tool to scan and validate e-commerce listings against Legal Metrology requirements.")

@st.cache_data(ttl=900) # Cache data for 15 minutes
def load_and_process_data():
    with st.spinner("Fetching product data from the API..."):
        products_api = fetch_products()
    if not products_api:
        st.error("Could not fetch any products. Please check the API connection.")
        return pd.DataFrame()
    
    with st.spinner("Analyzing products... This may take a few minutes for OCR processing."):
        report = process_data(products_api)
    return report

df_report = load_and_process_data()

if not df_report.empty:
    st.header("Dashboard Overview")
    total_products, critical_threats, high_threats = len(df_report), (df_report['overall_threat_level'] == 'CRITICAL').sum(), (df_report['overall_threat_level'] == 'HIGH').sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products Analyzed", total_products)
    col2.metric("Products with CRITICAL Threats", critical_threats, help="Products missing one or more critical compliance labels (e.g., FSSAI No., Expiry Date).")
    col3.metric("Products with HIGH Threats", high_threats, help="Products missing important regulatory labels (e.g., MRP, Net Quantity).")

    # Sidebar Filters
    st.sidebar.header("Filter Options")
    categories = sorted(df_report['category'].unique())
    selected_category = st.sidebar.multiselect('Filter by Category:', options=categories, default=categories)
    threat_levels = sorted(df_report['overall_threat_level'].unique())
    selected_threat = st.sidebar.multiselect('Filter by Threat Level:', options=threat_levels, default=threat_levels)

    df_filtered = df_report[df_report['category'].isin(selected_category) & df_report['overall_threat_level'].isin(selected_threat)]

    # Main Content
    tab1, tab2 = st.tabs(["📊 Summary View", "📄 Detailed Report"])
    with tab1:
        st.subheader("Threat Level Distribution")
        threat_counts = df_filtered['overall_threat_level'].value_counts()
        st.bar_chart(threat_counts)
        st.subheader("Compliance Issues by Category")
        df_filtered['total_missing'] = df_filtered['total_missing_flags']
        category_issues = df_filtered.groupby('category')['total_missing'].sum()
        st.bar_chart(category_issues)

    with tab2:
        st.subheader("Detailed Compliance Data")
        display_cols = ['product_name', 'category', 'overall_threat_level', 'threat_score', 'total_missing_flags', 'missing_flags_display']
        st.dataframe(df_filtered[display_cols].rename(columns={'missing_flags_display': 'Missing Flags'}), use_container_width=True)

else:
    st.info("Awaiting data to generate the report. If this persists, the API might be down.")

