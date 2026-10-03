# compliance_pipeline.py
import requests
import pandas as pd
import easyocr
import cv2
import numpy as np
import re
import os
import gc
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
from flask import Flask, jsonify, request
import io
from PIL import Image

# Memory optimizations for Mac
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Initialize Flask app
app = Flask(__name__)

# Phase 1: Setup
ocr_cache = {}

# Layer 1 (common for all categories)
layer1 = {
    "Manufacturer/Importer Name & Address": False,
    "MRP": False,
    "Net Quantity": False,
    "Date of Manufacture/Expiry": False,
    "Country of Origin": False,
    "Consumer Care Details": False
}

# Layer 2 (category specific)
layer2 = {
    "Mobiles": {
        "SAR Value": False,
        "Battery Capacity": False,
        "Charger/Adapter Info": False,
        "BIS Certification": False,
        "Warranty Period": False,
        "Customer Care Helpline": False,
        "Importer/Distributor Details": False
    },
    "Electronics": {
        "BIS Certification": False,
        "Warranty Period": False,
        "Power Consumption/Voltage": False,
        "Customer Care Helpline": False,
        "Importer/Distributor Details": False
    },
    "Grocery": {
        "FSSAI Number": False,
        "Expiry/Best Before Date": False,
        "Batch/Lot Number": False,
        "Ingredients List": False,
        "Allergen Info": False,
        "Storage Instructions": False,
        "Nutritional Info": False,
        "ISO/BRC/Organic Certification": False
    },
    "Furniture": {
        "Material Type": False,
        "Finish/Polish Details": False,
        "Load Capacity": False,
        "Fire Resistance/IS Certification": False,
        "Assembly Instructions": False,
        "Warranty/Guarantee": False
    },
    "Lifestyle": {
        "Size/Measurement": False,
        "Fabric/Material Composition": False,
        "Care Instructions": False,
        "ISI/ISO Eco-Label": False,
        "Origin Label": False,
        "Batch Code": False
    },
    "Books": {
        "Publisher Info": False,
        "Author Name": False,
        "ISBN": False,
        "Edition/Year": False,
        "Country of Publication": False
    },
    "Beauty": {
        "Ingredients List": False,
        "Batch/Lot Number": False,
        "Expiry/Best Before Date": False,
        "Manufacturing License Number": False,
        "Usage Instructions": False
    },
    "Toys": {
        "Age Group Recommendation": False,
        "Safety Warning Labels": False,
        "ISI/CE Certification": False,
        "Material Type": False,
        "Battery Info (if applicable)": False
    },
    "Appliances": {
        "Power Rating (Wattage)": False,
        "Voltage/Frequency": False,
        "BIS Certification": False,
        "Warranty Period": False,
        "Customer Care Helpline": False
    },
    "Home": {
        "Material Type": False,
        "Dimensions": False,
        "Load Capacity": False,
        "Warranty/Guarantee": False
    }
}

# Track completed products (skip if already processed)
completed_products = set()

# Phase 2: Fetch Data from API
API_URL = "https://flipkart-backend-8z7t.onrender.com/api/v1/product/all-products"

# Init EasyOCR with optimizations
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

# Fetch products with limit
def fetch_products():
    r = requests.get(API_URL)
    if r.status_code == 200:
        products = r.json().get("products", [])
        return products[:6]  # Limit to 6 products for performance
    else:
        return []

# Optimized OCR with caching and memory cleanup
def run_ocr(url):
    # Check cache first
    if url in ocr_cache:
        return ocr_cache[url]
    
    try:
        resp = requests.get(url, timeout=10)
        img_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return "Invalid image"

        # preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        results = reader.readtext(thresh, detail=1, paragraph=True)
        text_parts = [res[1] for res in results if len(res) >= 2]
        
        # Cache result
        result = " ".join(text_parts) if text_parts else "No text detected"
        ocr_cache[url] = result
        
        # Cleanup memory
        del img, gray, blur, thresh
        gc.collect()
        
        return result

    except Exception as e:
        return f"OCR Error: {str(e)}"

# OCR for uploaded image with cleanup
def run_ocr_on_uploaded_image(image_file):
    try:
        # Convert uploaded file to numpy array
        image = Image.open(image_file).convert('RGB')
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        results = reader.readtext(thresh, detail=1, paragraph=True)
        text_parts = [res[1] for res in results if len(res) >= 2]
        
        result = " ".join(text_parts) if text_parts else "No text detected"
        
        # Cleanup memory
        del img, gray, blur, thresh
        gc.collect()
        
        return result

    except Exception as e:
        return f"OCR Error: {str(e)}"

# Extract API text
def extract_api_text(product):
    desc = product.get("description", "")
    brand = product.get("brand", {}).get("name", "")
    highlights = " | ".join(product.get("highlights", []))

    mrp = "mrp: " + str(product.get("price", ""))

    specs_list = product.get("specifications", [])
    specs_texts = []
    for spec in specs_list:
        title = spec.get("title", "")
        detail = spec.get("description", "")
        if title or detail:
            specs_texts.append(f"{title}: {detail}")
    specs_text = " | ".join(specs_texts)

    # combine into single text field
    combined_text = f"Description: {desc} | Brand: {brand} | Highlights: {highlights} | Specs: {specs_text}"
    return {
        "description": desc,
        "brand": brand,
        "highlights": highlights,
        "specifications": specs_text,
        "price": mrp,
        "api_text": combined_text
    }

# Compliance classes
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
        self.regulatory_priority = self._setup_regulatory_priority()
        self.business_impact = self._setup_business_impact()

    def _setup_threat_levels(self):
        """WEAKENED threat levels for hackathon demo"""
        # Layer 1 - Much weaker threat levels
        layer1_threats = {
            "Manufacturer/Importer Name & Address": "MEDIUM",  # Reduced from CRITICAL
            "MRP": "HIGH",  # Reduced from CRITICAL
            "Net Quantity": "MEDIUM",  # Reduced from HIGH
            "Date of Manufacture/Expiry": "HIGH",  # Reduced from CRITICAL
            "Country of Origin": "MEDIUM",  # Reduced from HIGH
            "Consumer Care Details": "LOW"  # Reduced from MEDIUM
        }

        # Layer 2 - Much weaker category-specific threats
        layer2_threats = {
            "Mobiles": {
                "SAR Value": "HIGH",  # Reduced from CRITICAL
                "Battery Capacity": "MEDIUM",  # Reduced from HIGH
                "Charger/Adapter Info": "LOW",  # Reduced from MEDIUM
                "BIS Certification": "HIGH",  # Reduced from CRITICAL
                "Warranty Period": "LOW",  # Reduced from MEDIUM
                "Customer Care Helpline": "LOW",  # Reduced from MEDIUM
                "Importer/Distributor Details": "MEDIUM"  # Reduced from HIGH
            },
            "Electronics": {
                "BIS Certification": "HIGH",  # Reduced from CRITICAL
                "Warranty Period": "LOW",  # Reduced from MEDIUM
                "Power Consumption/Voltage": "MEDIUM",  # Reduced from HIGH
                "Customer Care Helpline": "LOW",  # Reduced from MEDIUM
                "Importer/Distributor Details": "MEDIUM"  # Reduced from HIGH
            },
            "Grocery": {
                "FSSAI Number": "HIGH",  # Reduced from CRITICAL
                "Expiry/Best Before Date": "HIGH",  # Reduced from CRITICAL
                "Batch/Lot Number": "MEDIUM",  # Reduced from HIGH
                "Ingredients List": "HIGH",  # Reduced from CRITICAL
                "Allergen Info": "MEDIUM",  # Reduced from CRITICAL
                "Storage Instructions": "MEDIUM",  # Reduced from HIGH
                "Nutritional Info": "LOW",  # Reduced from HIGH
                "ISO/BRC/Organic Certification": "LOW"  # Same
            },
            "Furniture": {
                "Material Type": "MEDIUM",  # Reduced from HIGH
                "Finish/Polish Details": "LOW",  # Reduced from MEDIUM
                "Load Capacity": "MEDIUM",  # Reduced from HIGH
                "Fire Resistance/IS Certification": "HIGH",  # Reduced from CRITICAL
                "Assembly Instructions": "LOW",  # Reduced from MEDIUM
                "Warranty/Guarantee": "LOW"  # Reduced from MEDIUM
            },
            "Lifestyle": {
                "Size/Measurement": "LOW",  # Reduced from MEDIUM
                "Fabric/Material Composition": "MEDIUM",  # Reduced from HIGH
                "Care Instructions": "LOW",  # Reduced from MEDIUM
                "ISI/ISO Eco-Label": "LOW",  # Same
                "Origin Label": "MEDIUM",  # Reduced from HIGH
                "Batch Code": "LOW"  # Reduced from MEDIUM
            },
            "Books": {
                "Publisher Info": "MEDIUM",  # Reduced from HIGH
                "Author Name": "LOW",  # Reduced from MEDIUM
                "ISBN": "MEDIUM",  # Reduced from HIGH
                "Edition/Year": "LOW",  # Same
                "Country of Publication": "LOW"  # Reduced from MEDIUM
            },
            "Beauty": {
                "Ingredients List": "HIGH",  # Reduced from CRITICAL
                "Batch/Lot Number": "MEDIUM",  # Reduced from HIGH
                "Expiry/Best Before Date": "HIGH",  # Reduced from CRITICAL
                "Manufacturing License Number": "HIGH",  # Reduced from CRITICAL
                "Usage Instructions": "MEDIUM"  # Reduced from HIGH
            },
            "Toys": {
                "Age Group Recommendation": "HIGH",  # Reduced from CRITICAL
                "Safety Warning Labels": "HIGH",  # Reduced from CRITICAL
                "ISI/CE Certification": "HIGH",  # Reduced from CRITICAL
                "Material Type": "MEDIUM",  # Reduced from HIGH
                "Battery Info (if applicable)": "MEDIUM"  # Reduced from HIGH
            },
            "Appliances": {
                "Power Rating (Wattage)": "MEDIUM",  # Reduced from HIGH
                "Voltage/Frequency": "MEDIUM",  # Reduced from HIGH
                "BIS Certification": "HIGH",  # Reduced from CRITICAL
                "Warranty Period": "LOW",  # Reduced from MEDIUM
                "Customer Care Helpline": "LOW"  # Reduced from MEDIUM
            },
            "Home": {
                "Material Type": "MEDIUM",  # Reduced from HIGH
                "Dimensions": "LOW",  # Reduced from MEDIUM
                "Load Capacity": "MEDIUM",  # Reduced from HIGH
                "Warranty/Guarantee": "LOW"  # Reduced from MEDIUM
            }
        }

        return {
            "layer1": layer1_threats,
            "layer2": layer2_threats
        }

    def _setup_regulatory_priority(self):
        """Same priority structure"""
        return {
            "CRITICAL": {
                "regulatory_weight": 1.0,
                "fine_risk": "HIGH",
                "legal_action_risk": "HIGH"
            },
            "HIGH": {
                "regulatory_weight": 0.7,
                "fine_risk": "MEDIUM",
                "legal_action_risk": "MEDIUM"
            },
            "MEDIUM": {
                "regulatory_weight": 0.4,
                "fine_risk": "LOW",
                "legal_action_risk": "LOW"
            },
            "LOW": {
                "regulatory_weight": 0.1,
                "fine_risk": "MINIMAL",
                "legal_action_risk": "MINIMAL"
            }
        }

    def _setup_business_impact(self):
        """Same business impact structure"""
        return {
            "CRITICAL": {
                "marketplace_action": "Product removal/blocking likely",
                "consumer_trust_impact": "HIGH",
                "brand_reputation_risk": "HIGH"
            },
            "HIGH": {
                "marketplace_action": "Warning/compliance notice likely",
                "consumer_trust_impact": "MEDIUM",
                "brand_reputation_risk": "MEDIUM"
            },
            "MEDIUM": {
                "marketplace_action": "Monitoring/review likely",
                "consumer_trust_impact": "LOW",
                "brand_reputation_risk": "LOW"
            },
            "LOW": {
                "marketplace_action": "Minimal action expected",
                "consumer_trust_impact": "MINIMAL",
                "brand_reputation_risk": "MINIMAL"
            }
        }

    def get_threat_level(self, flag_name: str, category: str) -> str:
        """Get threat level for a specific flag and category"""
        # Check layer1 first (common flags)
        if flag_name in self.threat_levels["layer1"]:
            return self.threat_levels["layer1"][flag_name]

        # Check layer2 (category-specific flags)
        if category in self.threat_levels["layer2"]:
            if flag_name in self.threat_levels["layer2"][category]:
                return self.threat_levels["layer2"][category][flag_name]

        # Default threat level if not found
        return "LOW"  # Changed from MEDIUM to LOW

    def calculate_threat_score(self, missing_flags: Dict[str, str], category: str = None) -> Dict:
        """HACKATHON VERSION - Much more lenient scoring"""
        threat_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        threat_weights = {"CRITICAL": 10, "HIGH": 5, "MEDIUM": 2, "LOW": 0.5}  # Reduced weights
        regulatory_weights = {"CRITICAL": 1.0, "HIGH": 0.5, "MEDIUM": 0.2, "LOW": 0.05}  # Much lower

        total_flags = len(missing_flags)
        if total_flags == 0:
            return self._get_no_threat_response()

        # Count missing flags by threat level
        for flag, threat_level in missing_flags.items():
            threat_counts[threat_level] += 1

        # Calculate weighted threat score
        total_score = sum(threat_counts[level] * threat_weights[level]
                         for level in threat_counts)

        # Calculate regulatory compliance score
        regulatory_score = sum(threat_counts[level] * regulatory_weights[level]
                             for level in threat_counts)

        max_possible_score = total_flags * threat_weights["CRITICAL"]
        max_regulatory_score = total_flags * regulatory_weights["CRITICAL"]
        
        threat_percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        regulatory_percentage = (regulatory_score / max_regulatory_score * 100) if max_regulatory_score > 0 else 0

        # MUCH MORE LENIENT threat determination for hackathon
        critical_flags = threat_counts["CRITICAL"]
        high_flags = threat_counts["HIGH"]
        
        # Much higher thresholds for CRITICAL
        if critical_flags >= 5 or (critical_flags >= 3 and threat_percentage >= 90):
            overall_threat = "CRITICAL"
            urgency = "IMMEDIATE"
        elif critical_flags >= 2 or (high_flags >= 5) or threat_percentage >= 70:
            overall_threat = "HIGH"
            urgency = "HIGH"
        elif high_flags >= 2 or threat_percentage >= 40:
            overall_threat = "MEDIUM"
            urgency = "MEDIUM"
        else:
            overall_threat = "LOW"
            urgency = "LOW"

        # Get business impact assessment
        business_impact = self.business_impact.get(overall_threat, {})
        regulatory_impact = self.regulatory_priority.get(overall_threat, {})

        return {
            "overall_threat_level": overall_threat,
            "urgency_level": urgency,
            "threat_score": total_score,
            "threat_percentage": round(threat_percentage, 1),
            "regulatory_compliance_score": round(100 - regulatory_percentage, 1),
            "threat_breakdown": threat_counts,
            "total_missing_flags": total_flags,
            "business_impact": business_impact,
            "regulatory_impact": regulatory_impact,
            "risk_factors": {
                "critical_missing": critical_flags,
                "high_missing": high_flags,
                "compliance_gap_severity": "HIGH" if critical_flags >= 3 else "MEDIUM" if high_flags >= 3 else "LOW"
            }
        }

    def _get_no_threat_response(self):
        """Return response when no flags are missing"""
        return {
            "overall_threat_level": "NONE",
            "urgency_level": "NONE",
            "threat_score": 0,
            "threat_percentage": 0,
            "regulatory_compliance_score": 100,
            "threat_breakdown": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "total_missing_flags": 0,
            "business_impact": {"marketplace_action": "No action expected", "consumer_trust_impact": "POSITIVE", "brand_reputation_risk": "NONE"},
            "regulatory_impact": {"fine_risk": "NONE", "legal_action_risk": "NONE"},
            "risk_factors": {"critical_missing": 0, "high_missing": 0, "compliance_gap_severity": "NONE"}
        }

class LocalComplianceModel:
    def __init__(self):
        self.rules = {}
        self.category_rules = defaultdict(dict)
        self._setup_rules()

    def _setup_rules(self):
        """HACKATHON VERSION - Much more lenient rules"""
        # Layer1 rules - VERY lenient for demo
        self.rules = {
            "Manufacturer/Importer Name & Address": ComplianceRule(
                flag_name="Manufacturer/Importer Name & Address",
                optional_keywords=["brand", "company", "corp", "inc", "ltd", "manufactured", "made", "by"],  # Much more lenient
                min_length=3  # Reduced from 10
            ),
            "MRP": ComplianceRule(
                flag_name="MRP",
                optional_keywords=["price", "mrp", "rs", "₹", "$", "cost", "amount"],  # Very lenient
                regex_patterns=[r'\d+']  # Any number
            ),
            "Net Quantity": ComplianceRule(
                flag_name="Net Quantity",
                optional_keywords=["weight", "quantity", "size", "ml", "kg", "g", "l", "pieces", "pcs"],  # Very lenient
                regex_patterns=[r'\d+']  # Any number
            ),
            "Date of Manufacture/Expiry": ComplianceRule(
                flag_name="Date of Manufacture/Expiry",
                optional_keywords=["date", "exp", "mfg", "manufactured", "expiry", "best", "before"],  # Very lenient
                regex_patterns=[r'\d{4}', r'\d{2}', r'20\d{2}']  # Any year or 2-digit number
            ),
            "Country of Origin": ComplianceRule(
                flag_name="Country of Origin",
                optional_keywords=["india", "china", "usa", "made", "origin", "country", "imported"],  # Common countries
                min_length=2
            ),
            "Consumer Care Details": ComplianceRule(
                flag_name="Consumer Care Details",
                optional_keywords=["contact", "phone", "email", "support", "help", "care", "service", "@"],  # Very lenient
                regex_patterns=[r'\d{7,}', r'@']  # Any 7+ digit number or email
            )
        }

        # Category-specific rules - Also very lenient
        self.category_rules["Grocery"] = {
            "FSSAI Number": ComplianceRule(
                flag_name="FSSAI Number",
                optional_keywords=["fssai", "license", "food", "safety", "lic"],
                regex_patterns=[r'\d{10,}']  # Any 10+ digit number
            ),
            "Expiry/Best Before Date": ComplianceRule(
                flag_name="Expiry/Best Before Date",
                optional_keywords=["exp", "expiry", "best", "before", "use"],
                regex_patterns=[r'\d{4}', r'\d{2}']  # Any year
            ),
            "Ingredients List": ComplianceRule(
                flag_name="Ingredients List",
                optional_keywords=["ingredients", "contains", "made", "wheat", "sugar", "salt"],
                min_length=5
            ),
            "Nutritional Info": ComplianceRule(
                flag_name="Nutritional Info",
                optional_keywords=["energy", "protein", "fat", "carbs", "nutrition", "kcal", "calories"],
                min_length=3
            )
        }

    def _check_text_against_rule(self, text: str, rule: ComplianceRule) -> bool:
        """HACKATHON VERSION - Much more lenient checking"""
        if not text or text.strip() == "":
            return False

        text_lower = text.lower()

        # Much more lenient length check
        if rule.min_length and len(text.strip()) < rule.min_length:
            return False

        # Skip forbidden keywords for hackathon
        # if rule.forbidden_keywords:
        #     for keyword in rule.forbidden_keywords:
        #         if keyword.lower() in text_lower:
        #             return False

        # Skip required keywords for hackathon
        # if rule.required_keywords:
        #     for keyword in rule.required_keywords:
        #         if keyword.lower() not in text_lower:
        #             return False

        # Very lenient optional keywords - only need one match
        if rule.optional_keywords:
            found = False
            for keyword in rule.optional_keywords:
                if keyword.lower() in text_lower:
                    found = True
                    break
            if not found and rule.regex_patterns is None:  # Only fail if no regex fallback
                return False

        # Very lenient regex - only need one match
        if rule.regex_patterns:
            found = False
            for pattern in rule.regex_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found = True
                    break
            if not found and rule.optional_keywords is None:  # Only fail if no keyword fallback
                return False

        return True

    def check_compliance(self, ocr_text: str, api_text: str, category: str, applicable_flags: List[str]) -> Dict[str, int]:
        combined_text = f"{ocr_text} {api_text}".strip()
        results = {}

        for flag in applicable_flags:
            rule = None
            if flag in self.rules:
                rule = self.rules[flag]
            elif category in self.category_rules and flag in self.category_rules[category]:
                rule = self.category_rules[category][flag]

            if rule:
                is_compliant = self._check_text_against_rule(combined_text, rule)
                results[flag] = 1 if is_compliant else 0
            else:
                results[flag] = 0

        return results

    def check_compliance_ocr_only(self, ocr_text: str, applicable_flags: List[str]) -> Dict[str, int]:
        """Check compliance using only OCR text (for single image endpoint)"""
        results = {}

        for flag in applicable_flags:
            rule = None
            if flag in self.rules:  # Only check Layer 1 rules
                rule = self.rules[flag]

            if rule:
                is_compliant = self._check_text_against_rule(ocr_text, rule)
                results[flag] = 1 if is_compliant else 0
            else:
                results[flag] = 0

        return results

def add_threat_analysis(df_final_report):
    """Add enhanced threat level analysis to the final compliance report"""
    threat_analyzer = ThreatLevelAnalyzer()

    # Add new columns for threat analysis
    threat_data = []

    for idx, row in df_final_report.iterrows():
        category = row['category']
        flags = row['flags']

        # Find missing flags (value = 0)
        missing_flags = {flag: threat_analyzer.get_threat_level(flag, category)
                        for flag, value in flags.items() if value == 0}

        # Calculate enhanced threat analysis
        if missing_flags:
            threat_analysis = threat_analyzer.calculate_threat_score(missing_flags, category)
        else:
            threat_analysis = threat_analyzer._get_no_threat_response()

        threat_data.append({
            'missing_flags': missing_flags,
            'overall_threat_level': threat_analysis['overall_threat_level'],
            'urgency_level': threat_analysis['urgency_level'],
            'threat_score': threat_analysis['threat_score'],
            'threat_percentage': threat_analysis['threat_percentage'],
            'regulatory_compliance_score': threat_analysis['regulatory_compliance_score'],
            'critical_missing': threat_analysis['threat_breakdown']['CRITICAL'],
            'high_missing': threat_analysis['threat_breakdown']['HIGH'],
            'medium_missing': threat_analysis['threat_breakdown']['MEDIUM'],
            'low_missing': threat_analysis['threat_breakdown']['LOW'],
            'total_missing_flags': threat_analysis['total_missing_flags'],
            'business_impact': threat_analysis['business_impact'],
            'regulatory_impact': threat_analysis['regulatory_impact'],
            'risk_factors': threat_analysis['risk_factors']
        })

    # Add threat analysis columns to the dataframe
    for key in threat_data[0].keys():
        df_final_report[key] = [item[key] for item in threat_data]

    return df_final_report

def merge_product_compliance(df_raw_report):
    """Merge multiple rows for the same product using OR operator for flags"""
    # Group by product and aggregate
    grouped = df_raw_report.groupby('product_id').agg({
        'category': 'first',
        'product_name': 'first',
        'image_urls': 'first',
        'num_images': 'first',
        'flags': lambda x: x.iloc[0]  # Take first flags (they should be the same after OR operation)
    }).reset_index()

    return grouped

def run_full_compliance_pipeline(layer1, layer2):
    """Run the full compliance pipeline and return the final report"""
    
    # Fetch products (limited to 6)
    products = fetch_products()
    print(f"Fetched {len(products)} products for processing")
    
    # Run OCR for all products and all images
    ocr_results = []
    for i, p in enumerate(products):
        product_id = p.get("_id", "unknown")
        product_name = p.get("name", "Unknown")
        category = p.get("category", "Uncategorized")
        
        print(f"Processing product {i+1}/{len(products)}: {product_name}")

        if "images" in p and p["images"]:
            # Limit to first 3 images per product for performance
            for j, img in enumerate(p["images"][:3]):
                img_url = img.get("url", None)
                print(f"  Processing image {j+1}/{min(len(p['images']), 3)}")
                ocr_text = run_ocr(img_url) if img_url else "No image found"

                ocr_results.append({
                    "product_id": product_id,
                    "category": category,
                    "product_name": product_name,
                    "image_url": img_url,
                    "ocr_text": ocr_text
                })

    # Create OCR dataframe
    df_ocr = pd.DataFrame(ocr_results)
    
    # Process all products for API data
    api_results = []
    for p in products:
        product_id = p.get("_id", "unknown")
        category = p.get("category", "Uncategorized")
        product_name = p.get("name", "Unknown")

        extracted = extract_api_text(p)

        api_results.append({
            "product_id": product_id,
            "category": category,
            "product_name": product_name,
            "description": extracted["description"],
            "brand": extracted["brand"],
            "highlights": extracted["highlights"],
            "specifications": extracted["specifications"],
            "price": extracted["price"],
            "api_text": extracted["api_text"]
        })

    # Create API dataframe
    df_api = pd.DataFrame(api_results)
    
    # Merge OCR and API data
    df_merged = df_ocr.merge(df_api, on=["product_id", "category", "product_name"], how="left")
    
    # Final structure
    df_final = df_merged[[
        "product_id",
        "category",
        "product_name",
        "image_url",
        "ocr_text",
        "api_text",
        "description",
        "brand",
        "highlights",
        "specifications",
        "price"
    ]]
    
    # Group by product to get all image URLs
    product_image_urls = df_final.groupby('product_id')['image_url'].apply(list).reset_index()
    product_image_urls.columns = ['product_id', 'image_urls']
    
    # Get number of images per product
    product_image_urls['num_images'] = product_image_urls['image_urls'].apply(len)
    
    # Get first row for each product for other data
    product_other_data = df_final.groupby('product_id').first().reset_index()
    product_other_data = product_other_data.drop('image_url', axis=1)
    
    # Merge back with image URLs
    df_final_grouped = product_other_data.merge(product_image_urls, on='product_id')
    
    # Initialize compliance model
    compliance_model = LocalComplianceModel()
    
    # Check compliance for each product
    compliance_results = []
    for idx, row in df_final_grouped.iterrows():
        # Get applicable flags based on category
        applicable_flags = list(layer1.keys())
        if row['category'] in layer2:
            applicable_flags.extend(list(layer2[row['category']].keys()))
        
        # Check compliance
        flags = compliance_model.check_compliance(
            row['ocr_text'], 
            row['api_text'], 
            row['category'], 
            applicable_flags
        )
        
        compliance_results.append(flags)
    
    # Add compliance results to dataframe
    df_final_grouped['flags'] = compliance_results
    
    # Add enhanced threat analysis
    df_final_report = add_threat_analysis(df_final_grouped)
    
    return df_final_report

# Flask Routes

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "endpoints": {
            "compliance_check": "/run-compliance",
            "image_analysis": "/analyze-image"
        },
        "products_limit": 6,
        "images_per_product_limit": 3,
        "mode": "HACKATHON - Lenient Rules"
    })

# Flask endpoint for full compliance pipeline
@app.route('/run-compliance', methods=['GET'])
def run_compliance():
    try:
        print("🚀 HACKATHON MODE: Starting lenient compliance pipeline...")
        result_df = run_full_compliance_pipeline(layer1, layer2)
        print(f"✅ HACKATHON: Pipeline completed successfully. Processed {len(result_df)} products.")
        
        return jsonify({
            "status": "success",
            "products_processed": len(result_df),
            "mode": "HACKATHON_LENIENT",
            "data": result_df.to_dict(orient='records')
        })
    except Exception as e:
        print(f"❌ HACKATHON: Pipeline error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# New endpoint for single image analysis
@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Endpoint to analyze a single uploaded image for compliance.
    Only uses Layer 1 flags and OCR text analysis.
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file provided"
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No image file selected"
            }), 400
        
        # Run OCR on uploaded image
        ocr_text = run_ocr_on_uploaded_image(image_file)
        
        # Initialize compliance model
        compliance_model = LocalComplianceModel()
        
        # Get Layer 1 flags only (common compliance requirements)
        applicable_flags = list(layer1.keys())
        
        # Check compliance using only OCR text
        flags = compliance_model.check_compliance_ocr_only(ocr_text, applicable_flags)
        
        # Initialize threat analyzer
        threat_analyzer = ThreatLevelAnalyzer()
        
        # Find missing flags (value = 0)
        missing_flags = {flag: threat_analyzer.get_threat_level(flag, "General")
                        for flag, value in flags.items() if value == 0}
        
        # Calculate threat analysis
        if missing_flags:
            threat_analysis = threat_analyzer.calculate_threat_score(missing_flags, "General")
        else:
            threat_analysis = threat_analyzer._get_no_threat_response()
        
        # Prepare response
        result = {
            "image_analysis": {
                "ocr_text": ocr_text,
                "flags_checked": flags,
                "compliance_summary": {
                    "total_flags": len(flags),
                    "compliant_flags": sum(flags.values()),
                    "non_compliant_flags": len(flags) - sum(flags.values()),
                    "compliance_percentage": round((sum(flags.values()) / len(flags)) * 100, 1) if flags else 0
                }
            },
            "threat_analysis": {
                "missing_flags": missing_flags,
                "overall_threat_level": threat_analysis['overall_threat_level'],
                "urgency_level": threat_analysis['urgency_level'],
                "threat_score": threat_analysis['threat_score'],
                "threat_percentage": threat_analysis['threat_percentage'],
                "regulatory_compliance_score": threat_analysis['regulatory_compliance_score'],
                "threat_breakdown": {
                    "critical_missing": threat_analysis['threat_breakdown']['CRITICAL'],
                    "high_missing": threat_analysis['threat_breakdown']['HIGH'],
                    "medium_missing": threat_analysis['threat_breakdown']['MEDIUM'],
                    "low_missing": threat_analysis['threat_breakdown']['LOW']
                },
                "business_impact": threat_analysis['business_impact'],
                "regulatory_impact": threat_analysis['regulatory_impact'],
                "risk_factors": threat_analysis['risk_factors']
            },
            "recommendations": generate_compliance_recommendations(missing_flags, threat_analysis)
        }
        
        return jsonify({
            "status": "success",
            "mode": "HACKATHON_LENIENT",
            "data": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error analyzing image: {str(e)}"
        }), 500

def generate_compliance_recommendations(missing_flags: Dict[str, str], threat_analysis: Dict) -> Dict:
    """Generate actionable recommendations based on missing compliance flags"""
    
    recommendations = {
        "immediate_actions": [],
        "short_term_actions": [],
        "long_term_actions": [],
        "priority_order": []
    }
    
    # Categorize missing flags by threat level and provide specific recommendations
    flag_recommendations = {
        "Manufacturer/Importer Name & Address": {
            "action": "Add complete manufacturer/importer name and full address including pincode",
            "priority": "IMMEDIATE",
            "details": "Legal requirement for product identification and accountability"
        },
        "MRP": {
            "action": "Clearly display Maximum Retail Price (MRP) in Indian Rupees",
            "priority": "IMMEDIATE",
            "details": "Mandatory under Legal Metrology Act for consumer protection"
        },
        "Net Quantity": {
            "action": "Display net quantity/weight in appropriate units (grams, kg, ml, liters)",
            "priority": "IMMEDIATE",
            "details": "Required for consumer awareness and fair trade practices"
        },
        "Date of Manufacture/Expiry": {
            "action": "Add manufacturing date and expiry/best before date in DD/MM/YYYY format",
            "priority": "IMMEDIATE",
            "details": "Critical for product safety and shelf-life management"
        },
        "Country of Origin": {
            "action": "Mention country of origin or 'Made in [Country]' marking",
            "priority": "HIGH",
            "details": "Required for customs and trade regulations"
        },
        "Consumer Care Details": {
            "action": "Provide customer care helpline number or email for consumer support",
            "priority": "MEDIUM",
            "details": "Helps in building consumer trust and addressing grievances"
        }
    }
    
    # Sort missing flags by threat level
    critical_flags = [flag for flag, level in missing_flags.items() if level == "CRITICAL"]
    high_flags = [flag for flag, level in missing_flags.items() if level == "HIGH"]
    medium_flags = [flag for flag, level in missing_flags.items() if level == "MEDIUM"]
    low_flags = [flag for flag, level in missing_flags.items() if level == "LOW"]
    
    # Generate immediate actions for critical flags
    for flag in critical_flags:
        if flag in flag_recommendations:
            recommendations["immediate_actions"].append({
                "flag": flag,
                "action": flag_recommendations[flag]["action"],
                "details": flag_recommendations[flag]["details"],
                "threat_level": "CRITICAL"
            })
            recommendations["priority_order"].append({"flag": flag, "priority": 1})
    
    # Generate short-term actions for high flags
    for flag in high_flags:
        if flag in flag_recommendations:
            recommendations["short_term_actions"].append({
                "flag": flag,
                "action": flag_recommendations[flag]["action"],
                "details": flag_recommendations[flag]["details"],
                "threat_level": "HIGH"
            })
            recommendations["priority_order"].append({"flag": flag, "priority": 2})
    
    # Generate long-term actions for medium and low flags
    for flag in medium_flags + low_flags:
        if flag in flag_recommendations:
            recommendations["long_term_actions"].append({
                "flag": flag,
                "action": flag_recommendations[flag]["action"],
                "details": flag_recommendations[flag]["details"],
                "threat_level": missing_flags[flag]
            })
            recommendations["priority_order"].append({"flag": flag, "priority": 3})
    
    # Add general recommendations based on threat level
    overall_threat = threat_analysis.get('overall_threat_level', 'MEDIUM')
    
    if overall_threat == "CRITICAL":
        recommendations["general_advice"] = {
            "urgency": "IMMEDIATE ACTION REQUIRED",
            "message": "Product has critical compliance gaps that may result in marketplace removal or legal action. Address immediately before listing/selling.",
            "timeline": "Within 24-48 hours"
        }
    elif overall_threat == "HIGH":
        recommendations["general_advice"] = {
            "urgency": "HIGH PRIORITY",
            "message": "Product has significant compliance issues that need prompt attention to avoid regulatory scrutiny.",
            "timeline": "Within 1-2 weeks"
        }
    elif overall_threat == "MEDIUM":
        recommendations["general_advice"] = {
            "urgency": "MODERATE PRIORITY",
            "message": "Product has some compliance gaps that should be addressed for better consumer trust and regulatory compliance.",
            "timeline": "Within 1 month"
        }
    else:
        recommendations["general_advice"] = {
            "urgency": "LOW PRIORITY",
            "message": "Product has minor compliance improvements that can be addressed during next packaging update.",
            "timeline": "Next packaging revision"
        }
    
    return recommendations

if __name__ == "__main__":
    app.run(debug=True)