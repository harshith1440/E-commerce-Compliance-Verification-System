import requests
import pandas as pd
import easyocr
import cv2
import numpy as np
import re
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
from flask import Flask, jsonify, request
import io
from PIL import Image
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR cache and other initializations remain the same
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

# Layer 2 (category specific) - keep all the same
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

# Track completed products and store results
completed_products = set()
processed_results = []  # Store all processed results

# Init EasyOCR
reader = easyocr.Reader(["en"], gpu=False)

# [Keep all the existing classes and functions as they are]
# Preprocess + OCR
def run_ocr(url):
    try:
        resp = requests.get(url, timeout=10)
        img_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # --- preprocessing ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        results = reader.readtext(thresh, detail=1, paragraph=True)

        text_parts = [res[1] for res in results if len(res) >= 2]
        return " ".join(text_parts) if text_parts else "No text detected"

    except Exception as e:
        return f"OCR Error: {str(e)}"

def run_ocr_on_uploaded_image(image_file):
    try:
        # Convert uploaded file to numpy array
        image = Image.open(image_file).convert('RGB')
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # --- preprocessing ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        results = reader.readtext(thresh, detail=1, paragraph=True)

        text_parts = [res[1] for res in results if len(res) >= 2]
        return " ".join(text_parts) if text_parts else "No text detected"

    except Exception as e:
        return f"OCR Error: {str(e)}"

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
    """Enhanced threat level analyzer with more sophisticated threat assessment"""
    def __init__(self):
        self.threat_levels = self._setup_threat_levels()
        self.regulatory_priority = self._setup_regulatory_priority()
        self.business_impact = self._setup_business_impact()

    def _setup_threat_levels(self):
        """Define enhanced threat levels for each compliance flag"""
        # Layer 1 threat levels (common for all categories) - Enhanced
        layer1_threats = {
            "Manufacturer/Importer Name & Address": "CRITICAL",  # Legal requirement
            "MRP": "CRITICAL",  # Price regulation mandate
            "Net Quantity": "HIGH",  # Consumer protection law
            "Date of Manufacture/Expiry": "CRITICAL",  # Safety and legal requirement
            "Country of Origin": "HIGH",  # Trade and customs requirement
            "Consumer Care Details": "MEDIUM"  # Consumer service requirement
        }

        # Layer 2 category-specific threat levels - Enhanced with more granular assessment
        layer2_threats = {
            "Mobiles": {
                "SAR Value": "CRITICAL",  # Health safety regulation
                "Battery Capacity": "HIGH",  # Technical specification requirement
                "Charger/Adapter Info": "MEDIUM",  # Accessory information
                "BIS Certification": "CRITICAL",  # Mandatory certification
                "Warranty Period": "MEDIUM",  # Consumer protection
                "Customer Care Helpline": "MEDIUM",  # Service requirement
                "Importer/Distributor Details": "HIGH"  # Supply chain transparency
            },
            "Electronics": {
                "BIS Certification": "CRITICAL",  # Mandatory for electronics
                "Warranty Period": "MEDIUM",
                "Power Consumption/Voltage": "HIGH",  # Safety specification
                "Customer Care Helpline": "MEDIUM",
                "Importer/Distributor Details": "HIGH"
            },
            "Grocery": {
                "FSSAI Number": "CRITICAL",  # Food safety mandate
                "Expiry/Best Before Date": "CRITICAL",  # Food safety
                "Batch/Lot Number": "HIGH",  # Traceability requirement
                "Ingredients List": "CRITICAL",  # Allergen and transparency
                "Allergen Info": "CRITICAL",  # Health safety
                "Storage Instructions": "HIGH",  # Food safety guidance
                "Nutritional Info": "HIGH",  # Health information
                "ISO/BRC/Organic Certification": "LOW"  # Optional certification
            },
            "Furniture": {
                "Material Type": "HIGH",  # Safety and quality
                "Finish/Polish Details": "MEDIUM",
                "Load Capacity": "HIGH",  # Safety specification
                "Fire Resistance/IS Certification": "CRITICAL",  # Safety mandate
                "Assembly Instructions": "MEDIUM",
                "Warranty/Guarantee": "MEDIUM"
            },
            "Lifestyle": {
                "Size/Measurement": "MEDIUM",
                "Fabric/Material Composition": "HIGH",  # Textile labeling law
                "Care Instructions": "MEDIUM",  # Consumer guidance
                "ISI/ISO Eco-Label": "LOW",
                "Origin Label": "HIGH",  # Textile origin requirement
                "Batch Code": "MEDIUM"
            },
            "Books": {
                "Publisher Info": "HIGH",  # Publishing requirement
                "Author Name": "MEDIUM",
                "ISBN": "HIGH",  # Cataloging requirement
                "Edition/Year": "LOW",
                "Country of Publication": "MEDIUM"
            },
            "Beauty": {
                "Ingredients List": "CRITICAL",  # Cosmetic safety law
                "Batch/Lot Number": "HIGH",  # Traceability
                "Expiry/Best Before Date": "CRITICAL",  # Safety
                "Manufacturing License Number": "CRITICAL",  # Regulatory compliance
                "Usage Instructions": "HIGH"  # Safety guidance
            },
            "Toys": {
                "Age Group Recommendation": "CRITICAL",  # Child safety
                "Safety Warning Labels": "CRITICAL",  # Child protection
                "ISI/CE Certification": "CRITICAL",  # Safety certification
                "Material Type": "HIGH",  # Safety specification
                "Battery Info (if applicable)": "HIGH"  # Safety for battery toys
            },
            "Appliances": {
                "Power Rating (Wattage)": "HIGH",  # Energy and safety
                "Voltage/Frequency": "HIGH",  # Electrical safety
                "BIS Certification": "CRITICAL",  # Mandatory certification
                "Warranty Period": "MEDIUM",
                "Customer Care Helpline": "MEDIUM"
            },
            "Home": {
                "Material Type": "HIGH",
                "Dimensions": "MEDIUM",
                "Load Capacity": "HIGH",  # Safety specification
                "Warranty/Guarantee": "MEDIUM"
            }
        }

        return {
            "layer1": layer1_threats,
            "layer2": layer2_threats
        }

    def _setup_regulatory_priority(self):
        """Define regulatory priority scores for different types of compliance"""
        return {
            "CRITICAL": {
                "regulatory_weight": 1.0,  # Full regulatory impact
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
        """Define business impact for different threat levels"""
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
        return "MEDIUM"

    def calculate_threat_score(self, missing_flags: Dict[str, str], category: str = None) -> Dict:
        """Enhanced threat score calculation with regulatory and business impact"""
        threat_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        threat_weights = {"CRITICAL": 10, "HIGH": 7, "MEDIUM": 4, "LOW": 1}
        regulatory_weights = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.1}

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

        # Enhanced overall threat determination
        critical_flags = threat_counts["CRITICAL"]
        high_flags = threat_counts["HIGH"]
        
        if critical_flags >= 3 or (critical_flags >= 1 and threat_percentage >= 70):
            overall_threat = "CRITICAL"
            urgency = "IMMEDIATE"
        elif critical_flags >= 1 or (high_flags >= 3) or threat_percentage >= 60:
            overall_threat = "HIGH"
            urgency = "HIGH"
        elif high_flags >= 1 or threat_percentage >= 30:
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
                "compliance_gap_severity": "HIGH" if critical_flags > 0 else "MEDIUM" if high_flags > 0 else "LOW"
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
        # Layer1 rules (common for all categories)
        self.rules = {
            "Manufacturer/Importer Name & Address": ComplianceRule(
                flag_name="Manufacturer/Importer Name & Address",
                optional_keywords=["manufactured", "mfd", "importer", "marketed by", "address", "ltd", "private limited"],
                regex_patterns=[r'manufactured.*by', r'mfd.*by', r'marketed.*by', r'importer.*:', r'\\d{6}', r'ltd\\.?'],
                min_length=10
            ),
            "MRP": ComplianceRule(
                flag_name="MRP",
                optional_keywords=["mrp", "price", "rs", "maximum retail price"],
                regex_patterns=[r'mrp.*₹?\\d+', r'₹\\d+', r'rs\\.?\\s*\\d+', r'\\$\\d+', r'price.*\\d+']
            ),
            "Net Quantity": ComplianceRule(
                flag_name="Net Quantity",
                optional_keywords=["net", "quantity", "weight", "volume"],
                regex_patterns=[r'net.*\\d+.*[gkml]', r'\\d+\\s*[gkml]', r'quantity.*\\d+', r'weight.*\\d+']
            ),
            "Date of Manufacture/Expiry": ComplianceRule(
                flag_name="Date of Manufacture/Expiry",
                optional_keywords=["expiry", "exp", "best before", "use by", "mfd", "manufactured"],
                regex_patterns=[r'exp.*\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}', r'best.*before.*\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}']
            ),
            "Country of Origin": ComplianceRule(
                flag_name="Country of Origin",
                optional_keywords=["made in", "country of origin", "manufactured in", "origin"],
                regex_patterns=[r'made.*in.*\\w+', r'country.*origin.*\\w+']
            ),
            "Consumer Care Details": ComplianceRule(
                flag_name="Consumer Care Details",
                optional_keywords=["customer care", "helpline", "support", "contact", "call"],
                regex_patterns=[r'\\d{10}', r'customer.*care', r'helpline.*\\d+']
            )
        }

        # Category-specific rules
        self.category_rules["Grocery"] = {
            "FSSAI Number": ComplianceRule(
                flag_name="FSSAI Number",
                optional_keywords=["fssai", "food safety", "license"],
                regex_patterns=[r'fssai.*\\d{14}', r'lic.*no.*\\d+', r'license.*\\d+']
            ),
            "Expiry/Best Before Date": ComplianceRule(
                flag_name="Expiry/Best Before Date",
                optional_keywords=["expiry", "best before", "exp", "use by"],
                regex_patterns=[r'exp.*\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}', r'best.*before']
            ),
            "Ingredients List": ComplianceRule(
                flag_name="Ingredients List",
                optional_keywords=["ingredients", "contains", "composition"],
                regex_patterns=[r'ingredients.*:', r'contains.*\\w+']
            ),
            "Nutritional Info": ComplianceRule(
                flag_name="Nutritional Info",
                optional_keywords=["nutrition", "calories", "protein", "carbs", "fat", "energy"],
                regex_patterns=[r'nutrition.*facts', r'per.*100g', r'energy.*kcal']
            )
        }

    def _check_text_against_rule(self, text: str, rule: ComplianceRule) -> bool:
        if not text or text.strip() == "":
            return False

        text_lower = text.lower()

        if rule.min_length and len(text.strip()) < rule.min_length:
            return False

        if rule.forbidden_keywords:
            for keyword in rule.forbidden_keywords:
                if keyword.lower() in text_lower:
                    return False

        if rule.required_keywords:
            for keyword in rule.required_keywords:
                if keyword.lower() not in text_lower:
                    return False

        if rule.optional_keywords:
            found = False
            for keyword in rule.optional_keywords:
                if keyword.lower() in text_lower:
                    found = True
                    break
            if not found:
                return False

        if rule.regex_patterns:
            found = False
            for pattern in rule.regex_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found = True
                    break
            if not found:
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
    if df_raw_report.empty:
        return df_raw_report
    
    # For System B, we need to merge flags using OR operation
    def merge_flags(flag_series):
        if flag_series.empty:
            return {}
        
        # Get the first flags dictionary as base
        merged_flags = flag_series.iloc[0].copy()
        
        # OR operation: if any image shows compliance (1), mark as compliant
        for flags_dict in flag_series:
            for flag, value in flags_dict.items():
                if value == 1:  # If any image shows compliance
                    merged_flags[flag] = 1
        
        return merged_flags
    
    # Group by product and aggregate
    grouped = df_raw_report.groupby('product_id').agg({
        'category': 'first',
        'product_name': 'first',
        'image_url': lambda x: list(x),  # Collect all image URLs
        'ocr_text': lambda x: ' | '.join(x),  # Combine all OCR texts
        'api_text': 'first',  # API text is same for all images of a product
        'flags': merge_flags  # Use OR operation for flags
    }).reset_index()
    
    # Rename and add additional columns for consistency
    grouped['image_urls'] = grouped['image_url']
    grouped['num_images'] = grouped['image_urls'].apply(len)
    grouped = grouped.drop('image_url', axis=1)

    return grouped

# MAIN ENDPOINT - Process product data from System A
@app.route('/process-product', methods=['POST'])
def process_product():
    """Main endpoint to receive and process product data from System A"""
    try:
        # Log the incoming request
        logger.info(f"Received product processing request from {request.remote_addr}")
        
        product = request.get_json()
        if not product:
            logger.error("No JSON data received")
            return jsonify({
                "status": "error",
                "message": "No product data received"
            }), 400
        
        product_id = product.get("_id", "unknown")
        logger.info(f"Processing product: {product_id}")
        
        # Skip if already processed
        if product_id in completed_products:
            logger.info(f"Product {product_id} already processed, skipping")
            return jsonify({
                "status": "skipped", 
                "message": "Product already processed",
                "product_id": product_id
            })
        
        # Extract basic product info
        product_name = product.get("name", "Unknown")
        category = product.get("category", "Uncategorized")
        
        logger.info(f"Product details - Name: {product_name}, Category: {category}")
        
        # Run OCR on all images
        ocr_results = []
        image_urls = []
        
        if "images" in product and product["images"]:
            logger.info(f"Processing {len(product['images'])} images")
            for i, img in enumerate(product["images"]):
                img_url = img.get("url", None)
                if img_url:
                    logger.info(f"Running OCR on image {i+1}: {img_url}")
                    ocr_text = run_ocr(img_url)
                    image_urls.append(img_url)
                    ocr_results.append({
                        "product_id": product_id,
                        "category": category,
                        "product_name": product_name,
                        "image_url": img_url,
                        "ocr_text": ocr_text
                    })
        else:
            logger.warning(f"No images found for product {product_id}")
            # Create a dummy entry for products without images
            ocr_results.append({
                "product_id": product_id,
                "category": category,
                "product_name": product_name,
                "image_url": None,
                "ocr_text": "No images available"
            })
        
        # Extract API text
        logger.info("Extracting API text")
        extracted = extract_api_text(product)
        
        # Initialize compliance model
        compliance_model = LocalComplianceModel()
        
        # Determine applicable flags
        applicable_flags = list(layer1.keys())
        if category in layer2:
            applicable_flags.extend(list(layer2[category].keys()))
        
        logger.info(f"Checking {len(applicable_flags)} compliance flags")
        
        # Check compliance for each OCR result
        compliance_results = []
        for ocr_result in ocr_results:
            flags = compliance_model.check_compliance(
                ocr_result["ocr_text"], 
                extracted["api_text"], 
                category, 
                applicable_flags
            )
            compliance_results.append({
                "product_id": product_id,
                "category": category,
                "product_name": product_name,
                "image_url": ocr_result["image_url"],
                "ocr_text": ocr_result["ocr_text"],
                "api_text": extracted["api_text"],
                "flags": flags
            })
        
        # Create a DataFrame from the compliance results
        df_raw_report = pd.DataFrame(compliance_results)
        
        # Merge results for the same product (using OR operation for flags)
        if len(compliance_results) > 1:
            df_merged = merge_product_compliance(df_raw_report)
        else:
            df_merged = df_raw_report
        
        # Add threat analysis
        df_final_report = add_threat_analysis(df_merged)
        
        # Convert to dictionary for JSON response
        result = df_final_report.to_dict(orient='records')[0]  # Only one product
        
        # Store the result
        processed_results.append(result)
        
        # Mark as completed
        completed_products.add(product_id)
        
        logger.info(f"Successfully processed product {product_id}")
        
        return jsonify({
            "status": "success",
            "message": f"Product {product_id} processed successfully",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Error processing product: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing product: {str(e)}"
        }), 500

# ENDPOINT - Get all processed results
@app.route('/get-results', methods=['GET'])
def get_results():
    """Get all processed compliance results"""
    try:
        return jsonify({
            "status": "success",
            "total_products": len(processed_results),
            "data": processed_results
        })
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ENDPOINT - Get specific product result
@app.route('/get-result/<product_id>', methods=['GET'])
def get_product_result(product_id):
    """Get compliance result for a specific product"""
    try:
        # Find the product result
        product_result = None
        for result in processed_results:
            if result.get('product_id') == product_id:
                product_result = result
                break
        
        if product_result:
            return jsonify({
                "status": "success",
                "data": product_result
            })
        else:
            return jsonify({
                "status": "not_found",
                "message": f"No results found for product ID: {product_id}"
            }), 404
            
    except Exception as e:
        logger.error(f"Error retrieving product result: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ENDPOINT - Get processing statistics
@app.route('/stats', methods=['GET'])
def get_stats():
    """Get processing statistics"""
    try:
        total_processed = len(processed_results)
        
        if total_processed == 0:
            return jsonify({
                "status": "success",
                "data": {
                    "total_processed": 0,
                    "threat_levels": {},
                    "categories": {},
                    "avg_compliance": 0
                }
            })
        
        # Calculate statistics
        threat_levels = {}
        categories = {}
        total_compliance_score = 0
        
        for result in processed_results:
            # Count threat levels
            threat_level = result.get('overall_threat_level', 'UNKNOWN')
            threat_levels[threat_level] = threat_levels.get(threat_level, 0) + 1
            
            # Count categories
            category = result.get('category', 'UNKNOWN')
            categories[category] = categories.get(category, 0) + 1
            
            # Sum compliance scores
            total_compliance_score += result.get('regulatory_compliance_score', 0)
        
        avg_compliance = round(total_compliance_score / total_processed, 2)
        
        return jsonify({
            "status": "success",
            "data": {
                "total_processed": total_processed,
                "threat_levels": threat_levels,
                "categories": categories,
                "avg_compliance_score": avg_compliance,
                "completed_products": len(completed_products)
            }
        })
        
    except Exception as e:
        logger.error(f"Error calculating stats: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ENDPOINT - Clear all data (for testing)
@app.route('/clear-data', methods=['POST'])
def clear_data():
    """Clear all processed data (use with caution)"""
    try:
        global processed_results, completed_products
        processed_results.clear()
        completed_products.clear()
        
        logger.info("All processed data cleared")
        
        return jsonify({
            "status": "success",
            "message": "All data cleared successfully"
        })
        
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ENDPOINT - Health check
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Product Compliance Processing System B",
        "processed_products": len(processed_results),
        "completed_products": len(completed_products)
    })

# ENDPOINT - Analyze single uploaded image (existing functionality)
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
    logger.info("Starting Product Compliance Processing System B on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)  # System B runs on port 5000