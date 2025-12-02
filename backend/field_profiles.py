"""
Centralized field profiles for job matching.

This module defines `_FIELD_KEYWORDS` and `FIELD_PROFILES` used by the
backend matcher. Keeping them in a separate module makes it easier to
export, edit, or replace with a DB-backed configuration in the future.
"""
from typing import Dict, Any
import json

# Keywords used to detect a job's field from title/description text
_FIELD_KEYWORDS = {
    "data": ["data", "analyst", "data scientist", "machine learning", "ml", "pandas", "analytics", "data engineer", "etl", "data analyst", "business intelligence"],
    "engineering": ["engineer", "developer", "software", "backend", "frontend", "fullstack", "sde", "software engineer", "web developer", "mobile developer", "dev"],
    "marketing": ["marketing", "seo", "content", "social", "growth", "campaign", "paid search", "ppc", "email marketing"],
    "product": ["product", "pm", "product manager", "product owner"],
    "design": ["designer", "ux", "ui", "graphic", "product designer", "ux/ui", "visual design", "interaction design"],
    "sales": ["sales", "account executive", "bd", "business development", "quota"],
    "customer_success": ["customer success", "customer", "cs", "support", "onboarding"],
    "operations": ["operations", "ops", "supply chain", "logistics", "operations manager"],
    "finance": ["finance", "accounting", "accountant", "cfo", "financial"],
    "hr": ["hr", "human resources", "recruiter", "talent"],
    "devops": ["devops", "site reliability", "sre", "ci/cd", "infrastructure", "cloud engineer"],
    "security": ["security", "infosec", "vulnerability", "threat"],
    "qa": ["qa", "test", "quality assurance", "tester", "automation"],
    "research": ["research", "scientist", "r&d", "researcher"],
    "education": ["teacher", "instructor", "education", "trainer"],
    "healthcare": ["nurse", "doctor", "clinician", "healthcare", "medical"],
    "legal": ["legal", "law", "attorney", "counsel"],
    "management": ["manager", "director", "lead", "head of"],
    "executive": ["ceo", "cto", "cfo", "chief"],
    "medicine": ["medicine", "physician", "doctor", "md", "rn", "clinical"],
    "pharma": ["pharmaceutical", "pharma", "clinical research", "regulatory"],
    "biotech": ["biotech", "biotechnology", "molecular biology", "cell culture"],
    "chemistry": ["chemistry", "chemist", "analytical chemistry", "organic chemistry"],
    "art_and_culture": ["art", "artist", "curator", "gallery", "exhibition"],
    "music": ["music", "musician", "composer", "producer"],
    "photography": ["photography", "photographer", "studio"],
    "fashion": ["fashion", "apparel", "designer", "textiles"],
    "retail": ["retail", "store", "merchandising", "inventory"],
    "hospitality": ["hospitality", "hotel", "guest services", "restaurant"],
    "real_estate": ["real estate", "property", "realtor", "leasing"],
    "construction": ["construction", "site manager", "foreman", "contractor"],
    "energy": ["energy", "oil", "gas", "renewables", "solar", "wind"],
    "mining": ["mining", "geology", "minerals"],
    "agriculture": ["agriculture", "farming", "crop", "horticulture"],
    "environmental": ["environment", "sustainability", "ecology"],
    "sports": ["sports", "coach", "trainer", "fitness"],
    "media": ["media", "journalism", "editor", "reporter"],
    "nonprofit": ["nonprofit", "ngo", "fundraising", "grant"],
}

FIELD_PROFILES = {
    "data": {
        "weights": {"skill": 0.68, "title": 0.18, "desc": 0.14},
        "priority_skills": [
            "python", "sql", "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
            "spark", "hive", "hadoop", "airflow", "dbt", "bigquery", "redshift", "snowflake",
            "etl", "data pipeline", "data engineering", "analytics", "bi", "tableau", "powerbi",
            "looker", "metabase", "data modeling", "dimensional modeling", "statistics", "r"
        ],
        "synonyms": {"ml": "machine learning", "ds": "data scientist", "bi": "business intelligence", "pd": "product data"},
    },
    "engineering": {
        "weights": {"skill": 0.57, "title": 0.25, "desc": 0.18},
        "priority_skills": [
            "java", "javascript", "typescript", "react", "reactjs", "angular", "vue", "node", "nodejs",
            "express", "spring", "dotnet", "c#", "c++", "golang", "go", "php", "ruby",
            "docker", "kubernetes", "microservices", "rest", "graphql", "api", "ci/cd", "git"
        ],
        "synonyms": {"js": "javascript", "ts": "typescript", "golang": "go", "nodejs": "node", "sde": "software engineer"},
    },
    "marketing": {
        "weights": {"skill": 0.45, "title": 0.35, "desc": 0.2},
        "priority_skills": [
            "seo", "sem", "ppc", "google ads", "facebook ads", "content", "content marketing", "copywriting",
            "social media", "analytics", "ga", "google analytics", "email marketing", "crm", "hubspot",
            "marketing automation", "growth", "a/b testing", "funnel optimization"
        ],
        "synonyms": {"ga": "google analytics", "ppc": "paid search", "smm": "social media marketing"},
    },
    "product": {
        "weights": {"skill": 0.66, "title": 0.2, "desc": 0.14},
        "priority_skills": [
            "product management", "product manager", "roadmap", "stakeholder", "metrics", "okrs",
            "user research", "ux research", "analytics", "sql", "a/b testing", "feature prioritization",
            "jira", "confluence", "agile", "kanban"
        ],
        "synonyms": {"pm": "product manager", "po": "product owner"},
    },
    "design": {
        "weights": {"skill": 0.55, "title": 0.3, "desc": 0.15},
        "priority_skills": [
            "ux", "ui", "ux design", "ui design", "product design", "interaction design", "visual design",
            "figma", "sketch", "adobe xd", "photoshop", "illustrator", "prototyping", "wireframing",
            "usability testing", "accessibility", "design systems"
        ],
        "synonyms": {"ux/ui": "ux ui", "ux designer": "product designer"},
    },
    "sales": {
        "weights": {"skill": 0.58, "title": 0.25, "desc": 0.17},
        "priority_skills": [
            "sales", "account executive", "ae", "business development", "bd", "enterprise sales", "inside sales",
            "crm", "salesforce", "quota", "pipeline", "prospecting", "closing", "negotiation", "lead generation"
        ],
        "synonyms": {"bd": "business development", "ae": "account executive", "bizdev": "business development"},
    },
    "customer_success": {
        "weights": {"skill": 0.6, "title": 0.25, "desc": 0.15},
        "priority_skills": [
            "customer success", "cs", "onboarding", "implementation", "support", "account management",
            "renewals", "churn", "customer engagement", "helpdesk", "intercom", "zendesk"
        ],
        "synonyms": {"cs": "customer success", "acct mgmt": "account management"},
    },
    "operations": {
        "weights": {"skill": 0.5, "title": 0.35, "desc": 0.15},
        "priority_skills": ["operations", "ops", "supply chain", "logistics", "procurement", "vendor management", "warehouse", "inventory"],
        "synonyms": {"ops": "operations", "supply": "supply chain"},
    },
    "devops": {
        "weights": {"skill": 0.66, "title": 0.2, "desc": 0.14},
        "priority_skills": [
            "devops", "sre", "site reliability", "ci/cd", "jenkins", "github actions", "gitlab ci",
            "docker", "kubernetes", "helm", "terraform", "ansible", "prometheus", "grafana", "cloudwatch"
        ],
        "synonyms": {"sre": "site reliability engineer", "k8s": "kubernetes"},
    },
    "security": {
        "weights": {"skill": 0.62, "title": 0.24, "desc": 0.14},
        "priority_skills": ["information security", "infosec", "security", "pentest", "vulnerability assessment", "sast", "dast", "ossec", "siem", "aws security"],
        "synonyms": {"infosec": "information security", "pentester": "penetration tester"},
    },
    "qa": {
        "weights": {"skill": 0.52, "title": 0.3, "desc": 0.18},
        "priority_skills": ["qa", "quality assurance", "automation", "selenium", "cypress", "pytest", "test plan", "load testing"],
        "synonyms": {"qe": "quality engineer", "sdet": "software development engineer in test"},
    },
    "finance": {
        "weights": {"skill": 0.56, "title": 0.28, "desc": 0.16},
        "priority_skills": [
            "finance", "accounting", "financial analysis", "cpa", "excel", "powerbi", "tableau", "forecasting",
            "reporting", "reconciliation", "tax", "audit", "sql", "erp", "sap"
        ],
        "synonyms": {"fa": "financial analyst", "fp&a": "financial planning and analysis"},
    },
    "hr": {
        "weights": {"skill": 0.38, "title": 0.42, "desc": 0.2},
        "priority_skills": ["hr", "human resources", "recruitment", "talent acquisition", "onboarding", "people operations", "compensation", "benefits"],
        "synonyms": {"ta": "talent acquisition", "phr": "professional in human resources"},
    },
    "research": {
        "weights": {"skill": 0.58, "title": 0.22, "desc": 0.2},
        "priority_skills": ["research", "r&d", "clinical research", "experiment design", "statistical analysis", "publications", "grant writing"],
        "synonyms": {"r and d": "r&d"},
    },
    "education": {
        "weights": {"skill": 0.44, "title": 0.36, "desc": 0.2},
        "priority_skills": ["teaching", "instruction", "curriculum", "lesson planning", "edtech", "training", "elearning"],
        "synonyms": {"instructor": "teacher", "trainer": "instructor"},
    },
    "healthcare": {
        "weights": {"skill": 0.56, "title": 0.26, "desc": 0.18},
        "priority_skills": ["nurse", "physician", "md", "rn", "clinical", "patient care", "electronic medical records", "emr", "clinical trials"],
        "synonyms": {"md": "physician", "rn": "registered nurse", "emr": "electronic medical records"},
    },
    "legal": {
        "weights": {"skill": 0.52, "title": 0.28, "desc": 0.2},
        "priority_skills": ["legal", "attorney", "counsel", "paralegal", "compliance", "regulatory", "contracts"],
        "synonyms": {"esq": "attorney", "counselor": "counsel"},
    },
    "management": {
        "weights": {"skill": 0.34, "title": 0.52, "desc": 0.14},
        "priority_skills": ["manager", "director", "head of", "people management", "team lead", "operations management"],
        "synonyms": {"mgr": "manager", "lead": "team lead"},
    },
    "executive": {
        "weights": {"skill": 0.22, "title": 0.6, "desc": 0.18},
        "priority_skills": ["ceo", "cto", "cfo", "chief", "vp", "executive leadership", "board"],
        "synonyms": {"vp": "vice president", "svp": "senior vice president"},
    },
    "machine_learning": {
        "weights": {"skill": 0.72, "title": 0.14, "desc": 0.14},
        "priority_skills": [
            "machine learning", "ml", "deep learning", "neural networks", "tensorflow", "pytorch", "keras", "scikit-learn",
            "sklearn", "xgboost", "lightgbm", "catboost", "modeling", "feature engineering", "hyperparameter tuning",
            "mlops", "model deployment", "onnx", "tensorflow serving"
        ],
        "synonyms": {"dl": "deep learning", "ai": "artificial intelligence"},
    },
    "nlp": {
        "weights": {"skill": 0.72, "title": 0.14, "desc": 0.14},
        "priority_skills": [
            "nlp", "natural language processing", "transformers", "bert", "gpt", "huggingface", "spaCy", "nltk",
            "word2vec", "glove", "sequence modeling", "text classification", "ner", "topic modeling"
        ],
        "synonyms": {"nlp engineer": "natural language processing engineer", "hf": "huggingface"},
    },
    "computer_vision": {
        "weights": {"skill": 0.72, "title": 0.14, "desc": 0.14},
        "priority_skills": ["computer vision", "cv", "opencv", "image processing", "object detection", "segmentation", "yolo", "resnet", "mask r-cnn", "pytorch", "tensorflow"],
        "synonyms": {"cv": "computer vision", "vision": "computer vision"},
    },
    "robotics": {
        "weights": {"skill": 0.64, "title": 0.18, "desc": 0.18},
        "priority_skills": ["robotics", "ros", "robot operating system", "slam", "perception", "control systems", "kinematics", "path planning"],
        "synonyms": {"ros": "robot operating system"},
    },
    "embedded": {
        "weights": {"skill": 0.66, "title": 0.18, "desc": 0.16},
        "priority_skills": ["embedded", "firmware", "c", "c++", "arm", "microcontroller", "rtos", "bare metal", "mcu", "i2c", "spi"],
        "synonyms": {"mcu": "microcontroller unit"},
    },
    "firmware": {
        "weights": {"skill": 0.66, "title": 0.18, "desc": 0.16},
        "priority_skills": ["firmware", "embedded", "device drivers", "bootloader", "c", "c++", "hardware integration"],
        "synonyms": {"fw": "firmware"},
    },
    "mobile": {
        "weights": {"skill": 0.62, "title": 0.24, "desc": 0.14},
        "priority_skills": ["mobile", "ios", "android", "react native", "flutter", "swift", "kotlin", "objective-c", "xcode", "android studio"],
        "synonyms": {"rn": "react native", "cordova": "phonegap"},
    },
    "ios": {
        "weights": {"skill": 0.66, "title": 0.24, "desc": 0.1},
        "priority_skills": ["ios", "swift", "objective-c", "xcode", "cocoa touch", "cocoapods", "swiftui"],
        "synonyms": {"objc": "objective-c", "swiftui": "swiftui"},
    },
    "android": {
        "weights": {"skill": 0.66, "title": 0.24, "desc": 0.1},
        "priority_skills": ["android", "kotlin", "java", "gradle", "android studio", "jetpack", "compose"],
        "synonyms": {"apk": "android package", "aosp": "android open source project"},
    },
    "game_dev": {
        "weights": {"skill": 0.62, "title": 0.24, "desc": 0.14},
        "priority_skills": ["game development", "unity", "unreal", "c#", "c++", "graphics", "rendering", "shader", "3d"],
        "synonyms": {"unity3d": "unity", "udk": "unreal"},
    },
    "blockchain": {
        "weights": {"skill": 0.62, "title": 0.24, "desc": 0.14},
        "priority_skills": ["blockchain", "solidity", "smart contracts", "ethereum", "web3", "dapp", "truffle", "hardhat"],
        "synonyms": {"web3": "web 3.0", "eth": "ethereum"},
    },
    "crypto": {
        "weights": {"skill": 0.58, "title": 0.26, "desc": 0.16},
        "priority_skills": ["crypto", "defi", "tokenomics", "solidity", "smart contracts"],
        "synonyms": {"defi": "decentralized finance", "nft": "non-fungible token"},
    },
    "solutions_architect": {
        "weights": {"skill": 0.52, "title": 0.34, "desc": 0.14},
        "priority_skills": ["solutions architect", "architecture", "system design", "cloud architecture", "aws", "azure", "gcp", "scalability"],
        "synonyms": {"sa": "solutions architect"},
    },
    "cloud": {
        "weights": {"skill": 0.62, "title": 0.24, "desc": 0.14},
        "priority_skills": ["cloud", "aws", "azure", "gcp", "serverless", "lambda", "cloudformation", "terraform"],
        "synonyms": {"iac": "infrastructure as code", "cf": "cloudformation"},
    },
    "network": {
        "weights": {"skill": 0.56, "title": 0.28, "desc": 0.16},
        "priority_skills": ["network", "network engineer", "routing", "switching", "tcp/ip", "cisco", "juniper"],
        "synonyms": {"lan": "local area network", "wan": "wide area network"},
    },
    "dba": {
        "weights": {"skill": 0.62, "title": 0.24, "desc": 0.14},
        "priority_skills": ["dba", "database administrator", "oracle", "mysql", "postgres", "postgresql", "sql server", "performance tuning", "indexing", "replication"],
        "synonyms": {"db": "database", "psql": "postgresql"},
    },
    "sysadmin": {
        "weights": {"skill": 0.56, "title": 0.28, "desc": 0.16},
        "priority_skills": ["system administrator", "linux", "windows server", "bash", "powershell", "monitoring", "ansible"],
        "synonyms": {"sysadmin": "system administrator", "sre": "site reliability engineer"},
    },
    "business_analyst": {
        "weights": {"skill": 0.52, "title": 0.32, "desc": 0.16},
        "priority_skills": ["business analyst", "ba", "requirements gathering", "process mapping", "stakeholder management", "sql", "excel", "powerbi"],
        "synonyms": {"ba": "business analyst"},
    },
    "project_manager": {
        "weights": {"skill": 0.42, "title": 0.44, "desc": 0.14},
        "priority_skills": ["project manager", "pmp", "scrum master", "agile", "kanban", "waterfall", "stakeholder management"],
        "synonyms": {"pm": "project manager", "scrum": "scrum master"},
    },
    "technical_writer": {
        "weights": {"skill": 0.44, "title": 0.34, "desc": 0.22},
        "priority_skills": ["technical writer", "documentation", "api docs", "openapi", "rest docs", "markdown", "md", "style guide"],
        "synonyms": {"doc": "documentation"},
    },
    "growth_marketing": {
        "weights": {"skill": 0.46, "title": 0.34, "desc": 0.2},
        "priority_skills": ["growth", "growth marketing", "growth hacking", "seo", "paid acquisition", "analytics", "a/b testing", "funnel"],
        "synonyms": {"ga": "google analytics", "smm": "social media marketing"},
    },
    "general": {
        "weights": {"skill": 0.5, "title": 0.3, "desc": 0.2},
        "priority_skills": [],
        "synonyms": {"misc": "general"},
    },
    "medicine": {
        "weights": {"skill": 0.6, "title": 0.25, "desc": 0.15},
        "priority_skills": ["medicine", "physician", "doctor", "md", "rn", "clinical", "internal medicine", "surgery", "residency", "clinical trials", "patient care", "emergency medicine"],
        "synonyms": {"md": "physician", "dr": "doctor"},
    },
    "pharma": {
        "weights": {"skill": 0.6, "title": 0.25, "desc": 0.15},
        "priority_skills": ["pharmaceutical", "pharma", "clinical research", "clinical trials", "regulatory", "drug development", "cGMP", "quality assurance", "qc", "regulatory affairs", "manufacturing associate", "validation", "quality control", "regulatory compliance", "drug safety"],
        "synonyms": {"cgmp": "cGMP"},
    },
    "biotech": {
        "weights": {"skill": 0.66, "title": 0.2, "desc": 0.14},
        "priority_skills": ["biotech", "biotechnology", "molecular biology", "cell culture", "assay development", "wet lab", "pipetting", "protein expression", "cloning", "flow cytometry"],
        "synonyms": {"wetlab": "wet lab"},
    },
    "chemistry": {
        "weights": {"skill": 0.62, "title": 0.24, "desc": 0.14},
        "priority_skills": ["chemistry", "chemist", "analytical chemistry", "organic chemistry", "synthetic chemistry", "hplc", "gc-ms", "spectroscopy"],
        "synonyms": {"hplc": "hplc", "gcms": "gc-ms"},
    },
    "art_and_culture": {
        "weights": {"skill": 0.46, "title": 0.36, "desc": 0.18},
        "priority_skills": ["artist", "curator", "gallery", "exhibition", "fine arts", "painting", "sculpture", "studio practice"],
        "synonyms": {"arts": "art"},
    },
    "music": {
        "weights": {"skill": 0.46, "title": 0.36, "desc": 0.18},
        "priority_skills": ["music", "musician", "composer", "producer", "audio engineering", "mixing", "mastering", "sound design"],
        "synonyms": {"edm": "electronic dance music"},
    },
    "photography": {
        "weights": {"skill": 0.46, "title": 0.36, "desc": 0.18},
        "priority_skills": ["photography", "photographer", "portrait", "commercial photography", "retouching", "studio", "lighting"],
        "synonyms": {"retouch": "retouching"},
    },
    "fashion": {
        "weights": {"skill": 0.46, "title": 0.36, "desc": 0.18},
        "priority_skills": ["fashion", "apparel", "designer", "textiles", "pattern making", "merchandising", "product development"],
        "synonyms": {"apparel": "apparel"},
    },
    "retail": {
        "weights": {"skill": 0.52, "title": 0.3, "desc": 0.18},
        "priority_skills": ["retail", "store manager", "merchandising", "visual merchandising", "inventory", "omnichannel", "ecommerce", "point of sale"],
        "synonyms": {"pos": "point of sale"},
    },
    "hospitality": {
        "weights": {"skill": 0.48, "title": 0.38, "desc": 0.14},
        "priority_skills": ["hospitality", "hotel", "guest services", "reception", "front desk", "food and beverage", "restaurant", "banquets"],
        "synonyms": {"f&b": "food and beverage"},
    },
    "real_estate": {
        "weights": {"skill": 0.52, "title": 0.33, "desc": 0.15},
        "priority_skills": ["real estate", "realtor", "leasing", "property management", "sales", "broker", "asset management"],
        "synonyms": {"prop mgmt": "property management"},
    },
    "construction": {
        "weights": {"skill": 0.56, "title": 0.3, "desc": 0.14},
        "priority_skills": ["construction", "site manager", "foreman", "estimating", "project management", "safety", "osha"],
        "synonyms": {"gc": "general contractor"},
    },
    "energy": {
        "weights": {"skill": 0.56, "title": 0.3, "desc": 0.14},
        "priority_skills": ["energy", "oil and gas", "renewables", "solar", "wind", "power systems", "grid", "transmission"],
        "synonyms": {"o&g": "oil and gas"},
    },
    "mining": {
        "weights": {"skill": 0.52, "title": 0.34, "desc": 0.14},
        "priority_skills": ["mining", "geology", "underground mining", "surface mining", "mine planning", "drilling"],
        "synonyms": {"geo": "geology"},
    },
    "agriculture": {
        "weights": {"skill": 0.48, "title": 0.36, "desc": 0.16},
        "priority_skills": ["agriculture", "farming", "crop management", "horticulture", "agronomy", "farm operations"],
        "synonyms": {"agri": "agriculture"},
    },
    "environmental": {
        "weights": {"skill": 0.52, "title": 0.3, "desc": 0.18},
        "priority_skills": ["environment", "sustainability", "conservation", "ecology", "epa", "environmental management"],
        "synonyms": {"env": "environment"},
    },
    "hospitality_and_travel": {
        "weights": {"skill": 0.46, "title": 0.38, "desc": 0.16},
        "priority_skills": ["travel", "tourism", "hospitality", "airline", "flight attendant", "guest services", "concierge"],
        "synonyms": {"ft": "flight attendant"},
    },
    "sports": {
        "weights": {"skill": 0.46, "title": 0.36, "desc": 0.18},
        "priority_skills": ["sports", "coach", "trainer", "fitness", "strength and conditioning", "sports management", "scouting"],
        "synonyms": {"s&c": "strength and conditioning"},
    },
    "media": {
        "weights": {"skill": 0.52, "title": 0.32, "desc": 0.16},
        "priority_skills": ["media", "journalism", "editor", "reporter", "broadcast", "film", "production", "videography", "editing"],
        "synonyms": {"vid": "video", "prd": "production"},
    },
    "journalism": {
        "weights": {"skill": 0.52, "title": 0.32, "desc": 0.16},
        "priority_skills": ["journalism", "reporter", "editor", "investigative", "news", "copyediting", "fact checking"],
        "synonyms": {"op-ed": "opinion editorial"},
    },
    "nonprofit": {
        "weights": {"skill": 0.46, "title": 0.4, "desc": 0.14},
        "priority_skills": ["nonprofit", "ngo", "fundraising", "grant writing", "program management", "volunteer coordination"],
        "synonyms": {"ngo": "nonprofit"},
    },
}


def to_dict() -> Dict[str, Any]:
    # Provide a canonical base skills list for resume extraction + matching
    base_skills = list(sorted(set().union(*[p.get("priority_skills", []) for p in FIELD_PROFILES.values()])))
    return {"field_keywords": _FIELD_KEYWORDS, "profiles": FIELD_PROFILES, "base_skills": base_skills}


def export_to(path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "backend/field_profiles.json"
    export_to(out)
    print(f"Exported field profiles to {out}")
