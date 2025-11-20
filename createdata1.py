import json
import pandas as pd
import re
import random
import sys

# --- CONFIGURATION ---
JSON_FILES = ['scraping1.json', 'scraping2.json', 'scraping3.json']
OUTPUT_CSV = 'dataset_train1.csv'
TARGET_ROWS = 12000  

# --- 1. THE "GOLD STANDARD" CATALOG ---
base_catalog = [
    # -- OFFRES MOBILES --
    {"id": "LEG_2000", "name": "Djezzy Legend 2000", "cat": "Mobile", "price": 2000, "desc": "Appels illimités + 70Go internet + 5000DA crédit", "tags": ["illimité", "légende", "2000da"]},
    {"id": "LEG_2500", "name": "Djezzy Legend 2500", "cat": "Mobile", "price": 2500, "desc": "Tout illimité vers Djezzy + 300Go + International", "tags": ["vip", "business", "monde"]},
    {"id": "HAY_500", "name": "Hayla Bezzef 500", "cat": "Mobile", "price": 500, "desc": "Offre hebdomadaire, appels et internet", "tags": ["semaine", "étudiant", "haïla"]},
    {"id": "HAY_1500", "name": "Hayla Bezzef 1500", "cat": "Mobile", "price": 1500, "desc": "Offre mensuelle hybride data et voix", "tags": ["mois", "hbal", "bezef"]},
    
    # -- INTERNET --
    {"id": "INT_JOUR", "name": "Pack Internet Jour", "cat": "Internet", "price": 100, "desc": "Connexion journalière 1Go", "tags": ["24h", "dépanage", "100da"]},
    {"id": "INT_MOIS", "name": "Pack Internet 60Go", "cat": "Internet", "price": 2000, "desc": "Forfait internet mensuel grand volume", "tags": ["mois", "wifi", "data"]},
    
    # -- HARDWARE & BOX --
    {"id": "BOX_TWIN", "name": "Djezzy TwinBox", "cat": "Maison", "price": 12900, "desc": "Box 4G + Android TV + Abonnement TOD", "tags": ["télé", "maison", "famille", "routeur"]},
    {"id": "MODEM_4G", "name": "Modem Djezzy 4G", "cat": "Maison", "price": 6000, "desc": "Routeur Wifi sans fil haut débit", "tags": ["partage", "pc", "bureau"]},
    
    # -- SERVICES --
    {"id": "SRV_FLEXY", "name": "Flexy / Recharge", "cat": "Service", "price": 100, "desc": "Rechargement de crédit instantané", "tags": ["crédit", "envoyer", "solde"]},
    {"id": "SRV_ANG", "name": "Anghami Plus", "cat": "Service", "price": 0, "desc": "Streaming musical illimité sans pub", "tags": ["musique", "chanson", "mp3"]}
]

# --- 2. HELPER FUNCTIONS ---

def clean_text(text):
    """Removes HTML artifacts and extra spaces."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_typo(text):
    """Simulates human typing errors."""
    if len(text) < 4: return text
    r = random.random()
    if r < 0.3: # Remove char
        idx = random.randint(0, len(text)-1)
        return text[:idx] + text[idx+1:]
    elif r < 0.6: # Swap chars
        idx = random.randint(0, len(text)-2)
        return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
    else: # Replace char
        idx = random.randint(0, len(text)-1)
        return text[:idx] + random.choice('azertyuiop') + text[idx+1:]

def get_synonyms(word):
    """Returns dialect/synonyms for a given keyword."""
    synonyms = {
        "internet": ["net", "connexion", "4g", "data", "wifi", "إنترنت"],
        "mobile": ["puce", "sim", "telephone", "hètf", "هاتف"],
        "legend": ["lejend", "legende", "gold", "premium"],
        "box": ["modem", "routeur", "dar", "maison"],
        "flexy": ["verser", "charger", "recharge", "فليكسي", "storm"],
        "illimité": ["batel", "gratuit", "infinity"],
        "djezzy": ["jazy", "jezzy", "جازي"]
    }
    return synonyms.get(word.lower(), [])

# --- 3. EXTRACTION FROM JSON FILES ---
extracted_texts = []

def scan_json_content(data):
    """Recursively searches for 'text' fields in complex JSONs."""
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ['text', 'content', 'description'] and isinstance(v, str):
                extracted_texts.append(v)
            else:
                scan_json_content(v)
    elif isinstance(data, list):
        for item in data:
            scan_json_content(item)

print("[INFO] Scanning JSON files for context...")
for jf in JSON_FILES:
    try:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            scan_json_content(data)
    except Exception as e:
        print(f"[WARN] Could not process {jf}: {e}")

print(f"[INFO] Extracted {len(extracted_texts)} text snippets.")

# --- 4. GENERATOR ENGINE ---
data_rows = []

print("[INFO] Generating synthetic AI training data...")

while len(data_rows) < TARGET_ROWS:
    product = random.choice(base_catalog)
    
    # --- Strategy A: Construct a Query ---
    query_type = random.choice(['exact', 'price', 'synonym', 'intent', 'arabic'])
    
    base_query = ""
    
    if query_type == 'exact':
        base_query = product['name'].lower()
        
    elif query_type == 'price':
        if product['price'] > 0:
            base_query = f"{product['cat'].lower()} {product['price']} da"
        else:
            base_query = f"{product['name'].lower()} gratuit"
            
    elif query_type == 'synonym':
        words = product['name'].lower().split()
        new_words = []
        for w in words:
            syns = get_synonyms(w)
            if syns:
                new_words.append(random.choice(syns))
            else:
                new_words.append(w)
        base_query = " ".join(new_words)
        
    elif query_type == 'intent':
        if "tv" in str(product.get('tags')):
            base_query = "regarder la télé"
        elif "illimité" in str(product.get('tags')):
            base_query = "appels gratuits tout le temps"
        elif "internet" in product['cat'].lower():
            base_query = "connexion rapide"
        else:
            base_query = f"acheter {product['name']}"
            
    elif query_type == 'arabic':
        if "internet" in product['cat'].lower():
            base_query = f"إنترنت {product['price']}"
        elif "legend" in product['name'].lower():
            base_query = "عروض ليجند"
        elif "flexy" in product['name'].lower():
            base_query = "تعبئة رصيد"
        else:
            base_query = product['name']

    # --- Strategy B: Add Noise (Typos) ---
    if random.random() < 0.3:
        base_query = generate_typo(base_query)

    # --- Strategy C: Labeling ---
    if random.random() > 0.1:
        data_rows.append({
            "product_id": product['id'],
            "product_name": product['name'],
            "category": product['cat'],
            "description": product['desc'],
            "price": product['price'],
            "user_query": clean_text(base_query),
            "relevance_label": 1
        })
    else:
        wrong_product = random.choice([p for p in base_catalog if p['id'] != product['id']])
        data_rows.append({
            "product_id": wrong_product['id'],
            "product_name": wrong_product['name'], 
            "category": wrong_product['cat'],
            "description": wrong_product['desc'],
            "price": wrong_product['price'],
            "user_query": clean_text(base_query), 
            "relevance_label": 0
        })

# --- 5. EXPORT ---
df = pd.DataFrame(data_rows)
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV (Uses utf-8-sig so Excel reads Arabic correctly)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"[SUCCESS] Generated {len(df)} rows.")
print(f"[SUCCESS] Saved to '{OUTPUT_CSV}'")

# --- SAFE PRINT BLOCK (FIX FOR WINDOWS ERRORS) ---
try:
    print("\n--- Data Preview ---")
    print(df.head(10))
except UnicodeEncodeError:
    print("[INFO] Your Windows terminal cannot display Arabic characters.")
    print("[INFO] But don't worry! The CSV file is saved correctly.")
    print(f"[INFO] Please open '{OUTPUT_CSV}' in Excel to view the data.")
except Exception as e:
    print(f"[WARN] Could not print preview: {e}")