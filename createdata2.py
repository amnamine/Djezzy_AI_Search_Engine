import json
import pandas as pd
import re
import random
import math

# --- CONFIGURATION ---
JSON_FILES = ['scraping1.json', 'scraping2.json', 'scraping3.json', 'scraping4.json']
OUTPUT_CSV = 'dataset_train2.csv'
TARGET_ROWS = 12000

# ==========================================
# 1. THE "GOLD STANDARD" CATALOG (Expanded)
# ==========================================
# We split the catalog into specific categories to ensure queries match products.

base_catalog = [
    # --- CATEGORY: PHONES (SMARTPHONES) ---
    {"id": "PH_S24", "name": "Samsung Galaxy S24", "cat": "Smartphone", "price": 180000, "desc": "Smartphone Android Samsung AI, 256Go", "tags": ["galaxy", "samsung", "android", "s24"]},
    {"id": "PH_A54", "name": "Samsung Galaxy A54", "cat": "Smartphone", "price": 70000, "desc": "Milieu de gamme excellent photo 5G", "tags": ["galaxy", "a54", "samsung"]},
    {"id": "PH_REDMI13", "name": "Xiaomi Redmi Note 13", "cat": "Smartphone", "price": 45000, "desc": "Xiaomi rapport qualité prix 128Go", "tags": ["xiaomi", "redmi", "note", "13"]},
    {"id": "PH_IPHONE15", "name": "Apple iPhone 15", "cat": "Smartphone", "price": 220000, "desc": "Dernier iPhone Apple iOS Dynamic Island", "tags": ["iphone", "apple", "ios", "15"]},
    {"id": "PH_OPPO_RENO", "name": "Oppo Reno 10", "cat": "Smartphone", "price": 85000, "desc": "Portrait Expert 5G Charge rapide", "tags": ["oppo", "reno", "charge"]},

    # --- CATEGORY: ROUTERS & BOX (HARDWARE) ---
    {"id": "HW_MODEM", "name": "Modem 4G Djezzy", "cat": "Router", "price": 6000, "desc": "Routeur Wifi sans fil haut débit pour la maison", "tags": ["modem", "wifi", "4g", "maison"]},
    {"id": "HW_TWIN", "name": "Djezzy TwinBox", "cat": "Router", "price": 12900, "desc": "Box 4G + Android TV + Abonnement TOD", "tags": ["tv", "box", "android", "tod"]},
    {"id": "HW_POCKET", "name": "Pocket Wifi", "cat": "Router", "price": 4500, "desc": "Modem portable petite taille", "tags": ["pocket", "wifi", "deplacement"]},

    # --- CATEGORY: MOBILE OFFERS (SIM) ---
    {"id": "LEG_2000", "name": "Djezzy Legend 2000", "cat": "Offer_Mobile", "price": 2000, "desc": "Appels illimités + 70Go internet + 5000DA crédit", "tags": ["illimité", "légende", "2000"]},
    {"id": "LEG_2500", "name": "Djezzy Legend 2500", "cat": "Offer_Mobile", "price": 2500, "desc": "Tout illimité vers Djezzy + 300Go + International", "tags": ["vip", "business", "monde"]},
    {"id": "HAY_500", "name": "Hayla Bezzef 500", "cat": "Offer_Mobile", "price": 500, "desc": "Offre hebdomadaire, appels et internet", "tags": ["semaine", "étudiant", "haïla"]},

    # --- CATEGORY: INTERNET OFFERS ---
    {"id": "INT_JOUR", "name": "Pack Internet Jour", "cat": "Offer_Internet", "price": 100, "desc": "Connexion journalière 1Go", "tags": ["24h", "dépanage", "100da"]},
    {"id": "INT_MOIS", "name": "Pack Internet 60Go", "cat": "Offer_Internet", "price": 2000, "desc": "Forfait internet mensuel grand volume", "tags": ["mois", "wifi", "data"]},
]

# ==========================================
# 2. INTELLIGENT QUERY GENERATION TOOLS
# ==========================================

def clean_text(text):
    """Cleans scraped text artifacts."""
    if not isinstance(text, str): return ""
    text = re.sub(r'(\d+)&nbsp(\d+)', r'\1\2', text) # Fix prices like 49&nbsp900
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_typo(text):
    """
    Simulates REALISTIC typing errors:
    - Skipping a char (samsung -> samsng)
    - Swapping chars (galaxy -> galxay)
    - Wrong key (iphone -> iphonr)
    """
    if len(text) < 4: return text
    
    # Don't mess up numbers too much (prices)
    if text.isdigit(): return text

    r = random.random()
    if r < 0.4: # Delete char (Most common mobile typo)
        idx = random.randint(0, len(text)-1)
        return text[:idx] + text[idx+1:]
    elif r < 0.7: # Swap chars
        idx = random.randint(0, len(text)-2)
        return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
    else: # Replace char
        idx = random.randint(0, len(text)-1)
        return text[:idx] + random.choice('azertyuiop') + text[idx+1:]

def get_category_synonyms(cat_type, product_name=""):
    """
    Returns specific keywords strictly related to the category.
    This prevents searching 'internet' when we want a 'phone'.
    """
    syns = []
    
    # 1. SMARTPHONES
    if cat_type == "Smartphone":
        syns = ["telephone", "mobile", "portable", "hètf", "هاتف", "smartphone", "cellulaire"]
        if "samsung" in product_name.lower(): syns += ["sam", "galaxi", "android"]
        if "iphone" in product_name.lower(): syns += ["apple", "ios", "ifone"]
        if "redmi" in product_name.lower() or "xiaomi" in product_name.lower(): syns += ["mi", "redmi", "chinois"]

    # 2. ROUTERS / BOX
    elif cat_type == "Router":
        syns = ["wifi", "modem", "box", "routeur", "signal", "dar", "maison", "internet maison"]

    # 3. MOBILE OFFERS (Legend, Hayla)
    elif cat_type == "Offer_Mobile":
        syns = ["puce", "sim", "promo", "offre", "abonnement", "line", "lign"]
        if "legend" in product_name.lower(): syns += ["gold", "premium", "business"]

    # 4. INTERNET OFFERS (Data only)
    elif cat_type == "Offer_Internet":
        syns = ["net", "connexion", "data", "4g", "gigaille", "recharge net"]

    return syns

# ==========================================
# 3. LOAD EXTRA DATA FROM JSONS
# ==========================================
def load_scraped_data():
    print("[INFO] Loading scraped files...")
    scraped_products = []
    
    for jf in JSON_FILES:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle List of items (scraping4.json style)
                if isinstance(data, list):
                    for item in data:
                        # Try to detect what this item is
                        if isinstance(item, dict):
                            name = clean_text(item.get('title') or item.get('name') or item.get('product_name'))
                            price_raw = str(item.get('price', '0'))
                            # Extract price digits
                            price_match = re.search(r'(\d[\d\s]*)', price_raw)
                            price = int(price_match.group(1).replace(" ", "")) if price_match else 0
                            
                            if not name: continue

                            # AUTO-CATEGORIZATION
                            cat = "Offer_Mobile" # Default
                            if any(x in name.lower() for x in ['samsung', 'xiaomi', 'redmi', 'oppo', 'iphone', 'infinix', 'realme', 'zte']):
                                cat = "Smartphone"
                            elif any(x in name.lower() for x in ['modem', 'box', 'wifi', 'router']):
                                cat = "Router"
                            elif any(x in name.lower() for x in ['internet', 'go', 'data']):
                                cat = "Offer_Internet"

                            scraped_products.append({
                                "id": f"SCRAP_{random.randint(10000,99999)}",
                                "name": name,
                                "cat": cat,
                                "price": price,
                                "desc": clean_text(item.get('description', name)),
                                "tags": name.lower().split()
                            })

                # Handle Dict style (scraping1/2/3 style)
                elif isinstance(data, dict):
                     # (Simplified extraction for single dicts if needed)
                     pass
                     
        except FileNotFoundError:
            print(f"[WARN] File {jf} not found (Skipping).")
        except Exception as e:
            pass 

    print(f"[INFO] Loaded {len(scraped_products)} extra products from JSONs.")
    return scraped_products

# ==========================================
# 4. MAIN GENERATION LOOP
# ==========================================
full_catalog = base_catalog + load_scraped_data()
data_rows = []

print(f"[INFO] Total Catalog Size: {len(full_catalog)} products.")
print("[INFO] Generating 10,000+ Synthetic Rows with Context-Aware Logic...")

while len(data_rows) < TARGET_ROWS:
    # Pick a product
    product = random.choice(full_catalog)
    
    # Decide Query Strategy
    strategy = random.choice(['exact', 'general_cat', 'price_search', 'spec_search', 'typo_heavy'])
    
    query = ""
    
    # --- 1. Exact / Name Search ---
    if strategy == 'exact':
        query = product['name'].lower()
    
    # --- 2. Category / Synonym Search (CONTEXT AWARE) ---
    elif strategy == 'general_cat':
        # Get synonyms specific to THIS product's category
        cat_syns = get_category_synonyms(product['cat'], product['name'])
        if cat_syns:
            query = random.choice(cat_syns)
            # Sometimes add brand name for phones
            if product['cat'] == "Smartphone":
                brand = next((t for t in product['tags'] if t in ['samsung', 'xiaomi', 'oppo', 'apple']), "")
                if brand and random.random() > 0.5:
                    query += f" {brand}"
        else:
            query = product['name'].lower()

    # --- 3. Price Search ---
    elif strategy == 'price_search':
        # e.g. "telephone 50000 da" or "internet 2000"
        base_word = random.choice(get_category_synonyms(product['cat'], product['name']) or [product['cat']])
        if product['price'] > 0:
            query = f"{base_word} {product['price']}"
            if random.random() > 0.5: query += " da"
        else:
            query = f"{base_word} gratuit"

    # --- 4. Specific Spec Search (Tags) ---
    elif strategy == 'spec_search':
        # e.g. "redmi 128go" or "modem 4g"
        relevant_tags = [t for t in product['tags'] if len(t) > 2]
        if relevant_tags:
            t1 = random.choice(relevant_tags)
            # Maybe add a synonym
            syn = random.choice(get_category_synonyms(product['cat'], product['name']) or [""])
            query = f"{syn} {t1}".strip()
        else:
            query = product['name'].lower()

    # --- 5. Heavy Typo (Raw User Input) ---
    else:
        query = product['name'].lower() 
        # We will apply heavy typos below

    # --- APPLY NOISE (TYPOS) ---
    # The rule: Typos must be based on the CURRENT query, which is already context-aware.
    if strategy == 'typo_heavy' or random.random() < 0.3:
        # Apply typo generation 1 or 2 times
        query = generate_typo(query)
        if random.random() < 0.3:
            query = generate_typo(query)

    # --- GENERATE ROW ---
    # 90% Positive Samples, 10% Negative
    if random.random() > 0.1:
        data_rows.append({
            "product_id": product['id'],
            "product_name": product['name'],
            "category": product['cat'],
            "description": product['desc'],
            "price": product['price'],
            "user_query": clean_text(query),
            "relevance_label": 1
        })
    else:
        # Negative Sample: User asks for Phone, we show Internet? -> Label 0
        # Pick a random DIFFERENT product
        wrong = random.choice(full_catalog)
        # Ensure it's actually different
        while wrong['id'] == product['id']:
            wrong = random.choice(full_catalog)
            
        data_rows.append({
            "product_id": wrong['id'],
            "product_name": wrong['name'],
            "category": wrong['cat'],
            "description": wrong['desc'],
            "price": wrong['price'],
            "user_query": clean_text(query), # The query was for the Original product
            "relevance_label": 0
        })

# ==========================================
# 5. SAVE & PREVIEW
# ==========================================
df = pd.DataFrame(data_rows)
df = df.sample(frac=1).reset_index(drop=True)

# Export
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"[SUCCESS] Saved '{OUTPUT_CSV}' with {len(df)} rows.")

# Safe Preview
try:
    print(df[['product_name', 'category', 'user_query', 'relevance_label']].head(10))
except:
    print("[INFO] Done. Check CSV file.")