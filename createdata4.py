import json
import csv
import random
import re
import uuid

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'scraping4.json'
OUTPUT_FILE = 'dataset_train4.csv'
TARGET_DATASET_SIZE = 10500  # Aiming for >10k

# Keywords to identify categories
CATEGORY_KEYWORDS = {
    "Smartphone": ["zte", "tecno", "oppo", "samsung", "realme", "blade", "nubia", "pova", "spark", "infinix", "galaxy", "redmi", "xiaomi", "v60", "a75", "a35"],
    "Routeur_Modem": ["d-link", "tcl", "modem", "box", "dwr", "wifi", "4g"],
    "Tablette": ["tablette", "tab", "d-tech 10", "d-tech 8", "ipad"],
    "Accessoire_Audio": ["earbuds", "ecouteur", "airpods", "casque", "kit", "bluetooth", "hoco", "revaleo"],
    "Accessoire_Charge": ["cable", "chargeur", "power bank", "usb", "type-c", "lightning", "batterie"],
    "Accessoire_Auto": ["support", "car", "voiture", "fm", "transmitter"]
}

# Synonyms and Intent words for augmentation
INTENTS_PREFIX = ["achat", "acheter", "prix", "combien coute", "chercher", "trouver", "le", "la", "les", "promo", "nouveau"]
INTENTS_SUFFIX = ["algerie", "djezzy", "pas cher", "en ligne", "livraison", "original", "2025", "promo", "solde", "magasin"]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_text(text):
    """Removes special characters and extra spaces."""
    if not text: return ""
    # Remove HTML tags if any exist
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()

def clean_price(price_str):
    """
    Fixes the price format issues.
    Converts "25&nbsp500 DA" -> "25 500 DA"
    """
    if not price_str: return "0 DA"
    
    # 1. Replace &nbsp variants with a standard space
    clean = str(price_str).replace('&nbsp;', ' ').replace('&nbsp', ' ')
    
    # 2. Replace Unicode non-breaking space (\xa0) if present
    clean = clean.replace('\xa0', ' ')
    
    # 3. Remove any other weird characters but keep numbers and letters (DA)
    # This ensures "25  500" becomes "25 500"
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    return clean

def get_category(text):
    """Determines category based on text content."""
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return cat
    return "Autre"

def generate_typo(text):
    """Randomly swaps characters or drops one to simulate typing errors."""
    if len(text) < 4: return text
    if random.random() > 0.5: # Swap
        idx = random.randint(0, len(text) - 2)
        chars = list(text)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        return "".join(chars)
    else: # Drop char
        idx = random.randint(0, len(text) - 1)
        return text[:idx] + text[idx+1:]

def augment_query(base_query):
    """Creates a variation of the query using prefixes, suffixes, or typos."""
    method = random.choice(["prefix", "suffix", "typo", "raw", "combination"])
    query = base_query.lower()
    
    if method == "typo":
        return generate_typo(query)
    elif method == "prefix":
        return f"{random.choice(INTENTS_PREFIX)} {query}"
    elif method == "suffix":
        return f"{query} {random.choice(INTENTS_SUFFIX)}"
    elif method == "combination":
        return f"{random.choice(INTENTS_PREFIX)} {query} {random.choice(INTENTS_SUFFIX)}"
    
    return query

# ==========================================
# MAIN GENERATOR
# ==========================================

def create_large_dataset():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Make sure it is in the same folder.")
        return

    # 1. Clean and Structure Data
    products = []
    print("Cleaning product data and fixing prices...")
    
    for item in raw_data:
        title = clean_text(item.get("title", ""))
        desc = clean_text(item.get("description", ""))
        
        # FIX: Apply the price cleaner here
        raw_price = item.get("price", "0 DA")
        fixed_price = clean_price(raw_price)
        
        full_name = f"{title} {desc}".strip()
        
        products.append({
            "id": str(uuid.uuid4())[:8],
            "brand": title,
            "model": desc,
            "name": full_name,
            "category": get_category(full_name),
            "price": fixed_price 
        })

    dataset_rows = []
    total_products = len(products)
    
    # We need ~100 rows per product to reach target size
    if total_products == 0:
        print("Error: No products found in JSON.")
        return

    rows_per_product = max(1, TARGET_DATASET_SIZE // total_products)
    
    print(f"Processing {total_products} products. Generating ~{rows_per_product} rows per product...")

    for prod in products:
        # === A. POSITIVE SAMPLES (User wants THIS product) ===
        n_pos = int(rows_per_product * 0.4)
        
        base_positives = [
            prod['name'],                   
            prod['model'],                  
            f"{prod['category']} {prod['brand']}", 
            prod['brand'],                  
            f"{prod['model']} {prod['price']}" 
        ]
        
        for _ in range(n_pos):
            base = random.choice(base_positives)
            query = augment_query(base)
            dataset_rows.append({
                "product_id": prod['id'],
                "product_name": prod['name'],
                "category": prod['category'],
                "description": prod['model'],
                "price": prod['price'], # Uses the fixed price
                "user_query": query,
                "relevance_label": 1 # MATCH
            })

        # === B. NEGATIVE SAMPLES (User wants SOMETHING ELSE) ===
        n_neg = rows_per_product - n_pos
        
        for _ in range(n_neg):
            other = random.choice(products)
            while other['id'] == prod['id']:
                other = random.choice(products)
            
            if other['category'] == prod['category'] and random.random() < 0.5:
                query_base = other['brand'] if random.random() < 0.5 else other['model']
            else:
                query_base = other['name']
            
            query = augment_query(query_base)
            
            dataset_rows.append({
                "product_id": prod['id'],
                "product_name": prod['name'],
                "category": prod['category'],
                "description": prod['model'],
                "price": prod['price'], # Uses the fixed price
                "user_query": query,
                "relevance_label": 0 # NO MATCH
            })

    # Shuffle final dataset
    random.shuffle(dataset_rows)

    # Write to CSV
    headers = ["product_id", "product_name", "category", "description", "price", "user_query", "relevance_label"]
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(f"Done! Generated {len(dataset_rows)} training examples in '{OUTPUT_FILE}'.")
    print(f"Sample Price Check: {dataset_rows[0]['price']}")

if __name__ == "__main__":
    create_large_dataset()