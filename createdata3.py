import json
import csv
import random
import re
import uuid

# ==========================================
# CONFIGURATION & HELPERS
# ==========================================

OUTPUT_FILE = "dataset_train3.csv"
FILES = {
    "site": "scraping1.json",
    "account": "scraping2.json",
    "offers": "scraping3.json",
    "shop": "scraping4.json"
}

# Synonyms and Keywords for "Intelligent" Query Generation
SYNONYMS = {
    "smartphone": ["mobile", "téléphone", "portable", "cellulaire", "phone", "gsm"],
    "internet": ["data", "web", "connexion", "net", "4g", "wifi", "modem"],
    "credit": ["recharge", "flexy", "solde", "unité"],
    "offer": ["forfait", "abonnement", "promo", "plan", "pack"],
    "price": ["prix", "combien", "tarif", "coût", "da", "dinar"],
    "buy": ["acheter", "achat", "commander", "je veux", "chercher"],
}

TYPO_MAPPING = {
    'a': 'q', 'e': 'r', 'i': 'o', 'o': 'p', 'u': 'y', # Keyboard proximity
    's': 'z', 'd': 's', 'f': 'd', 'g': 'f',
    'm': 'n', 'n': 'm',
    'y': 'i', 'ph': 'f'
}

def clean_price(price_str):
    """Extracts numeric value from strings like '49&nbsp900 DA'"""
    if not price_str:
        return 0
    # Remove non-numeric chars except digits
    clean = re.sub(r'[^\d]', '', str(price_str))
    return int(clean) if clean else 0

def introduce_typo(text, probability=0.2):
    """Simulates user spelling errors."""
    if random.random() > probability:
        return text
    
    chars = list(text)
    if not chars: return text
    
    idx = random.randint(0, len(chars) - 1)
    char = chars[idx].lower()
    
    if char in TYPO_MAPPING:
        chars[idx] = TYPO_MAPPING[char]
    elif random.random() < 0.5 and len(chars) > 1:
        # Delete a character
        del chars[idx]
    
    return "".join(chars)

def generate_synthetic_queries(product_name, category, tags, price):
    """
    Generates a list of synthetic user queries for a specific product 
    to train the search engine.
    """
    queries = []
    
    p_name = product_name.lower()
    cat = category.lower()
    
    # 1. Exact Match & Variations
    queries.append(p_name)
    queries.append(f"{cat} {p_name}")
    
    # 2. Price based queries
    if price > 0:
        queries.append(f"prix {p_name}")
        queries.append(f"{p_name} {price} da")
        queries.append(f"combien coute {p_name}")

    # 3. Intent based (Natural Language)
    action = random.choice(SYNONYMS["buy"])
    queries.append(f"{action} {p_name}")
    queries.append(f"je cherche {p_name}")
    
    # 4. Category generic queries (Lower relevance usually, but relevant for this item)
    syn_cat = random.choice(SYNONYMS.get(cat, [cat]))
    queries.append(f"meilleur {syn_cat}")
    queries.append(f"{syn_cat} djezzy")
    
    # 5. Typo Injection (Simulating real users)
    queries.append(introduce_typo(p_name, probability=1.0))
    
    # 6. Tag based
    for tag in tags:
        queries.append(f"{cat} {tag}")
        queries.append(f"{p_name} {tag}")

    return list(set(queries)) # Remove duplicates

# ==========================================
# DATA EXTRACTION LOGIC
# ==========================================

def process_mobiles(json_data):
    """Extracts phones and accessories from scraping4.json"""
    products = []
    
    for item in json_data:
        brand = item.get("title", "").strip()
        model = item.get("description", "").strip()
        full_name = f"{brand} {model}".strip()
        price_raw = item.get("price", "0")
        price = clean_price(price_raw)
        
        # Determine Category
        category = "Smartphone"
        acc_keywords = ['cable', 'charger', 'kit', 'power bank', 'earbuds', 'ecouteur', 'holder', 'd-link', 'modem', 'router']
        if any(k in full_name.lower() for k in acc_keywords):
            category = "Accessoire"
            if "modem" in full_name.lower() or "d-link" in full_name.lower():
                category = "Modem/Routeur"

        # Create Entry
        products.append({
            "id": str(uuid.uuid4())[:8],
            "name": full_name,
            "category": category,
            "desc": f"{category} {brand} - {model}. Disponible chez Djezzy.",
            "price": price,
            "tags": [brand.lower(), model.lower(), "4g", "boutique"]
        })
    return products

def process_offers(json_data):
    """Extracts offers (Legend, etc) from scraping3.json"""
    products = []
    
    # Extracting Internet Offers
    if isinstance(json_data, list):
        for section in json_data:
            # Legend Offers
            if "legend_offers" in section:
                # Note: The JSON structure for legend_offers in scraping3 seems empty in the sample, 
                # but there is raw text in "info". We will parse "internet_offers" which is populated.
                pass

            if "internet_offers" in section:
                for offer in section["internet_offers"]:
                    name = offer.get("name", "Offre Internet")
                    details = offer.get("price_and_validity", "")
                    
                    # Construct a readable name
                    full_name = f"Offre Internet {name} Go"
                    if "Jours" in details or "24 H" in details:
                        full_name += f" ({details})"
                    
                    price = clean_price(details)
                    
                    products.append({
                        "id": str(uuid.uuid4())[:8],
                        "name": full_name,
                        "category": "Offre Internet",
                        "desc": f"Forfait internet Djezzy. {details} {offer.get('discount_info', '')}",
                        "price": price,
                        "tags": ["internet", "data", "4g", "forfait", "connexion"]
                    })

            # Twinbox
            if "twinbox" in section:
                tb = section["twinbox"]
                products.append({
                    "id": str(uuid.uuid4())[:8],
                    "name": "Djezzy Twinbox",
                    "category": "Modem/Routeur",
                    "desc": "Box All in One 4G/ADSL. TV, Internet et Téléphonie.",
                    "price": 12900, # Extracted from text description
                    "tags": ["wifi", "maison", "box", "tv", "adsl"]
                })
                
    return products

def process_services(json_data):
    """Extracts services from scraping3.json"""
    products = []
    if isinstance(json_data, list):
        for section in json_data:
            if "services" in section:
                for service in section["services"]:
                    title = service.get("title", "")
                    desc = service.get("description", "")
                    
                    products.append({
                        "id": str(uuid.uuid4())[:8],
                        "name": title,
                        "category": "Service",
                        "desc": desc if desc else f"Service Djezzy {title}",
                        "price": 0, # Services often have variable pricing or are free/included
                        "tags": ["service", "option", "activation", "code"]
                    })
    return products

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    all_items = []

    # 1. Load and Process Mobiles (Scraping 4)
    try:
        with open(FILES['shop'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_items.extend(process_mobiles(data))
            print(f"Loaded {len(all_items)} items from shop.")
    except FileNotFoundError:
        print(f"Warning: {FILES['shop']} not found.")

    # 2. Load and Process Offers/Services (Scraping 3)
    try:
        with open(FILES['offers'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            offers = process_offers(data)
            services = process_services(data)
            all_items.extend(offers)
            all_items.extend(services)
            print(f"Loaded {len(offers)} offers and {len(services)} services.")
    except FileNotFoundError:
        print(f"Warning: {FILES['offers']} not found.")

    # 3. Generate Dataset
    print("Generating synthetic queries and relevance labels...")
    
    csv_rows = []
    
    for item in all_items:
        # Generate relevant queries (Positive Samples)
        queries = generate_synthetic_queries(
            item['name'], 
            item['category'], 
            item['tags'], 
            item['price']
        )
        
        for query in queries:
            csv_rows.append({
                "product_id": item['id'],
                "product_name": item['name'],
                "category": item['category'],
                "description": item['desc'],
                "price": item['price'],
                "user_query": query,
                "relevance_label": 1 # 1 = Relevant
            })

    # 4. Write to CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["product_id", "product_name", "category", "description", "price", "user_query", "relevance_label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Successfully created {OUTPUT_FILE} with {len(csv_rows)} training examples.")

if __name__ == "__main__":
    main()