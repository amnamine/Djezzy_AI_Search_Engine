import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATASET_FILE = "Djezzy_AI_Training_Dataset_L3.csv"
MODEL_FILE = "djezzy_ai_model.pkl"

# --- 1. THE BRAIN: SYNONYM MAPPING ---
SYNONYMS = {
    "kitman": "ecouteurs",
    "wifi": "modem",
    "net": "internet",
    "hbal": "hayla",
    "bezef": "hayla",
    "puce": "sim",
    "legende": "legend",
    "verser": "flexy",
    "storm": "flexy"
}

def preprocess_query(query):
    """Cleans text and expands synonyms."""
    text = str(query).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    
    words = text.split()
    expanded = []
    for w in words:
        expanded.append(w)
        if w in SYNONYMS:
            expanded.append(SYNONYMS[w])
            
    return " ".join(expanded)

# --- 2. THE AI ENGINE CLASS ---
class DjezzySearchAI:
    def __init__(self):
        self.product_db = None
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))),
            ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42))
        ])
        
    def train(self, csv_path):
        print(f"[AI] Loading dataset from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print("[ERROR] Dataset not found. Please run the generator script first.")
            return

        df['features'] = df['user_query'].apply(preprocess_query) + " | " + \
                         df['product_name'] + " " + df['category'] + " " + df['description']
        
        X = df['features']
        y = df['relevance_label']

        print(f"[AI] Training model on {len(df)} examples...")
        self.pipeline.fit(X, y)
        
        self.product_db = df[df['relevance_label'] == 1].drop_duplicates(subset=['product_id']).copy()
        self.product_db['search_text'] = self.product_db['product_name'] + " " + \
                                         self.product_db['category'] + " " + \
                                         self.product_db['description']
        
        # FIXED: Removed Emoji
        print("[AI] Model Trained Successfully! [OK]")

    def search(self, user_query, top_k=5):
        if self.product_db is None:
            print("[ERROR] Model not trained.")
            return

        clean_query = preprocess_query(user_query)
        
        candidates = self.product_db.copy()
        candidate_features = clean_query + " | " + candidates['search_text']
        
        probs = self.pipeline.predict_proba(candidate_features)[:, 1]
        candidates['ai_score'] = probs
        
        final_results = candidates.sort_values(by='ai_score', ascending=False).head(top_k)
        return final_results[['product_name', 'price', 'ai_score', 'description']]

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    engine = DjezzySearchAI()
    engine.train(DATASET_FILE)
    
    test_queries = [
        "internet 2000",        
        "legende deux mille",   
        "samsng galaxy",        
        "verser credit",        
        "wifi dar",             
    ]
    
    # FIXED: Removed Emojis from banner
    print("\n" + "="*50)
    print("   DJIBLY INTELLIGENT SEARCH DEMO   ")
    print("="*50)

    for q in test_queries:
        # FIXED: Removed Emoji
        print(f"\n>> User Search: '{q}'")
        try:
            results = engine.search(q)
            
            for i, row in results.iterrows():
                if row['ai_score'] > 0.2:
                    # FIXED: Replaced Star Emoji with Asterisk (*)
                    stars = "*" * int(row['ai_score'] * 5)
                    print(f"   {stars} ({row['ai_score']:.2f}) -> {row['product_name']} [{row['price']} DA]")
                else:
                    print("   (No relevant results found)")
                    break
        except Exception as e:
            print(f"Error displaying results: {e}")