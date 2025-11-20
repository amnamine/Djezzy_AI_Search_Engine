import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# --- CONFIGURATION ---
DATASET_FILE = "Djezzy_AI_Training_Dataset_L3.csv"
MODEL_FILE = "djezzy_ai_brain.pkl"

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
        # The 'Brain' (Pipeline)
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

        # Create features
        df['features'] = df['user_query'].apply(preprocess_query) + " | " + \
                         df['product_name'] + " " + df['category'] + " " + df['description']
        
        X = df['features']
        y = df['relevance_label']

        print(f"[AI] Training model on {len(df)} examples...")
        self.pipeline.fit(X, y)
        
        # Prepare the searchable database (only valid products)
        self.product_db = df[df['relevance_label'] == 1].drop_duplicates(subset=['product_id']).copy()
        self.product_db['search_text'] = self.product_db['product_name'] + " " + \
                                         self.product_db['category'] + " " + \
                                         self.product_db['description']
        
        print("[AI] Training Complete.")

    def save_model(self, filename):
        """Saves the trained pipeline AND the product database to a file."""
        if self.product_db is None:
            print("[ERROR] Cannot save: Model is not trained yet.")
            return
            
        # We package everything into a dictionary
        model_package = {
            'pipeline': self.pipeline,
            'database': self.product_db
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_package, f)
            print(f"[SUCCESS] Model saved to '{filename}'")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")

    def load_model(self, filename):
        """Loads a pre-trained model from disk."""
        print(f"[AI] Loading model from '{filename}'...")
        try:
            with open(filename, 'rb') as f:
                model_package = pickle.load(f)
            
            self.pipeline = model_package['pipeline']
            self.product_db = model_package['database']
            print("[SUCCESS] Model loaded! Ready to search.")
            return True
        except FileNotFoundError:
            print(f"[WARN] '{filename}' not found. You need to train first.")
            return False

    def search(self, user_query, top_k=5):
        if self.product_db is None:
            print("[ERROR] Model not ready (Train or Load first).")
            return pd.DataFrame()

        clean_query = preprocess_query(user_query)
        
        # Create candidates
        candidates = self.product_db.copy()
        candidate_features = clean_query + " | " + candidates['search_text']
        
        # AI Prediction (Probability)
        probs = self.pipeline.predict_proba(candidate_features)[:, 1]
        candidates['ai_score'] = probs
        
        final_results = candidates.sort_values(by='ai_score', ascending=False).head(top_k)
        return final_results[['product_name', 'price', 'ai_score', 'description']]

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    engine = DjezzySearchAI()
    
    # CHECK: If model exists, load it. If not, train it.
    if os.path.exists(MODEL_FILE):
        engine.load_model(MODEL_FILE)
    else:
        print("[INFO] No saved model found. Starting training...")
        engine.train(DATASET_FILE)
        engine.save_model(MODEL_FILE) # <--- SAVING HAPPENS HERE
    
    # --- DEMO ---
    test_queries = [
        "internet 2000",        
        "legende deux mille",   
        "samsng galaxy",        
        "verser credit",        
        "wifi dar",             
    ]
    
    print("\n" + "="*50)
    print("   DJIBLY INTELLIGENT SEARCH DEMO   ")
    print("="*50)

    for q in test_queries:
        print(f"\n>> User Search: '{q}'")
        try:
            results = engine.search(q)
            
            if results.empty:
                 print("   (System not ready)")
                 continue

            found_any = False
            for i, row in results.iterrows():
                if row['ai_score'] > 0.2:
                    found_any = True
                    # Using simple characters for Windows compatibility
                    stars = "*" * int(row['ai_score'] * 5)
                    print(f"   [{stars}] ({row['ai_score']:.2f}) -> {row['product_name']} [{row['price']} DA]")
            
            if not found_any:
                print("   (No relevant results found)")

        except Exception as e:
            print(f"Error displaying results: {e}")