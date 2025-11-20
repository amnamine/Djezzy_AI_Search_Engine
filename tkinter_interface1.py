import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# ==========================================
# 1. THE AI BACKEND
# ==========================================
SYNONYMS = {
    "kitman": "ecouteurs", "wifi": "modem", "net": "internet",
    "hbal": "hayla", "bezef": "hayla", "puce": "sim",
    "legende": "legend", "verser": "flexy", "storm": "flexy"
}

def preprocess_query(query):
    text = str(query).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    expanded = []
    for w in words:
        expanded.append(w)
        if w in SYNONYMS:
            expanded.append(SYNONYMS[w])
    return " ".join(expanded)

class DjezzySearchAI:
    def __init__(self):
        self.product_db = None
        self.pipeline = None

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                model_package = pickle.load(f)
            self.pipeline = model_package['pipeline']
            self.product_db = model_package['database']
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def search(self, user_query, top_k=10):
        if self.product_db is None: return pd.DataFrame()
        clean_query = preprocess_query(user_query)
        candidates = self.product_db.copy()
        candidate_features = clean_query + " | " + candidates['search_text']
        probs = self.pipeline.predict_proba(candidate_features)[:, 1]
        candidates['ai_score'] = probs
        return candidates.sort_values(by='ai_score', ascending=False).head(top_k)

# ==========================================
# 2. THE IMPROVED UI
# ==========================================
class DjezzySearchApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("DJIBLY Intelligent Search v2.0")
        self.geometry("550x800")
        self.configure(bg="#F0F2F5") 
        
        self.COLORS = {
            "primary": "#E3001B",   # Djezzy Red
            "dark": "#2D3436",      # Dark Text
            "bg": "#F0F2F5",        # App Background
            "card": "#FFFFFF",      # White Card
            "accent": "#00B894",    # Green Success
            "chip": "#DFE6E9",      # Chip Grey
            "chip_text": "#636E72"
        }
        self.FONTS = {
            "header": ("Segoe UI", 18, "bold"),
            "sub": ("Segoe UI", 9),
            "title": ("Segoe UI", 11, "bold"),
            "body": ("Segoe UI", 10),
            "price": ("Segoe UI", 12, "bold")
        }

        # --- Load AI ---
        self.engine = DjezzySearchAI()
        self.model_loaded = False
        if os.path.exists("djezzy_ai_brain.pkl"):
            if self.engine.load_model("djezzy_ai_brain.pkl"):
                self.model_loaded = True
            else:
                messagebox.showerror("Error", "Failed to load brain.")
        else:
            messagebox.showwarning("Warning", "Model file not found!")

        # --- Build Layout ---
        self.create_header()
        self.create_search_area()
        self.create_suggestions()
        self.create_results_area()
        self.create_footer()

        self.bind('<Return>', lambda event: self.run_search())

    def create_header(self):
        header = tk.Frame(self, bg=self.COLORS["primary"], height=90)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        # Logo Text
        lbl_title = tk.Label(header, text="DJIBLY PoS", font=self.FONTS["header"], 
                             bg=self.COLORS["primary"], fg="white")
        lbl_title.pack(pady=(15, 0))
        
        lbl_sub = tk.Label(header, text="AI-Powered Search Module", font=self.FONTS["sub"], 
                           bg=self.COLORS["primary"], fg="#ffcccc")
        lbl_sub.pack()

    def create_search_area(self):
        frame = tk.Frame(self, bg=self.COLORS["bg"], pady=15, padx=20)
        frame.pack(fill="x")

        self.search_var = tk.StringVar()
        
        # Wrapper for entry + icon
        entry_frame = tk.Frame(frame, bg="white", bd=1, relief="solid")
        entry_frame.pack(fill="x", pady=5)
        
        self.entry = tk.Entry(entry_frame, textvariable=self.search_var, font=self.FONTS["body"], 
                              bd=0, bg="white")
        self.entry.pack(fill="x", padx=10, pady=10)
        self.entry.focus()

        # Buttons Row
        btn_frame = tk.Frame(frame, bg=self.COLORS["bg"])
        btn_frame.pack(fill="x", pady=(10,0))

        # Search Button
        btn_search = tk.Button(btn_frame, text="🔍 SEARCH", command=self.run_search,
                               bg=self.COLORS["dark"], fg="white", font=("Segoe UI", 10, "bold"),
                               relief="flat", cursor="hand2", width=20)
        btn_search.pack(side="left", padx=(0, 5))

        # Reset Button
        btn_reset = tk.Button(btn_frame, text="❌ RESET", command=self.reset_app,
                              bg="#B2BEC3", fg="white", font=("Segoe UI", 10, "bold"),
                              relief="flat", cursor="hand2", width=10)
        btn_reset.pack(side="right")

    def create_suggestions(self):
        # Quick keywords container
        frame = tk.Frame(self, bg=self.COLORS["bg"], padx=20)
        frame.pack(fill="x", pady=(0, 10))

        lbl = tk.Label(frame, text="Quick Keywords:", font=("Segoe UI", 8, "bold"), 
                       bg=self.COLORS["bg"], fg="#636E72")
        lbl.pack(anchor="w", pady=(0, 5))

        # Chip Container
        chips_frame = tk.Frame(frame, bg=self.COLORS["bg"])
        chips_frame.pack(anchor="w")

        keywords = ["Internet", "Legend 2000", "Modem", "Flexy", "TwinBox"]
        
        for kw in keywords:
            btn = tk.Button(chips_frame, text=kw, 
                            command=lambda k=kw: self.fill_search(k),
                            bg="white", fg=self.COLORS["primary"], 
                            bd=1, relief="solid", font=("Segoe UI", 9),
                            cursor="hand2", padx=10, pady=2)
            btn.pack(side="left", padx=(0, 8))

    def create_results_area(self):
        container = tk.Frame(self, bg=self.COLORS["bg"])
        container.pack(fill="both", expand=True, padx=15)

        self.canvas = tk.Canvas(container, bg=self.COLORS["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.COLORS["bg"])

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=500)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def create_footer(self):
        footer = tk.Frame(self, bg="#DFE6E9", height=30)
        footer.pack(fill="x", side="bottom")
        
        self.status_lbl = tk.Label(footer, text="Ready", font=("Segoe UI", 8), 
                                   bg="#DFE6E9", fg="#636E72")
        self.status_lbl.pack(pady=5)

    # --- Logic ---

    def fill_search(self, text):
        self.search_var.set(text)
        self.run_search()

    def reset_app(self):
        self.search_var.set("")
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.status_lbl.config(text="Ready")
        self.entry.focus()

    def run_search(self):
        if not self.model_loaded: return
        query = self.search_var.get()
        if not query.strip(): return

        # Clear old
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Search
        results = self.engine.search(query, top_k=15)
        relevant = results[results['ai_score'] > 0.1]

        if relevant.empty:
            lbl = tk.Label(self.scrollable_frame, text=f"No matches found for '{query}'", 
                           bg=self.COLORS["bg"], fg="#b2bec3", font=("Segoe UI", 12))
            lbl.pack(pady=40)
            self.status_lbl.config(text="0 results found.")
        else:
            self.status_lbl.config(text=f"Found {len(relevant)} relevant results.")
            for _, row in relevant.iterrows():
                self.draw_card(row)

    def draw_card(self, row):
        # Main Card
        card = tk.Frame(self.scrollable_frame, bg="white", padx=15, pady=12)
        card.pack(fill="x", pady=6)
        
        # Header Row (Name + Price)
        header = tk.Frame(card, bg="white")
        header.pack(fill="x")
        
        tk.Label(header, text=row['product_name'], font=self.FONTS["title"], 
                 bg="white", fg=self.COLORS["dark"]).pack(side="left")
        
        tk.Label(header, text=f"{row['price']} DA", font=self.FONTS["price"], 
                 bg="white", fg=self.COLORS["primary"]).pack(side="right")
        
        # Description
        desc = str(row['description'])
        if len(desc) > 70: desc = desc[:70] + "..."
        tk.Label(card, text=desc, font=("Segoe UI", 9), bg="white", fg="#636E72", anchor="w").pack(fill="x", pady=(2, 8))

        # AI Confidence Bar
        score = int(row['ai_score'] * 100)
        bar_color = self.COLORS["accent"] if score > 60 else "#FDCB6E"
        
        bar_frame = tk.Frame(card, bg="white")
        bar_frame.pack(fill="x")
        
        # Grey background of bar
        tk.Label(bar_frame, text="Relevance:", font=("Segoe UI", 7, "bold"), bg="white", fg="#B2BEC3").pack(side="left")
        
        progress_bg = tk.Frame(bar_frame, bg="#F0F2F5", height=4, width=100)
        progress_bg.pack(side="left", padx=10)
        progress_bg.pack_propagate(False)
        
        progress_fill = tk.Frame(progress_bg, bg=bar_color, height=4, width=score)
        progress_fill.pack(side="left")
        
        tk.Label(bar_frame, text=f"{score}%", font=("Segoe UI", 8, "bold"), bg="white", fg=bar_color).pack(side="right")

if __name__ == "__main__":
    app = DjezzySearchApp()
    app.mainloop()