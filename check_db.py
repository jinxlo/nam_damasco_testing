# check_db.py
import os
from sqlalchemy import create_engine, or_, and_, func
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Assuming your Product model is in namwoo_app.models.product
from namwoo_app.models.product import Product

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env file")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_query(description, query):
    print(f"--- {description} ---")
    with SessionLocal() as db:
        results = query(db).all()
        if results:
            print(f"FOUND {len(results)} item(s):")
            for product in results:
                print(f"  - ID: {product.id}, SKU: {product.item_code}, Name: {product.item_name}")
        else:
            print("  -> No items found.")
    print("-" * 20 + "\n")

# --- QUERIES TO TEST ---

# Test 1: The flawed query I wrote before
def flawed_query(db):
    search_term = "ITEL A80"
    return db.query(Product).filter(Product.item_name.ilike(f'%{search_term}%'))

# Test 2: The CORRECT query that should have been written
def correct_query(db):
    search_terms = "ITEL A80".split()
    conditions = [Product.item_name.ilike(f'%{term}%') for term in search_terms]
    return db.query(Product).filter(and_(*conditions))
    
# Test 3: Specific SKU search to prove the item exists
def sku_query(db):
    sku = "D0008436" # The SKU for ITEL A80 from your data
    return db.query(Product).filter(Product.item_code == sku)


if __name__ == "__main__":
    print("Running database diagnostics...\n")
    test_query("Test 1: Flawed ILIKE search (what's currently failing)", flawed_query)
    test_query("Test 2: Correct word-based ILIKE search (what we need)", correct_query)
    test_query("Test 3: Direct SKU search (to confirm data exists)", sku_query)