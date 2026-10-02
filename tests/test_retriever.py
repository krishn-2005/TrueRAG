from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.retriever import get_retriever


retriever = get_retriever()

docs = retriever.invoke("What is self-attention?")

for doc in docs:
    print(doc.page_content[:300])
    print(doc.metadata.page)
    print("-" * 50)