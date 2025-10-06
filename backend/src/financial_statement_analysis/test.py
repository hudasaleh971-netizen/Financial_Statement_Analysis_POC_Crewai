from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# 1. Connect to your local Qdrant instance
client = QdrantClient(path="./qdrant_db")  # stored on disk

# 2. Name of the collection
collection_name = "financial_docs"

# 3. The filename you want to search for
filename = "HSBC-11.pdf"

# 4. Build a metadata filter
metadata_filter = qmodels.Filter(
    must=[
        qmodels.FieldCondition(
            key="metadata.filename",           # or just "filename" if that's how it was stored
            match=qmodels.MatchValue(value=filename)
        )
    ]
)

# 5. Perform the search using scroll (no vector similarity involved)
points, next_page_offset = client.scroll(
    collection_name=collection_name,
    scroll_filter=metadata_filter,
    limit=20,       # adjust number of records to fetch
    with_payload=True,
    with_vectors=False
)

# 6. Display results
print(f"Found {len(points)} points for file: {filename}\n")
for p in points:
    print(f"ID: {p.id}")
    print(f"Payload: {p.payload}")
    print("-" * 40)
