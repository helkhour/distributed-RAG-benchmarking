from datasets import load_dataset
import json

# --- Step 1: Load HotpotQA dataset ---
hotpotqa = load_dataset("facebook/kilt_tasks", name="hotpotqa", split="test")

# --- Step 2: Extract relevant titles ---
relevant_titles = set()
for entry in hotpotqa:
    for output in entry.get("output", []):
        for prov in output.get("provenance", []):
            title = prov.get("title")
            if title:
                relevant_titles.add(title.strip())

print(f"Extracted {len(relevant_titles)} unique relevant Wikipedia titles.")

# Optional: Save to file
with open("relevant_titles.txt", "w") as f:
    for title in sorted(relevant_titles):
        f.write(title + "\n")

# --- Step 3: Load KILT corpus and filter ---
print("Loading KILT corpus...")
kilt_corpus = load_dataset("corag/kilt-corpus", split="train", streaming=False)

filtered_corpus = []
for entry in kilt_corpus:
    if entry.get("title", "").strip() in relevant_titles:
        filtered_corpus.append({
            "wikipedia_id": entry["wikipedia_id"],
            "wikipedia_title": entry["title"],
            "contents": entry["contents"]
        })

print(f"Filtered corpus size: {len(filtered_corpus)}")

# --- Step 4: Save filtered corpus for embedding/testing ---
with open("sanity_corpus.jsonl", "w") as f:
    for doc in filtered_corpus:
        f.write(json.dumps(doc) + "\n")

print("Saved filtered corpus to sanity_corpus.jsonl")
