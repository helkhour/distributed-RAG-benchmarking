import json
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Create a KILT corpus subset containing documents referenced by HotpotQA queries.")
    parser.add_argument("--dataset", default="facebook/kilt_tasks", help="Dataset with HotpotQA tasks")
    parser.add_argument("--subset", default="hotpotqa", help="Subset name (default: hotpotqa)")
    parser.add_argument("--corpus", default="corag/kilt-corpus", help="KILT corpus dataset name")
    parser.add_argument("--output", default="hotpotqa_corpus_subset.jsonl", help="Output JSONL file")
    parser.add_argument("--split", default="train", help="Split to use for HotpotQA queries (default: train)")
    args = parser.parse_args()

    print(f"Loading HotpotQA dataset: {args.dataset} ({args.subset})")
    hotpotqa = load_dataset(args.dataset, name=args.subset, split=args.split)

    wiki_ids = set()
    titles = set()
    for entry in hotpotqa:
        for output in entry.get("output", []):
            for prov in output.get("provenance", []):
                wid = prov.get("wikipedia_id")
                title = prov.get("title")
                if wid:
                    wiki_ids.add(str(wid))
                if title:
                    titles.add(title.strip())

    print(f"Found {len(wiki_ids)} unique wikipedia_ids and {len(titles)} titles in provenance")

    print(f"Streaming KILT corpus {args.corpus} to gather matching documents...")
    corpus = load_dataset(args.corpus, split="train", streaming=True)
    matched = 0
    with open(args.output, "w") as f:
        for doc in corpus:
            if str(doc.get("wikipedia_id")) in wiki_ids or doc.get("title", "").strip() in titles:
                matched += 1
                record = {
                    "wikipedia_id": doc["wikipedia_id"],
                    "wikipedia_title": doc["title"],
                    "contents": doc["contents"],
                }
                f.write(json.dumps(record) + "\n")
    print(f"Saved {matched} documents to {args.output}")


if __name__ == "__main__":
    main()
