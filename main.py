import sys

from src.graph_builder import app


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Chemistry query: ").strip()
    if not query:
        print("No query provided.")
        sys.exit(1)

    print(f"\nQuery: {query}\n")
    result = app.invoke(
        {"query": query, "llm_response": "", "text": "", "smiles": None, "error": None}
    )

    if result.get("error"):
        print(f"Error: {result['error']}")
        print(f"Raw response:\n{result.get('llm_response', '')}")
    else:
        print("=== Explanation ===")
        print(result["text"])
        print("\n=== SMILES ===")
        print(result["smiles"])


if __name__ == "__main__":
    main()
