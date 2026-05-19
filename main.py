import sys

from src.graph_builder import app


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Chemistry query: ").strip()
    if not query:
        print("No query provided.")
        sys.exit(1)

    print(f"\nQuery: {query}\n")
    result = app.invoke({"query": query})

    if result.get("error"):
        print(f"Warning: {result['error']}")

    print("=== Explanation ===")
    print(result.get("text", ""))

    reactions = result.get("reactions") or []
    print(f"\n=== Reactions ({len(reactions)}) ===")
    for i, rxn in enumerate(reactions, 1):
        print(f"\n[{i}] {rxn.get('caption') or ''}")
        print(f"    SMILES:     {rxn.get('smiles')}")
        if rxn.get("reagents"):
            print(f"    Reagents:   {', '.join(rxn['reagents'])}")
        if rxn.get("conditions"):
            print(f"    Conditions: {', '.join(rxn['conditions'])}")

    follow = result.get("follow_up") or []
    if follow:
        print("\n=== Follow-up ===")
        for q in follow:
            print(f"  • {q}")


if __name__ == "__main__":
    main()
