import ast
import pathlib

roots = ['app','core','indexer','retrieval','orchestrator','tracing']
seen = set()

for root in roots:
    for path in pathlib.Path(root).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    seen.add(n.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                seen.add(node.module.split('.')[0])

print("\n".join(sorted(seen)))