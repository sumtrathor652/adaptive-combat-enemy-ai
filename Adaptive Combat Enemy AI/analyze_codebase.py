import ast
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

FILES = ["brain.py", "fighter.py", "main.py"]


def analyze_file(filepath: Path) -> dict:
    source = filepath.read_text(encoding="utf-8")
    lines = source.splitlines()
    total_lines = len(lines)

    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith("#"):
            comment_lines += 1
        else:
            code_lines += 1

    tree = ast.parse(source, filename=str(filepath))

    imports = []
    classes = []
    functions = []  # (qualified_name, line_count)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    span = (getattr(item, "end_lineno", item.lineno) - item.lineno) + 1
                    functions.append((f"{node.name}.{item.name}", span))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            span = (getattr(node, "end_lineno", node.lineno) - node.lineno) + 1
            functions.append((node.name, span))

    return {
        "name": filepath.name,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "imports": sorted(set(imports)),
        "import_count": len(set(imports)),
        "class_count": len(classes),
        "func_count": len(functions),
        "functions": functions,
    }


def main(base_dir: str = "."):
    root = Path(base_dir)
    stats = []
    for fname in FILES:
        fpath = root / fname
        if fpath.exists():
            stats.append(analyze_file(fpath))
        else:
            print(f"Warning: {fpath} not found, skipping.")

    if not stats:
        print("No target files found to analyze.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("3-File Codebase Structure & Complexity Analysis", fontsize=15, fontweight="bold")

    names = [s["name"] for s in stats]

    # 1. Code vs Comment vs Blank lines
    ax1 = axes[0, 0]
    code_vals = [s["code_lines"] for s in stats]
    comm_vals = [s["comment_lines"] for s in stats]
    blank_vals = [s["blank_lines"] for s in stats]

    ax1.bar(names, code_vals, label="Code Lines", color="#3b82f6")
    ax1.bar(names, comm_vals, bottom=code_vals, label="Comment Lines", color="#10b981")
    bottom_both = [c + m for c, m in zip(code_vals, comm_vals)]
    ax1.bar(names, blank_vals, bottom=bottom_both, label="Blank Lines", color="#94a3b8")

    for i, s in enumerate(stats):
        ax1.text(i, s["total_lines"] + 4, f"{s['total_lines']}L", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_title("Line Breakdown by File")
    ax1.set_ylabel("Lines")
    ax1.set_ylim(0, max(s["total_lines"] for s in stats) * 1.15 + 5)
    ax1.legend(loc="upper left", fontsize=8)

    # 2. Structural elements (Classes, Functions/Methods, Unique Imports)
    ax2 = axes[0, 1]
    x_idx = range(len(names))
    width = 0.25
    cls_vals = [s["class_count"] for s in stats]
    fn_vals = [s["func_count"] for s in stats]
    imp_vals = [s["import_count"] for s in stats]

    ax2.bar([i - width for i in x_idx], cls_vals, width=width, label="Classes", color="#8b5cf6")
    ax2.bar(x_idx, fn_vals, width=width, label="Functions/Methods", color="#f59e0b")
    ax2.bar([i + width for i in x_idx], imp_vals, width=width, label="Imports", color="#06b6d4")

    ax2.set_xticks(list(x_idx))
    ax2.set_xticklabels(names)
    ax2.set_title("Structural Elements per File")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper left", fontsize=8)

    # 3. Longest functions/methods across the 3 files
    ax3 = axes[1, 0]
    all_funcs = []
    for s in stats:
        for fn_name, span in s["functions"]:
            all_funcs.append((f"{s['name']}: {fn_name}", span))
    all_funcs.sort(key=lambda item: item[1], reverse=True)
    top_funcs = all_funcs[:8][::-1]  # bottom-to-top horizontal bar

    if top_funcs:
        fn_labels = [item[0] for item in top_funcs]
        fn_spans = [item[1] for item in top_funcs]
        ax3.barh(fn_labels, fn_spans, color="#ef4444")
        for idx, val in enumerate(fn_spans):
            ax3.text(val + 0.8, idx, f"{val}L", va="center", fontsize=8)
    ax3.set_title("Largest Functions / Methods (Lines)")
    ax3.set_xlabel("Line Span")

    # 4. Module dependency & import summary
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("Module Imports & Coupling Check")

    y_pos = 0.85
    for s in stats:
        imp_str = ", ".join(s["imports"]) if s["imports"] else "none"
        has_pygame = "pygame" in s["imports"]
        badge = " [uses pygame]" if has_pygame else " [no pygame]"
        line_txt = f"{s['name']}{badge}:\n   imports -> {imp_str}"
        color = "#b45309" if (s["name"] == "brain.py" and has_pygame) else "#1e293b"
        ax4.text(0.03, y_pos, line_txt, fontsize=9.5, color=color, va="top", family="monospace")
        y_pos -= 0.28

    patch_ok = mpatches.Patch(color="#10b981", label="Clean separation")
    ax4.legend(handles=[patch_ok], loc="lower left", fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = root / "codebase_analysis.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved chart to {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()