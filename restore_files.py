"""
Restore corrupted Python files by using token start positions to reconstruct
inter-token whitespace. The files were corrupted by writing tokens without
inter-token whitespace, but indentation structure is preserved.

The corruption turns:
    from __future__ import annotations
into:
    from__future__importannotations

This script uses a different approach: since the indentation is preserved,
we can reconstruct spacing by re-rendering each (line, col) position.
"""
from __future__ import annotations

import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

TARGETS_CORRUPTED = [
    ROOT / "demo",
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "conftest.py",
]

SKIP_DIRS = {"__pycache__", ".venv", ".pytest_cache", ".git", "strip_comments.py"}


def collect_files() -> list[Path]:
    files: list[Path] = []
    for target in TARGETS_CORRUPTED:
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            for p in target.rglob("*.py"):
                if not any(part in SKIP_DIRS for part in p.parts):
                    files.append(p)
    files.sort()
    return files


def reconstruct_from_tokens(source: str) -> str:
    """
    Given a corrupted source (tokens without inter-token spaces),
    reconstruct the proper Python source by using token positions.

    tokenize gives us (line, col) for each token. We can reconstruct
    by placing each token at its column position on its line.
    """
    lines_in = source.splitlines(keepends=True)

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError as e:
        print(f"    tokenize failed: {e}")
        return source

    # Build output: for each output line, we place chars at specific columns
    # First, figure out max line number
    max_line = max((tok_end[0] for _, _, _, tok_end, _ in tokens), default=0)
    max_line = max(max_line, len(lines_in))

    # Initialize output lines (0-indexed)
    out_lines: list[list[str]] = [[] for _ in range(max_line + 1)]

    # Process each output line: determine its indentation from the input
    for i, line in enumerate(lines_in):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        out_lines[i] = list(indent)  # pre-fill with indentation chars

    for tok_type, tok_string, tok_start, tok_end, _ in tokens:
        if tok_type in (tokenize.ENDMARKER, tokenize.ENCODING):
            continue
        if tok_type in (tokenize.NEWLINE, tokenize.NL):
            line_idx = tok_start[0] - 1
            if 0 <= line_idx < len(out_lines):
                # Ensure newline at end
                current = "".join(out_lines[line_idx])
                if not current.endswith("\n"):
                    out_lines[line_idx].append("\n")
            continue
        if tok_type in (tokenize.INDENT, tokenize.DEDENT):
            continue
        if tok_type == tokenize.COMMENT:
            continue

        line_idx = tok_start[0] - 1
        col = tok_start[1]

        if 0 <= line_idx < len(out_lines):
            # Pad to col position
            current_len = len("".join(out_lines[line_idx]))
            if current_len < col:
                out_lines[line_idx].append(" " * (col - current_len))
            # Append the token string (handle multi-line tokens)
            tok_lines = tok_string.split("\n")
            if len(tok_lines) == 1:
                out_lines[line_idx].append(tok_string)
            else:
                # Multi-line token (triple-quoted string etc.)
                out_lines[line_idx].append(tok_lines[0])
                for extra_idx, extra_line in enumerate(tok_lines[1:], 1):
                    target_line = line_idx + extra_idx
                    if target_line < len(out_lines):
                        out_lines[target_line].append(extra_line)

    # Join all lines
    reconstructed = ""
    for idx, chars in enumerate(out_lines):
        line_str = "".join(chars)
        # Ensure newline
        if line_str and not line_str.endswith("\n"):
            # Only add newline if not the last line or if original had it
            if idx < len(lines_in):
                if lines_in[idx].endswith("\n"):
                    line_str += "\n"
        reconstructed += line_str

    return reconstructed


def is_valid_python(source: str) -> bool:
    try:
        compile(source, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def process_file(path: Path) -> bool:
    try:
        corrupted = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  SKIP (read error) : {path.relative_to(ROOT)} — {e}")
        return False

    if is_valid_python(corrupted):
        print(f"  already valid     : {path.relative_to(ROOT)}")
        return False

    restored = reconstruct_from_tokens(corrupted)

    if is_valid_python(restored):
        path.write_text(restored, encoding="utf-8")
        print(f"  restored          : {path.relative_to(ROOT)}")
        return True
    else:
        print(f"  FAILED to restore : {path.relative_to(ROOT)}")
        return False


def main() -> None:
    files = collect_files()
    print(f"Found {len(files)} Python files.\n")
    restored_count = 0
    failed: list[Path] = []

    for f in files:
        ok = process_file(f)
        if ok:
            restored_count += 1
        # Check if failed
        try:
            src = f.read_text(encoding="utf-8")
            if not is_valid_python(src):
                failed.append(f)
        except Exception:
            pass

    print(f"\nRestored: {restored_count}/{len(files)}")
    if failed:
        print(f"Still invalid ({len(failed)}):")
        for f in failed:
            print(f"  {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
