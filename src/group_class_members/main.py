"""
Group class members by category.

Order enforced:
  1. Class variables / ClassVar annotations
  2. Instance variable annotations (no default, e.g. x: int)
  3. __init__ and __new__
  4. Other dunder methods
  5. @classmethod and @staticmethod
  6. @property and @<prop>.setter/.deleter
  7. Public methods
  8. Private methods (_foo)
  9. Dunder-private methods (__foo, name-mangled)
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path

# ---------------------------------------------------------------------------
# Member categories (lower value = earlier in class body)
# ---------------------------------------------------------------------------


class Category(IntEnum):
    """Order of member categories."""

    CLASS_VAR = auto()  # ClassVar[...] or bare assignment at class level
    INSTANCE_VAR = auto()  # Bare annotation without value: x: int
    PROPERTY = auto()  # @property, @x.setter, @x.deleter
    INIT_NEW = auto()  # __init__, __new__
    DUNDER = auto()  # Other __foo__ methods
    CLASS_STATIC = auto()  # @classmethod, @staticmethod
    PUBLIC_METHOD = auto()  # def foo(...)
    PRIVATE_METHOD = auto()  # def _foo(...)
    MANGLED_METHOD = auto()  # def __foo (no trailing __)


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
    return False


def _is_classvar(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    # ClassVar or ClassVar[X]
    if isinstance(annotation, ast.Name) and annotation.id == "ClassVar":
        return True
    if isinstance(annotation, ast.Subscript):
        v = annotation.value
        if isinstance(v, ast.Name) and v.id == "ClassVar":
            return True
        if isinstance(v, ast.Attribute) and v.attr == "ClassVar":
            return True
    return False


def categorize(node: ast.stmt) -> Category:
    """Return the Category for a single class-body statement."""
    # --- Annotated assignments: x: T  or  x: T = value ---
    if isinstance(node, ast.AnnAssign):
        if _is_classvar(node.annotation):
            return Category.CLASS_VAR
        return Category.INSTANCE_VAR

    # --- Plain assignments: x = value  (class-level) ---
    if isinstance(node, ast.Assign):
        return Category.CLASS_VAR

    # --- Function / async function ---
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        name = node.name

        # @property family
        if _has_decorator(node, "property"):
            return Category.PROPERTY
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter", "getter"):
                return Category.PROPERTY

        # @classmethod / @staticmethod
        if _has_decorator(node, "classmethod") or _has_decorator(node, "staticmethod"):
            return Category.CLASS_STATIC

        # __init__ / __new__
        if name in ("__init__", "__new__"):
            return Category.INIT_NEW

        # other dunders: __foo__
        if name.startswith("__") and name.endswith("__"):
            return Category.DUNDER

        # name-mangled: __foo  (no trailing __)
        if name.startswith("__"):
            return Category.MANGLED_METHOD

        # private: _foo
        if name.startswith("_"):
            return Category.PRIVATE_METHOD

        return Category.PUBLIC_METHOD

    # Everything else (expressions, pass, …) stays where it is
    return Category.CLASS_VAR


# ---------------------------------------------------------------------------
# Source extraction helpers (preserve original text incl. decorators)
# ---------------------------------------------------------------------------


@dataclass
class MemberChunk:
    """A contiguous block of source lines belonging to one class member."""

    category: Category
    lines: list[str] = field(default_factory=list)


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _is_attr_docstring(node: ast.stmt, prev: ast.stmt | None) -> bool:
    """
    Return True if *node* is a string-literal attribute docstring.

    An attribute docstring is a bare ``Expr(Constant(str))`` immediately
    following an ``AnnAssign`` or ``Assign``.
    """
    if not isinstance(node, ast.Expr):
        return False
    if not isinstance(node.value, ast.Constant):
        return False
    if not isinstance(node.value.value, str):
        return False
    return isinstance(prev, (ast.AnnAssign, ast.Assign))


def _member_start(node: ast.stmt) -> int:
    """First source line of *node*, including decorators (1-based)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _member_end(node: ast.stmt) -> int:
    """Last source line of *node* (1-based, inclusive)."""
    return node.end_lineno  # type: ignore[attr-defined]


def _build_chunks(
    body: list[ast.stmt],
    all_lines: list[str],
    class_header_end: int,
) -> list[MemberChunk]:
    """
    Group body statements into sortable chunks.

    Each chunk contains:
    - leading ``#``-comment lines (blank lines are dropped and re-emitted
      canonically during reassembly)
    - the member's own source lines
    - the attribute docstring immediately following it, if any

    Parameters
    ----------
    body : list[ast.stmt]
        ``class_node.body``
    all_lines : list[str]
        Full file split into lines (keepends=True).
    class_header_end : int
        1-based line number of the last line of the ``class Foo:`` header,
        i.e. the line directly before the first body member.

    Returns
    -------
    list[MemberChunk]

    """
    # First pass: merge attribute docstrings into their owner node.
    # Produce list of (primary_node, docstring_node | None).
    grouped: list[tuple[ast.stmt, ast.stmt | None]] = []
    skip_next = False
    for i, node in enumerate(body):
        if skip_next:
            skip_next = False
            continue
        next_node = body[i + 1] if i + 1 < len(body) else None
        if next_node is not None and _is_attr_docstring(next_node, node):
            grouped.append((node, next_node))
            skip_next = True
        else:
            grouped.append((node, None))

    # Second pass: collect source lines for each group.
    chunks: list[MemberChunk] = []
    prev_source_end = class_header_end  # 1-based line of previous member's last line

    for primary, docstr in grouped:
        start = _member_start(primary)
        end = _member_end(docstr if docstr is not None else primary)

        # Leading comment lines between prev member and this one.
        # Blank lines are intentionally dropped here.
        leading: list[str] = []
        for li in range(prev_source_end, start - 1):  # li is 0-based
            line = all_lines[li]
            if line.lstrip().startswith("#"):
                leading.append(line)

        member_lines = leading + all_lines[start - 1 : end]
        chunks.append(
            MemberChunk(
                category=categorize(primary),
                lines=member_lines,
            )
        )
        prev_source_end = end  # end is 1-based, but used as 0-based offset next iter

    return chunks


# ---------------------------------------------------------------------------
# Core transformer
# ---------------------------------------------------------------------------


def sort_class_body(
    source: str,
    class_node: ast.ClassDef,
    all_lines: list[str],
) -> str:
    """
    Return the rewritten source with *class_node*'s body sorted.

    Parameters
    ----------
    source : str
        Full file source.
    class_node : ast.ClassDef
        The class whose body should be sorted.
    all_lines : list[str]
        ``source.splitlines(keepends=True)``

    Returns
    -------
    str
        Modified full file source (unchanged if already sorted).

    """
    body = class_node.body
    class_header_end = body[0].lineno - 1  # 1-based line just before first member

    chunks = _build_chunks(body, all_lines, class_header_end)

    # Keep the original order within each category by relying on Python's
    # stable sort and sorting by category only.
    sorted_chunks = sorted(chunks, key=lambda c: c.category)

    # Already sorted?
    if all(a is b for a, b in zip(chunks, sorted_chunks)):
        return source

    # ------------------------------------------------------------------
    # Reassemble with canonical spacing:
    #   same category  → 1 blank line between members
    #   diff category  → 2 blank lines between members
    # ------------------------------------------------------------------
    new_body_lines: list[str] = []
    prev_cat: Category | None = None

    for chunk in sorted_chunks:
        if prev_cat is not None:
            blank_count = 2 if chunk.category != prev_cat else 1
            new_body_lines.extend(["\n"] * blank_count)

        new_body_lines.extend(chunk.lines)
        # Guarantee trailing newline
        if new_body_lines and not new_body_lines[-1].endswith("\n"):
            new_body_lines[-1] += "\n"
        prev_cat = chunk.category

    # Splice into the full file.
    # body[0].lineno - 1  → 0-based index of first member line
    # body[-1].end_lineno → 1-based last line  ⟹  slice end (0-based exclusive)
    body_slice_start = body[0].lineno - 1
    body_slice_end = body[-1].end_lineno  # type: ignore[attr-defined]

    new_lines = all_lines[:body_slice_start] + new_body_lines + all_lines[body_slice_end:]
    return "".join(new_lines)


def process_source(source: str) -> str:
    """
    Sort all top-level and nested class bodies in *source*.

    Parameters
    ----------
    source : str
        Python source code.

    Returns
    -------
    str
        Transformed source code.

    """
    # Re-parse after each edit so line numbers stay valid for outer classes.
    # This avoids stale AST coordinates when sorting nested classes changes
    # spacing and shifts later members.
    while True:
        tree = ast.parse(source)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        classes.sort(key=lambda c: c.lineno, reverse=True)

        changed = False
        for cls in classes:
            all_lines = _source_lines(source)
            updated = sort_class_body(source, cls, all_lines)
            if updated != source:
                source = updated
                changed = True
                break

        if not changed:
            break

    return source


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", type=Path, help="Python source files")
    parser.add_argument(
        "--inplace",
        "-i",
        action="store_true",
        help="Edit files in-place (default: print to stdout)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only; exit 1 if any file would be changed",
    )
    args = parser.parse_args()

    any_changed = False

    for path in args.files:
        source = path.read_text(encoding="utf-8")
        try:
            result = process_source(source)
        except SyntaxError as exc:
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            sys.exit(2)

        changed = result != source

        if args.check:
            if changed:
                print(f"would reformat: {path}")
                any_changed = True
            else:
                print(f"already sorted: {path}")
        elif args.inplace:
            if changed:
                path.write_text(result, encoding="utf-8")
                print(f"reformatted: {path}")
            else:
                print(f"unchanged: {path}")
        else:
            sys.stdout.write(result)

    if args.check and any_changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
