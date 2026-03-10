"""Tests for group_class_members.py.

These are the test cases exercised during development, collected into a
single file using pytest conventions.
"""

import ast
import subprocess
import sys
import textwrap

import pytest

from group_class_members.main import process_source


def sort(source: str) -> str:
    """Run the tool on *source* and return stdout."""
    return process_source(textwrap.dedent(source))


def check(source: str) -> bool:
    """Return True if the source would be reformatted."""
    return sort(source) != textwrap.dedent(source)


class TestBasicOrdering:
    """Members end up in the canonical category order."""

    SOURCE = """\
        from __future__ import annotations
        from typing import ClassVar


        class Animal:
            def speak(self) -> str:
                return f"{self.name} says {self._sound}"

            def _make_sound(self) -> str:
                return "..."

            @property
            def sound(self) -> str:
                return self._sound

            @sound.setter
            def sound(self, value: str) -> None:
                self._sound = value

            name: str
            _sound: str
            count: ClassVar[int] = 0
            species: ClassVar[str]

            def __repr__(self) -> str:
                return f"Animal({self.name!r})"

            def __init__(self, name: str, sound: str) -> None:
                self.name = name
                self._sound = sound
                Animal.count += 1

            @classmethod
            def from_dict(cls, data: dict) -> "Animal":
                return cls(data["name"], data["sound"])

            @staticmethod
            def valid_name(name: str) -> bool:
                return bool(name.strip())

            def __private(self) -> None:
                pass

            def train(self) -> None:
                pass

            def __len__(self) -> int:
                return len(self.name)
        """

    def test_classvars_come_first(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        class_start = next(i for i, l in enumerate(lines) if l.startswith("class Animal"))
        body_lines = [l.strip() for l in lines[class_start + 1 :] if l.strip()]
        # First non-empty body line should be a ClassVar annotation
        assert "ClassVar" in body_lines[0]

    def test_init_before_other_methods(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        init_idx = next(i for i, l in enumerate(lines) if "def __init__" in l)
        speak_idx = next(i for i, l in enumerate(lines) if "def speak" in l)
        assert init_idx < speak_idx

    def test_property_before_public_method(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        prop_idx = next(i for i, l in enumerate(lines) if "@property" in l)
        speak_idx = next(i for i, l in enumerate(lines) if "def speak" in l)
        assert prop_idx < speak_idx

    def test_classmethod_before_property(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        cm_idx = next(i for i, l in enumerate(lines) if "@classmethod" in l)
        prop_idx = next(i for i, l in enumerate(lines) if "@property" in l)
        assert cm_idx < prop_idx

    def test_private_method_after_public(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        pub_idx = next(i for i, l in enumerate(lines) if "def speak" in l)
        priv_idx = next(i for i, l in enumerate(lines) if "def _make_sound" in l)
        assert pub_idx < priv_idx

    def test_mangled_method_last(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        priv_idx = next(i for i, l in enumerate(lines) if "def _make_sound" in l)
        mangle_idx = next(i for i, l in enumerate(lines) if "def __private" in l)
        assert priv_idx < mangle_idx

    def test_dunder_before_classmethod(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        repr_idx = next(i for i, l in enumerate(lines) if "def __repr__" in l)
        cm_idx = next(i for i, l in enumerate(lines) if "@classmethod" in l)
        assert repr_idx < cm_idx

    def test_output_is_valid_python(self):
        result = sort(self.SOURCE)
        ast.parse(result)  # raises SyntaxError if broken


class TestAttributeDocstrings:
    """Attribute docstrings (bare string literals after AnnAssign) travel
    with their owning attribute."""

    SOURCE = """\
        from __future__ import annotations
        from typing import ClassVar


        class Animal:
            def speak(self) -> str:
                return f"{self.name} says {self._sound}"

            name: str
            \"\"\"The animal's name.\"\"\"

            _sound: str
            \"\"\"Internal sound representation.\"\"\"

            count: ClassVar[int] = 0
            \"\"\"Total number of animals created.\"\"\"

            species: ClassVar[str]

            def __init__(self, name: str, sound: str) -> None:
                self.name = name
                self._sound = sound
        """

    def test_docstring_follows_attribute(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        for i, line in enumerate(lines):
            if "name: str" in line and "sound" not in line:
                # next non-empty line should be its docstring
                rest = [l.strip() for l in lines[i + 1 :] if l.strip()]
                assert rest[0].startswith('"""The animal'), (
                    f"Expected docstring after 'name: str', got: {rest[0]!r}"
                )
                break

    def test_count_docstring_follows_count(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        for i, line in enumerate(lines):
            if "count: ClassVar[int]" in line:
                rest = [l.strip() for l in lines[i + 1 :] if l.strip()]
                assert "Total number" in rest[0]
                break

    def test_no_orphan_docstrings_at_top(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        class_start = next(i for i, l in enumerate(lines) if l.startswith("class Animal"))
        # First non-empty body line must NOT be a bare string
        body = [l.strip() for l in lines[class_start + 1 :] if l.strip()]
        assert not body[0].startswith('"""'), f"Orphan docstring at top of class body: {body[0]!r}"

    def test_output_is_valid_python(self):
        result = sort(self.SOURCE)
        ast.parse(result)


class TestBlankLines:
    """Canonical blank-line spacing is enforced."""

    SOURCE = """\
        class Foo:
            def b(self) -> None: pass
            x: int
            def _c(self) -> None: pass
            def a(self) -> None: pass
        """

    def test_two_blank_lines_between_categories(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        # Find transition from instance-var block to init/method block
        # There should be at least one pair of consecutive blank lines
        blank_runs = []
        run = 0
        for line in lines:
            if line.strip() == "":
                run += 1
            else:
                if run:
                    blank_runs.append(run)
                run = 0
        assert max(blank_runs) >= 2, "Expected at least one 2-blank-line gap between categories"

    def test_one_blank_line_within_category(self):
        source = textwrap.dedent("""\
            class Foo:
                def a(self) -> None: pass
                def b(self) -> None: pass
                def c(self) -> None: pass
            """)
        result = sort(source)
        lines = result.splitlines()
        blank_runs = []
        run = 0
        for line in lines:
            if line.strip() == "":
                run += 1
            else:
                if run:
                    blank_runs.append(run)
                run = 0
        assert all(r == 1 for r in blank_runs), (
            f"Expected only single blank lines within a category, got runs: {blank_runs}"
        )

    def test_no_leading_blank_lines_in_body(self):
        result = sort(self.SOURCE)
        lines = result.splitlines()
        class_start = next(i for i, l in enumerate(lines) if l.startswith("class Foo"))
        first_body = lines[class_start + 1]
        assert first_body.strip() != "", "Body must not start with a blank line"


class TestIdempotency:
    """Running the tool twice produces the same result as running it once."""

    SOURCES = [
        # Basic unsorted class
        """\
            from typing import ClassVar
            class A:
                def foo(self): pass
                x: int
                X: ClassVar[int] = 0
                def __init__(self): pass
            """,
        # Class with attribute docstrings
        """\
            class B:
                def bar(self): pass
                name: str
                \"\"\"Name docstring.\"\"\"
                def __init__(self): pass
            """,
        # Already sorted — should be unchanged
        """\
            class C:
                x: int
                def __init__(self): pass
                def foo(self): pass
                def _bar(self): pass
            """,
    ]

    @pytest.mark.parametrize("source", SOURCES)
    def test_idempotent(self, source):
        first = sort(source)
        second = sort(first)
        assert first == second, "Tool is not idempotent"


class TestAlreadySorted:
    """Already-sorted classes are returned unchanged."""

    def test_already_sorted_unchanged(self):
        source = textwrap.dedent("""\
            from typing import ClassVar

            class Animal:
                count: ClassVar[int] = 0
                name: str

                def __init__(self, name: str) -> None:
                    self.name = name

                def __repr__(self) -> str:
                    return f"Animal({self.name!r})"

                @classmethod
                def create(cls, name: str) -> "Animal":
                    return cls(name)

                @property
                def upper(self) -> str:
                    return self.name.upper()

                def speak(self) -> None:
                    pass

                def _helper(self) -> None:
                    pass
            """)
        assert not check(source), "Already-sorted class should not be flagged"


class TestLeadingComments:
    """# comments immediately before a member travel with it."""

    def test_comment_travels_with_member(self):
        source = textwrap.dedent("""\
            class Foo:
                def bar(self): pass
                # This comment belongs to baz
                def baz(self): pass
            """)
        result = sort(source)
        lines = result.splitlines()
        comment_idx = next(i for i, l in enumerate(lines) if "This comment belongs to baz" in l)
        baz_idx = next(i for i, l in enumerate(lines) if "def baz" in l)
        assert baz_idx == comment_idx + 1, (
            "Comment should be immediately before its method after sorting"
        )


class TestNestedClasses:
    """Nested classes are sorted independently."""

    def test_nested_class_sorted(self):
        source = textwrap.dedent("""\
            class Outer:
                class Inner:
                    def foo(self): pass
                    x: int
                    def __init__(self): pass

                def outer_method(self): pass
                y: int
            """)

        result = sort(source)
        ast.parse(result)
        lines = result.splitlines()
        # Inner class: x: int should come before __init__
        x_idx = next(i for i, l in enumerate(lines) if "x: int" in l)
        init_idx = next(i for i, l in enumerate(lines) if "def __init__" in l)
        assert x_idx < init_idx


class TestSubclass:
    """Subclasses are sorted independently of their parent."""

    def test_subclass_sorted(self):
        source = textwrap.dedent("""\
            from __future__ import annotations
            from typing import ClassVar


            class Animal:
                def speak(self) -> str:
                    return f"{self.name} says {self._sound}"

                name: str
                _sound: str
                count: ClassVar[int] = 0
                species: ClassVar[str]

                def __init__(self, name: str, sound: str) -> None:
                    self.name = name
                    self._sound = sound
                    Animal.count += 1


            class Dog(Animal):
                def bark(self) -> str:
                    return "Woof!"

                breed: str

                def __init__(self, name: str, breed: str) -> None:
                    super().__init__(name, "Woof")
                    self.breed = breed
            """)

        result = sort(source)
        ast.parse(result)
        lines = result.splitlines()
        # Dog: breed: str should come before __init__
        breed_idx = next(i for i, l in enumerate(lines) if "breed: str" in l)
        init_idx = next(
            i
            for i, l in enumerate(lines)
            if "def __init__" in l
            and lines[i - 1 : i + 2]
            and "super()" in "\n".join(lines[i : i + 3])
        )
        assert breed_idx < init_idx


class TestCLI:
    """Command-line interface behaves correctly."""

    def _run(self, args: list[str], source: str) -> subprocess.CompletedProcess:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(textwrap.dedent(source))
            name = f.name
        try:
            return subprocess.run(
                [sys.executable, "-m", "group_class_members.main", *args, name],
                capture_output=True,
                text=True,
            )
        finally:
            os.unlink(name)

    def test_check_exits_1_when_unsorted(self):
        proc = self._run(
            ["--check"],
            """\
            class Foo:
                def bar(self): pass
                x: int
            """,
        )
        assert proc.returncode == 1

    def test_check_exits_0_when_sorted(self):
        proc = self._run(
            ["--check"],
            """\
            class Foo:
                x: int
                def bar(self): pass
            """,
        )
        assert proc.returncode == 0

    def test_stdout_output(self):
        proc = self._run(
            [],
            """\
            class Foo:
                def bar(self): pass
                x: int
            """,
        )
        assert proc.returncode == 0
        # Output should have x: int before def bar
        lines = proc.stdout.splitlines()
        x_idx = next(i for i, l in enumerate(lines) if "x: int" in l)
        bar_idx = next(i for i, l in enumerate(lines) if "def bar" in l)
        assert x_idx < bar_idx
