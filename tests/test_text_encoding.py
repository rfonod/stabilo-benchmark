"""Every text file the package and its tests read or write names its encoding."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted([*ROOT.glob('stabilo_benchmark/**/*.py'), *ROOT.glob('tests/**/*.py')])


def unspecified(tree: ast.AST):
    """Line numbers of text-mode `read_text`, `write_text` and `open` calls without an `encoding`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or any(k.arg in ('encoding', None) for k in node.keywords):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name == 'read_text' and not node.args:
            yield node.lineno
        elif name == 'write_text' and len(node.args) == 1:
            yield node.lineno
        elif name == 'open' and isinstance(func, ast.Name):
            mode = (
                node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == 'mode'), None)
            )
            if mode is None or (isinstance(mode, ast.Constant) and 'b' not in mode.value):
                yield node.lineno


def test_text_io_names_its_encoding():
    # Without one, Python uses the locale's encoding, cp1252 on Windows, which cannot read UTF-8.
    offenders = {
        str(path.relative_to(ROOT)): lines
        for path in SOURCES
        if (lines := list(unspecified(ast.parse(path.read_text(encoding='utf-8')))))
    }
    assert len(SOURCES) > 50
    assert offenders == {}
