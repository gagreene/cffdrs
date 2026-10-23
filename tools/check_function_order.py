"""Check the function-order style rule (exit status 1 when a module breaks it).

Rule: the functions of a module (and the methods of a class or ``impl`` block) are in alphabetical order, case
insensitive. Python: names starting with an underscore first (dunders lead), then public names; same-named
consecutive definitions (a property and its setter) count once. Rust: functions without ``pub`` first, then ``pub``
functions. Test files: helpers first in the same order, then the tests in groups that start with a
``# ---- <label> ----`` (Rust: ``// ---- <label> ----``) comment; the labels are alphabetical (``other tests`` last)
and the tests keep their order inside a group.

Usage: python tools/check_function_order.py [path ...]   (default: src tools tests rust/crates)
"""
import ast
import re
import sys
from pathlib import Path

SKIP = {'.venv', 'target', '__pycache__', '.git', 'node_modules', 'build', 'dist'}
LABEL = re.compile(r'^\s*(?:#|//) ---- (.*) ----\s*$', re.M)

FN_RE = re.compile(r'^(?P<vis>pub(?:\([^)]*\))?\s+)?'
                   r'(?:(?:const|async|unsafe|extern(?:\s+"[^"]*")?)\s+)*fn\s+(?P<name>\w+)')


def body_span(chunk):
    """Offsets of the text strictly inside the outermost braces of an item (first '{' to its matching '}')."""
    depth = 0
    i = 0
    n = len(chunk)
    opening = None
    while i < n:
        c = chunk[i]
        if chunk.startswith('//', i):
            j = chunk.find('\n', i)
            i = n if j < 0 else j + 1
            continue
        if chunk.startswith('/*', i):
            j = chunk.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        m = re.match(r'b?r(#*)"', chunk[i:i + 12])
        if m and (i == 0 or not (chunk[i - 1].isalnum() or chunk[i - 1] == '_')):
            close = '"' + m.group(1)
            j = chunk.find(close, i + len(m.group(0)))
            i = n if j < 0 else j + len(close)
            continue
        if c == '"':
            i += 1
            while i < n and chunk[i] != '"':
                i += 2 if chunk[i] == '\\' else 1
            i += 1
            continue
        if c == "'":
            m = re.match(r"'(\\.[^']*|[^\\'])'", chunk[i:i + 12])
            i += len(m.group(0)) if m else 1
            continue
        if c in '{':
            if opening is None and depth == 0:
                opening = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and opening is not None:
                return opening + 1, i
        i += 1
    return None


def check_py_body(nodes, scope, is_test, problems, path):
    names = []
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not names or names[-1] != n.name):
            names.append(n.name)
    if is_test:
        names = [n for n in names if not n.startswith('test')]
    expected = sorted(names, key=py_key)
    if names != expected:
        first = next(a for a, b in zip(names, expected, strict=True) if a != b)
        problems.append(f'{path}: {scope}: functions out of order near {first!r} '
                        f'(expected {expected[names.index(first)]!r})')


def check_python(path, problems):
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    is_test = path.name.startswith('test_')
    check_py_body(tree.body, '<module>', is_test, problems, path.as_posix())
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            check_py_body(node.body, node.name, False, problems, path.as_posix())
    if is_test:
        labels_in_order(text, problems, path.as_posix())


def check_rust(path, problems):
    text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    header = re.match(r'((?://![^\n]*\n|\s*\n)*)', text)
    check_rust_items(text[header.end():], path.as_posix(), problems)
    labels_in_order(text, problems, path.as_posix())


def check_rust_items(text, where, problems):
    spans = scan_items(text)
    infos = [classify(text[a:b]) for a, b in spans]
    fns = [(i[1], i[2], i[3]) for i in infos if i[0] == 'fn']
    helpers = [f for f in fns if not f[2]] if any(f[2] for f in fns) else fns
    keys = [sort_key(n, pub) for n, pub, _t in helpers]
    if keys != sorted(keys):
        problems.append(f'{where}: functions out of order: {[n for n, _p, _t in helpers][:6]}')
    for (a, b), info in zip(spans, infos, strict=True):
        code = info[1] or ''
        if info[0] == 'other' and (re.match(r'(unsafe\s+)?impl\b', code)
                                   or any(x.startswith('#[cfg(test)]') for x in info[4])):
            span = body_span(text[a:b])
            if span:
                check_rust_items(text[a:b][span[0]:span[1]], where + ' > ' + code[:40], problems)


def classify(chunk):
    """('fn', name, is_pub, is_test) or ('other', None, False, False) for one item's text."""
    lines = chunk.split('\n')
    attrs = []
    code = None
    k = 0
    while k < len(lines):
        s = lines[k].strip()
        if s.startswith('#['):
            attrs.append(s)
            # multi-line attribute
            while attrs[-1].count('[') > attrs[-1].count(']') and k + 1 < len(lines):
                k += 1
                attrs[-1] += lines[k].strip()
        elif s and not s.startswith('//'):
            code = s
            break
        k += 1
    if code is None:
        return 'other', None, False, False, attrs
    m = FN_RE.match(code)
    if m:
        return 'fn', m.group('name'), bool(m.group('vis')), any(a.startswith('#[test') for a in attrs), attrs
    return 'other', code, False, False, attrs


def labels_in_order(text, problems, where):
    labels = [x for x in LABEL.findall(text) if x != 'helpers']
    if labels != sorted(labels, key=lambda s: (s == 'other tests', s)):
        problems.append(f'{where}: test group labels are not alphabetical: {labels[:6]}')


def main(argv):
    roots = [Path(a) for a in argv] or [Path(p) for p in ('src', 'tools', 'tests', 'rust/crates') if Path(p).exists()]
    problems = []
    for root in roots:
        for path in sorted(root.rglob('*') if root.is_dir() else [root]):
            if SKIP & set(path.parts) or path.suffix not in ('.py', '.rs'):
                continue
            (check_python if path.suffix == '.py' else check_rust)(path, problems)
    for problem in problems:
        print(problem)
    print(f'{len(problems)} function-order problem(s)')
    return 1 if problems else 0


def py_key(name):
    return (0 if name.startswith('_') else 1, name.lower(), name)


def scan_items(text):
    """Split ``text`` into top-level items: list of (start, end) offsets (end exclusive), leading trivia included."""
    items = []
    n = len(text)
    i = 0
    start = 0
    depth = 0
    item_has_code = False
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j + 1
            continue
        if text.startswith('/*', i):
            level = 1
            i += 2
            while i < n and level:
                if text.startswith('/*', i):
                    level += 1
                    i += 2
                elif text.startswith('*/', i):
                    level -= 1
                    i += 2
                else:
                    i += 1
            continue
        m = re.match(r'b?r(#*)"', text[i:i + 12])
        if m and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')):
            close = '"' + m.group(1)
            j = text.find(close, i + len(m.group(0)))
            i = n if j < 0 else j + len(close)
            item_has_code = True
            continue
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == '\\' else 1
            i += 1
            item_has_code = True
            continue
        if c == "'":
            m = re.match(r"'(\\.[^']*|[^\\'])'", text[i:i + 12])
            if m:
                i += len(m.group(0))
                item_has_code = True
                continue
            i += 1
            continue
        if c in '([{':
            depth += 1
            item_has_code = True
        elif c in ')]}':
            depth -= 1
            if depth == 0 and c == '}':
                k = i + 1
                while k < n and text[k] in ' \t':
                    k += 1
                if k < n and text[k] == ';':
                    i = k
                elif not text[k:k + 1] or True:
                    # end of a braced item (consume the rest of the line)
                    j = text.find('\n', i)
                    i = n if j < 0 else j + 1
                    items.append((start, i))
                    start = i
                    item_has_code = False
                    continue
        if c == ';' and depth == 0 and item_has_code:
            j = text.find('\n', i)
            i = n if j < 0 else j + 1
            items.append((start, i))
            start = i
            item_has_code = False
            continue
        if not c.isspace():
            item_has_code = True
        i += 1
    if start < n and text[start:].strip():
        items.append((start, n))
    return items


def sort_key(name, is_pub):
    return (1 if is_pub else 0, name.lower(), name)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
