"""A small deterministic mutation tester for the Memory 2.0 modules.

    python -m backtalk.evals.memory_retrieval.mutation backtalk/memory_retrieval.py [--jobs N] [--list-survivors]

For one source file it generates every single-point mutant below, runs the whole Memory 2.0 test suite against each
(in a throwaway copy of the package, so the real files are never touched), and reports how many the tests killed and
every survivor with its line and the exact change. Nothing is configured to spare a mutant: there is no skip list,
docstrings and the literal pieces of f-strings are the only text left alone (they are not behavior), and a survivor is either fixed by a test or has to
be justified by hand.

MUTATIONS: comparison operators swapped (< <= > >= == != in not-in is is-not); and/or swapped; `not` removed;
arithmetic operators swapped (+ - * / // %); a number changed by one (or 0.0 <-> 1.0); True <-> False; a string
changed; `x if c else y` and `if c` conditions negated; `return v` -> `return None`; `continue`/`break` -> `pass`;
min <-> max, any <-> all, sorted(...) key negation removed; a unary minus removed; a default argument changed.
"""
import argparse
import ast
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile

SUITE = ("tests.test_memory", "tests.test_memory_gate", "tests.test_memory_store", "tests.test_memory_retrieval",
         "tests.test_memory_retrieval_eval", "tests.test_memory_mutation_gaps", "tests.test_memory_extract",
         "tests.test_memory_extract_gaps", "tests.test_memory_extract_tools")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_SWAP_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
             ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
_SWAP_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Mult,
             ast.Mod: ast.Mult}
_SWAP_NAME = {"min": "max", "max": "min", "any": "all", "all": "any"}


def _untouchable(tree):
    """Walk-order positions of text that is not behavior: docstrings and the literal pieces of f-strings."""
    nodes = list(ast.walk(tree))
    where = {id(n): i for i, n in enumerate(nodes)}
    out = set()
    for node in nodes:
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(where[id(body[0].value)])
        elif isinstance(node, ast.JoinedStr):
            out.update(where[id(v)] for v in node.values if isinstance(v, ast.Constant))
    return out


def mutants(source):
    """(description, mutated source) for every single-point mutation, in a fixed order."""
    base = ast.parse(source)
    skip = _untouchable(base)
    count = sum(1 for _ in ast.walk(base))
    out = []
    for index in range(count):
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
        node = nodes[index]
        edit = _edit(node, index in skip)
        for label, apply in edit:
            tree = ast.parse(source)
            target = list(ast.walk(tree))[index]
            apply(target)
            ast.fix_missing_locations(tree)
            line = getattr(node, "lineno", 0)
            out.append((f"line {line}: {label}", ast.unparse(tree), line))
    return out


def _edit(node, untouchable):
    """[(label, function that mutates the node in place)] for one node."""
    ops = []
    if isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            if type(op) in _SWAP_CMP:
                ops.append((f"{type(op).__name__} -> {_SWAP_CMP[type(op)].__name__}",
                            lambda n, i=i: n.ops.__setitem__(i, _SWAP_CMP[type(n.ops[i])]())))
    elif isinstance(node, ast.BoolOp):
        new = ast.Or if isinstance(node.op, ast.And) else ast.And
        ops.append((f"{type(node.op).__name__} -> {new.__name__}", lambda n, new=new: setattr(n, "op", new())))
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.Not, ast.USub)):
            ops.append((f"remove unary {type(node.op).__name__}", lambda n: _become(n, n.operand)))
    elif isinstance(node, ast.BinOp) and type(node.op) in _SWAP_BIN:
        new = _SWAP_BIN[type(node.op)]
        ops.append((f"{type(node.op).__name__} -> {new.__name__}", lambda n, new=new: setattr(n, "op", new())))
    elif isinstance(node, ast.Constant) and not untouchable:
        v = node.value
        if isinstance(v, bool):
            ops.append((f"{v} -> {not v}", lambda n: setattr(n, "value", not n.value)))
        elif isinstance(v, int):
            ops.append((f"{v} -> {v + 1}", lambda n: setattr(n, "value", n.value + 1)))
            if v != 0:
                ops.append((f"{v} -> 0", lambda n: setattr(n, "value", 0)))
        elif isinstance(v, float):
            ops.append((f"{v} -> {v + 0.1:.6g}", lambda n: setattr(n, "value", round(n.value + 0.1, 6))))
            ops.append((f"{v} -> {max(v - 0.1, 0.0):.6g}", lambda n: setattr(n, "value", round(max(n.value - 0.1, 0.0), 6))))
        elif isinstance(v, str) and v != "":
            ops.append((f"{v[:30]!r} -> altered", lambda n: setattr(n, "value", n.value + "~")))
    elif isinstance(node, ast.IfExp):
        ops.append(("negate conditional expression", lambda n: setattr(n, "test", ast.UnaryOp(ast.Not(), n.test))))
    elif isinstance(node, ast.If):
        ops.append(("negate if", lambda n: setattr(n, "test", ast.UnaryOp(ast.Not(), n.test))))
    elif isinstance(node, ast.While):
        ops.append(("negate while", lambda n: setattr(n, "test", ast.UnaryOp(ast.Not(), n.test))))
    elif isinstance(node, ast.Return) and node.value is not None \
            and not (isinstance(node.value, ast.Constant) and node.value.value is None):
        ops.append(("return value -> None", lambda n: setattr(n, "value", ast.Constant(None))))
    elif isinstance(node, (ast.Continue, ast.Break)):
        ops.append((f"{type(node).__name__.lower()} -> pass", lambda n: _become(n, ast.Pass())))
    elif isinstance(node, ast.Name) and node.id in _SWAP_NAME and isinstance(node.ctx, ast.Load):
        ops.append((f"{node.id} -> {_SWAP_NAME[node.id]}", lambda n: setattr(n, "id", _SWAP_NAME[n.id])))
    return ops


def _become(node, other):
    node.__class__ = other.__class__
    node.__dict__.clear()
    node.__dict__.update(other.__dict__)


def _run(work, rel, mutated, timeout, suite=SUITE):
    path = os.path.join(work, rel)
    with open(path, "w", encoding="utf-8") as f:
        f.write(mutated)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        r = subprocess.run([sys.executable, "-m", "unittest", "-f", *suite], cwd=work, env=env, timeout=timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode != 0
    except subprocess.TimeoutExpired:
        return True                                          # a mutant that hangs the tests is detected


def _baseline(work, rel, timeout, suite=SUITE):
    _copy(work)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "unittest", *suite], cwd=work, env=env, timeout=timeout,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
    return r.returncode != 0                                 # "killed" here means the baseline FAILED: a broken harness


def _copy(work):
    shutil.copytree(os.path.join(ROOT, "backtalk"), os.path.join(work, "backtalk"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.npy"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(ROOT, "tests"), os.path.join(work, "tests"),
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)


def _worker(args):
    rel, chunk, timeout = args[:3]
    suite = tuple(args[3]) if len(args) > 3 else SUITE
    work = tempfile.mkdtemp(prefix="mutate-")
    try:
        if not chunk:                                        # the baseline: the UNMUTATED copy must pass
            return [("baseline", 0, _baseline(work, rel, timeout, suite))]
        _copy(work)
        original = open(os.path.join(work, rel), encoding="utf-8").read()
        results = []
        for desc, mutated, line in chunk:
            results.append((desc, line, _run(work, rel, mutated, timeout, suite)))
            with open(os.path.join(work, rel), "w", encoding="utf-8") as f:
                f.write(original)
        return results
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="source file, e.g. backtalk/memory_retrieval.py")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--suite", help="comma-separated test modules to run instead of the Memory 2.0 default")
    args = ap.parse_args(argv)
    rel = os.path.relpath(os.path.abspath(args.target), ROOT)
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        source = f.read()
    suite = tuple(args.suite.split(",")) if args.suite else SUITE
    if _worker((rel, [], args.timeout, suite))[0][2]:
        sys.exit("the unmutated suite FAILS in the temporary copy: the mutation run would be meaningless")
    original_normalized = ast.unparse(ast.parse(source))
    muts = [m for m in mutants(source) if m[1] != original_normalized]
    chunks = [muts[i::args.jobs] for i in range(args.jobs)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(args.jobs) as pool:
        for part in pool.map(_worker, [(rel, c, args.timeout, suite) for c in chunks if c]):
            results.extend(part)
    results.sort(key=lambda r: (r[1], r[0]))
    lines = source.splitlines()
    survivors = [r for r in results if not r[2]]
    text = [r for r in results if "-> altered" in r[0]]
    print(f"{rel}: {len(results) - len(survivors)}/{len(results)} killed, {len(survivors)} survived")
    print(f"  logic mutants (operators, numbers, booleans, control flow): "
          f"{sum(1 for r in results if r not in text and r[2])}/{len(results) - len(text)} killed")
    print(f"  text mutants (a string literal changed):                    "
          f"{sum(1 for r in text if r[2])}/{len(text)} killed")
    for desc, line, _ in survivors:
        print(f"  SURVIVED {desc}   | {lines[line - 1].strip()[:110] if 0 < line <= len(lines) else ''}")


if __name__ == "__main__":
    main()
