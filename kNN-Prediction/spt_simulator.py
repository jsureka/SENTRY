"""
spt_simulator.py — Pure-Python Semantic-Preserving Transformation (SPT) Simulator
===================================================================================
Simulates the same classes of transformations that CodeImprove's TXL/GA pipeline
applies, without requiring TXL binaries.

Transformations implemented (matching CodeImprove's 7 TXL rules):
  1. rename_vars       — rename local variable identifiers to var_0, var_1, ...
  2. insert_dead_code  — insert unreachable if(0){...} blocks
  3. add_noop          — insert void no-op statements between lines
  4. switch_if_else    — swap if/else branch order with condition negation
  5. add_redundant_cast— add (void*) casts around pointer expressions
  6. wrap_compound     — wrap single-statement if-bodies in braces
  7. remove_comments   — strip // and /* */ comments

Usage:
    from spt_simulator import apply_n_spts, generate_spt_dataset

    # Apply 2 random transformations to a single code string
    mutated = apply_n_spts(code, n=2, seed=42)

    # Generate a full OOD-shifted test set from a jsonl file
    generate_spt_dataset('test.jsonl', 'test_2spt.jsonl', n_transforms=2)
"""

import re
import json
import random
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Individual SPT functions
# ---------------------------------------------------------------------------

def spt_rename_vars(code: str, seed: int = 42) -> str:
    """
    SPT-1: Rename local variable identifiers.
    Finds all identifiers used in assignment LHS (`identifier =`) and renames
    them to var_0, var_1, ... throughout the snippet.
    """
    random.seed(seed)
    # Find all potential variable names: lowercase identifier followed by '='
    pattern = r'\b([a-z_][a-zA-Z0-9_]*)\s*(?:\[.*?\])?\s*='
    candidates = list(dict.fromkeys(re.findall(pattern, code)))  # preserve order, dedupe
    # Exclude common C keywords
    keywords = {
        'int', 'char', 'float', 'double', 'long', 'short', 'unsigned',
        'signed', 'void', 'if', 'else', 'for', 'while', 'do', 'return',
        'break', 'continue', 'switch', 'case', 'default', 'struct',
        'union', 'enum', 'typedef', 'static', 'extern', 'const', 'volatile',
        'sizeof', 'true', 'false', 'null', 'NULL',
    }
    candidates = [v for v in candidates if v not in keywords]
    if not candidates:
        return code
    # Build mapping
    mapping = {v: f'var_{i}' for i, v in enumerate(candidates)}
    for old, new in mapping.items():
        code = re.sub(r'\b' + re.escape(old) + r'\b', new, code)
    return code


def spt_insert_dead_code(code: str) -> str:
    """
    SPT-2: Insert an unreachable if(0){...} block after the first '{'.
    Preserves semantics because the block is never executed.
    """
    dead_block = '\n    if (0) { int __nop__ = 0; (void)__nop__; }\n'
    # Insert after the very first opening brace
    pos = code.find('{')
    if pos == -1:
        return code
    return code[:pos + 1] + dead_block + code[pos + 1:]


def spt_add_noop(code: str) -> str:
    """
    SPT-3: Insert a void cast no-op after several semicolons.
    `int x = 0;` → `int x = 0; (void)0;`
    """
    # Inject after every 3rd semicolon to avoid bloat
    count = [0]
    def replace_semi(m):
        count[0] += 1
        if count[0] % 3 == 0:
            return '; (void)0;'
        return ';'
    return re.sub(r';(?!\s*\n?\s*[\)\}])', replace_semi, code)


def spt_switch_if_else(code: str) -> str:
    """
    SPT-4: Swap if/else branches by negating the condition.
    `if (cond) { A } else { B }` → `if (!(cond)) { B } else { A }`
    Only applies to simple if/else pairs (no else-if chains).
    """
    pattern = re.compile(
        r'if\s*\(([^)]+)\)\s*(\{[^{}]*\})\s*else\s*(\{[^{}]*\})',
        re.DOTALL
    )
    def swap(m):
        cond, then_b, else_b = m.group(1), m.group(2), m.group(3)
        return f'if (!({cond})) {else_b} else {then_b}'
    return pattern.sub(swap, code, count=1)


def spt_add_redundant_cast(code: str) -> str:
    """
    SPT-5: Add (int) redundant casts before integer literal assignments.
    `x = 5;` → `x = (int)(5);`
    """
    pattern = re.compile(r'=\s*(\d+)\s*;')
    def add_cast(m):
        return f'= (int)({m.group(1)});'
    return pattern.sub(add_cast, code, count=3)


def spt_wrap_compound(code: str) -> str:
    """
    SPT-6: Wrap bare (no braces) single-line if-body into compound statement.
    `if (x) stmt;` → `if (x) { stmt; }`
    """
    pattern = re.compile(r'(if\s*\([^)]+\))\s*(?!\{)([^\{\};]+;)', re.DOTALL)
    return pattern.sub(r'\1 { \2 }', code, count=2)


def spt_remove_comments(code: str) -> str:
    """
    SPT-7: Strip C-style comments.
    Removes // line comments and /* block */ comments.
    """
    # Remove block comments (non-greedy)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove line comments
    code = re.sub(r'//[^\n]*', '', code)
    return code


# ---------------------------------------------------------------------------
# Ordered list of all SPTs
# ---------------------------------------------------------------------------
ALL_SPTS = [
    spt_rename_vars,
    spt_insert_dead_code,
    spt_add_noop,
    spt_switch_if_else,
    spt_add_redundant_cast,
    spt_wrap_compound,
    spt_remove_comments,
]
SPT_NAMES = [f.__name__ for f in ALL_SPTS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_n_spts(code: str, n: int, seed: int = 42) -> str:
    """
    Apply n randomly-chosen distinct SPTs to a code string.

    Args:
        code:  Input source code (C/Java string).
        n:     Number of transformations to apply (0 = no change).
        seed:  Random seed for reproducibility.

    Returns:
        Mutated code string.
    """
    if n <= 0:
        return code
    random.seed(seed)
    chosen = random.sample(ALL_SPTS, min(n, len(ALL_SPTS)))
    for transform in chosen:
        try:
            code = transform(code)
        except Exception:
            # Never crash — if a transform fails, skip it
            pass
    return code


def generate_spt_dataset(
    input_path: str,
    output_path: str,
    n_transforms: int,
    seed: int = 42,
    code_field: str = 'input',
    label_field: str = 'label',
    id_field: str = 'id',
) -> int:
    """
    Apply n_transforms SPTs to every sample in a JSONL dataset.

    Args:
        input_path:    Path to input .jsonl (each line: {"id":..,"input":..,"label":..})
        output_path:   Path for output .jsonl
        n_transforms:  Number of SPTs to apply per sample
        seed:          Base random seed (incremented per sample for variety)
        code_field:    JSON field containing source code
        label_field:   JSON field containing label
        id_field:      JSON field containing sample id

    Returns:
        Number of samples processed
    """
    count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            original_code = js.get(code_field, '')
            mutated_code  = apply_n_spts(original_code, n_transforms, seed=seed + i)
            js[code_field] = mutated_code
            js['spt_n']    = n_transforms       # record shift level
            js['spt_seed'] = seed + i
            fout.write(json.dumps(js) + '\n')
            count += 1

    print(f"[SPT] Generated {count} samples with {n_transforms} SPT(s): {output_path}")
    return count


def generate_spt_shift_levels(
    input_path: str,
    output_dir: str,
    max_n: int = 3,
    seed: int = 42,
) -> dict:
    """
    Generate OOD-shifted datasets for n = 0, 1, 2, ..., max_n transformations.
    Creates output_dir/test_N_spt.jsonl for each N.

    Args:
        input_path:  Original test JSONL
        output_dir:  Directory to place shifted datasets
        max_n:       Highest number of SPTs to apply
        seed:        Random seed

    Returns:
        Mapping {n: output_path}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for n in range(0, max_n + 1):
        out_path = str(Path(output_dir) / f'test_{n}_spt.jsonl')
        generate_spt_dataset(input_path, out_path, n_transforms=n, seed=seed)
        paths[n] = out_path
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply SPTs to a JSONL dataset')
    parser.add_argument('--input',      required=True, help='Input .jsonl path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--max_n',      type=int, default=3,
                        help='Max number of SPTs (generates 0 through max_n)')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--code_field', default='input')
    args = parser.parse_args()

    paths = generate_spt_shift_levels(
        args.input, args.output_dir, max_n=args.max_n, seed=args.seed
    )
    print("\n[SPT] All shifted datasets:")
    for n, p in paths.items():
        print(f"  N={n}: {p}")
