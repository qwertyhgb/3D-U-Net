"""
文件名称：comment_coverage_check.py
文件功能：静态检查注释覆盖率（模块/类/函数文档），确保不低于 90%。
创建日期：2025-11-18
最后修改日期：2025-11-18
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.
"""

import os
import ast

ROOT = os.path.dirname(os.path.dirname(__file__))

def analyze_file(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
    has_module_doc = ast.get_docstring(tree) is not None
    total = 1  # module-level
    ok = 1 if has_module_doc else 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            total += 1
            if ast.get_docstring(node):
                ok += 1
    return ok, total

def main():
    targets = []
    for base in ['src', 'scripts']:
        d = os.path.join(ROOT, base)
        for r, _, files in os.walk(d):
            for fn in files:
                if fn.endswith('.py'):
                    targets.append(os.path.join(r, fn))
    ok_sum, total_sum = 0, 0
    for p in targets:
        ok, total = analyze_file(p)
        ok_sum += ok
        total_sum += total
        print(f'{os.path.relpath(p, ROOT)}: {ok}/{total} ({ok/total*100:.1f}%)')
    cov = ok_sum / max(total_sum, 1)
    print(f'Total coverage: {cov*100:.1f}%')
    assert cov >= 0.90, '注释覆盖率低于 90%'

if __name__ == '__main__':
    main()