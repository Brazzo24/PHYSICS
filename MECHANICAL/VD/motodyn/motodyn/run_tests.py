"""Lightweight test runner so the suite runs without pytest."""
import sys
import traceback
import importlib.util

spec = importlib.util.spec_from_file_location("test_core", "tests/test_core.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

tests = [(n, getattr(mod, n)) for n in dir(mod) if n.startswith("test_")]
passed = failed = 0
for name, fn in tests:
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()
        failed += 1

print(f"\n{passed}/{len(tests)} passed, {failed} failed")
sys.exit(1 if failed else 0)
