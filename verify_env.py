#!/usr/bin/env python3
"""
verify_env.py
Quick environment verification script for Parkinsons-Voice-Detection.
Run: python verify_env.py
Exits with code 0 when all core imports succeed, non-zero otherwise.
"""

import sys
import platform
import importlib

CORE_MODULES = [
    'numpy',
    'pandas',
    'sklearn',
    'joblib',
    'flask',
    'sounddevice',
    'librosa',
    'parselmouth',
    'scipy'
]


def check_import(module_name):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except Exception as e:
        return False, str(e)


def main():
    print('== Parkinsons-Voice-Detection: Environment Verification ==')
    impl = platform.python_implementation()
    py_ver = platform.python_version()
    arch = platform.machine()
    print(f'Python: {impl} {py_ver} ({arch})')

    # Warn about Parselmouth wheel compatibility
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        print('\nWARNING: The included Parselmouth wheel in this repo is built for CPython 3.10 (cp310).')
        print('If you are not using Python 3.10, Parselmouth may fail to import. Consider using Python 3.10 or a matching wheel.')

    all_ok = True
    print('\nChecking core Python packages...')
    for mod in CORE_MODULES:
        ok, info = check_import(mod)
        if ok:
            print(f'  OK:   {mod} (version: {info})')
        else:
            all_ok = False
            print(f'  FAIL: {mod} -> {info}')

    if all_ok:
        print('\nRESULT: All core imports succeeded. Your environment looks good.')
        print('Tip: run `python app.py` to start the server.')
        return 0

    # Some imports failed
    print('\nRESULT: Some imports failed. Suggested next steps:')
    print('  1) Activate your virtual environment and install pinned dependencies:')
    print('       python -m venv .\\venv')
    print('       .\\venv\\Scripts\\Activate.ps1')
    print('       python -m pip install --upgrade pip wheel')
    print('       pip install -r requirements.txt')
    print('\n  2) If Parselmouth import fails and you are on Python 3.10, install the provided wheel:')
    print('       pip install .\\praat_parselmouth-0.4.6-cp310-cp310-win_amd64.whl')
    print('\n  3) If installation raises build errors (numba, soundfile, etc.), install Microsoft Visual C++ Build Tools or use prebuilt wheels.')
    return 2


if __name__ == '__main__':
    sys.exit(main())
