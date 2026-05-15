"""
Build script for packaging the Angular Velocity Demo into a standalone .exe.
Uses PyInstaller to create a single-file executable with all dependencies.

Usage:
    python build.py              - Build in current directory (dist/)
    python build.py --clean      - Clean previous build artifacts first

Output:
    dist/AngularVelocity.exe   - Standalone executable (Windows)
"""

import os
import sys
import subprocess
import shutil


def check_pyinstaller():
    """Check if PyInstaller is installed, install if not."""
    try:
        import PyInstaller
        return True
    except ImportError:
        print("[!] PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True


def clean_artifacts():
    """Remove previous build and dist artifacts."""
    dirs_to_clean = ["build", "dist"]
    files_to_clean = ["driver.spec"]
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"[+] Cleaning: {d}/")
            shutil.rmtree(d)
    
    for f in files_to_clean:
        if os.path.exists(f):
            print(f"[+] Cleaning: {f}")
            os.remove(f)


def build_exe():
    """Run PyInstaller to create the standalone .exe."""
    
    # PyInstaller options
    #   --onefile:     Create a single bundled file
    #   --windowed:    No console window (pygame app)
    #   --name:        Output file name
    #   --icon:        Optional: path to .ico file
    #   --add-data:    Include additional files (config.yaml)
    #   --hidden-import: Ensure all imports are included
    #   --exclude-module: Exclude unnecessary modules to reduce size
    
    pyinstaller_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "AngularVelocity",
        "--add-data", f"config.yaml;.",      # Add config.yaml to the bundle root
        "--hidden-import", "pygame",
        "--hidden-import", "numpy",
        "--hidden-import", "yaml",
        "--hidden-import", "jaraco",
        "--hidden-import", "jaraco.text",
        "--collect-all", "setuptools",
        "--exclude-module", "tkinter",
        "--exclude-module", "unittest",
        "--exclude-module", "pytest",
        "driver.py",
    ]
    
    print("\n" + "=" * 60)
    print("  Building AngularVelocity.exe with PyInstaller")
    print("=" * 60 + "\n")
    
    result = subprocess.run(pyinstaller_cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("  BUILD SUCCESS!")
        print("=" * 60)
        print(f"\nExecutable location: dist/AngularVelocity.exe")
        print(f"File size: {get_file_size('dist/AngularVelocity.exe')}")
        print("\nYou can distribute 'dist/AngularVelocity.exe' as a standalone application.")
        print("No Python installation required on the target machine.")
        print("=" * 60)
    else:
        print("\n[!] Build failed. Check the output above for errors.")
        sys.exit(1)


def get_file_size(filepath):
    """Get file size as a human-readable string."""
    if not os.path.exists(filepath):
        return "N/A"
    size_bytes = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build standalone .exe for Angular Velocity Demo")
    parser.add_argument("--clean", action="store_true", help="Clean previous build artifacts")
    args = parser.parse_args()
    
    # Step 1: Check/install PyInstaller
    check_pyinstaller()
    
    # Step 2: Clean if requested
    if args.clean:
        clean_artifacts()
    
    # Step 3: Build
    build_exe()


if __name__ == "__main__":
    main()