"""
Test runner script for the CDN cache simulator.

Provides easy way to run tests with coverage reporting and
various test configurations.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """Run all tests with coverage reporting."""
    print("🧪 Running CDN Cache Simulator Test Suite")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Run pytest with coverage
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "-v",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        print(f"📊 Coverage report generated in htmlcov/index.html")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False


def run_specific_test(test_file: str):
    """Run a specific test file."""
    print(f"🧪 Running {test_file}")
    print("=" * 30)
    
    cmd = ["python", "-m", "pytest", f"tests/{test_file}", "-v"]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ {test_file} passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {test_file} failed with exit code {e.returncode}")
        return False


def run_fast_tests():
    """Run tests without coverage (faster)."""
    print("🧪 Running Fast Test Suite")
    print("=" * 30)
    
    cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Fast tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Fast tests failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "fast":
            success = run_fast_tests()
        elif command == "specific" and len(sys.argv) > 2:
            test_file = sys.argv[2]
            success = run_specific_test(test_file)
        else:
            print("Usage: python test_runner.py [fast|specific <test_file>]")
            sys.exit(1)
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
