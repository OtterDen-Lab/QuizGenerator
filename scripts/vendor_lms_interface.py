#!/usr/bin/env python3
"""
Script to vendor LMSInterface into QuizGenerator for PyPI release.

This script:
1. Copies lms_interface code from ../LMSInterface into QuizGenerator/canvas/
2. Updates imports to use the vendored version
3. Extracts version info for tracking
4. Updates pyproject.toml dependencies

Usage:
    python scripts/vendor_lms_interface.py [--dry-run] [--lms-path PATH]
"""

import argparse
import re
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime


def get_lms_version(lms_path: Path) -> str:
    """Extract version from LMSInterface pyproject.toml"""
    pyproject = lms_path / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"

    content = pyproject.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else "unknown"


def copy_lms_files(lms_source: Path, target_dir: Path, dry_run: bool = False):
    """Copy LMSInterface files to QuizGenerator/canvas/"""
    files_to_copy = [
        "__init__.py",
        "canvas_interface.py",
        "classes.py",
    ]

    print(f"\n📦 Copying files from {lms_source} to {target_dir}")

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    for filename in files_to_copy:
        source = lms_source / filename
        dest = target_dir / filename

        if not source.exists():
            print(f"  ❌ Warning: {source} not found, skipping")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would copy: {source} → {dest}")
        else:
            shutil.copy2(source, dest)
            print(f"  ✅ Copied: {filename}")

    return True


def update_canvas_init(target_dir: Path, version: str, dry_run: bool = False):
    """Add version info to canvas/__init__.py"""
    init_file = target_dir / "__init__.py"

    version_header = f'''"""
Canvas LMS integration for QuizGenerator.

Vendored from LMSInterface v{version} ({datetime.now().strftime('%Y-%m-%d')})

This module provides Canvas API integration for uploading quizzes
and managing course content.
"""

__version__ = "{version}"
__vendored_from__ = "LMSInterface"
__vendored_date__ = "{datetime.now().strftime('%Y-%m-%d')}"

'''

    print(f"\n📝 Updating {init_file.name} with version info")

    if dry_run:
        print(f"  [DRY RUN] Would add version header:")
        print(f"    Version: {version}")
        print(f"    Date: {datetime.now().strftime('%Y-%m-%d')}")
    else:
        # Read existing content (if any)
        existing_content = ""
        if init_file.exists():
            existing_content = init_file.read_text()
            # Remove old version header if it exists
            existing_content = re.sub(
                r'^""".*?""".*?__vendored_date__.*?\n\n',
                '',
                existing_content,
                flags=re.DOTALL | re.MULTILINE
            )

        # Write new version header + existing content
        init_file.write_text(version_header + existing_content.lstrip())
        print(f"  ✅ Updated with version {version}")

    return True


def update_generate_quiz_imports(target_dir: Path, dry_run: bool = False):
    """Update imports in generate_quiz.py"""
    generate_quiz = target_dir / "generate.py"

    if not generate_quiz.exists():
        print(f"  ❌ Warning: {generate_quiz} not found")
        return False

    print(f"\n📝 Updating imports in generate.py")

    content = generate_quiz.read_text()

    # Check if already using vendored import
    if "from QuizGenerator.canvas.canvas_interface import" in content:
        print("  ℹ️  Already using vendored imports, no changes needed")
        return True

    old_import = "from lms_interface.canvas_interface import CanvasInterface, CanvasCourse"
    new_import = "from QuizGenerator.canvas.canvas_interface import CanvasInterface, CanvasCourse"

    if old_import not in content:
        print(f"  ⚠️  Warning: Expected import not found. Current imports:")
        for line in content.split('\n')[:20]:
            if 'import' in line.lower() and ('canvas' in line.lower() or 'lms' in line.lower()):
                print(f"    {line}")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would replace:")
        print(f"    OLD: {old_import}")
        print(f"    NEW: {new_import}")
    else:
        new_content = content.replace(old_import, new_import)
        generate_quiz.write_text(new_content)
        print(f"  ✅ Updated import statement")

    return True


def update_pyproject_toml(repo_root: Path, lms_path: Path, dry_run: bool = False):
    """Update pyproject.toml to use vendored dependencies"""
    pyproject = repo_root / "pyproject.toml"

    if not pyproject.exists():
        print(f"  ❌ Error: {pyproject} not found")
        return False

    print(f"\n📝 Updating pyproject.toml")

    content = pyproject.read_text()

    # Get dependencies from LMSInterface
    lms_pyproject = lms_path / "pyproject.toml"
    lms_deps = []
    if lms_pyproject.exists():
        lms_content = lms_pyproject.read_text()
        # Extract dependencies between [project] and next section
        in_deps = False
        for line in lms_content.split('\n'):
            if 'dependencies = [' in line:
                in_deps = True
                continue
            if in_deps:
                if line.strip().startswith(']'):
                    break
                if line.strip() and not line.strip().startswith('#'):
                    # Extract the dependency
                    dep = line.strip().strip(',').strip('"')
                    if dep:
                        lms_deps.append(dep)

    print(f"  ℹ️  Found LMSInterface dependencies: {lms_deps}")

    changes = []

    # 1. Remove lms-interface dependency
    if '"lms-interface"' in content or "'lms-interface'" in content:
        changes.append("Remove lms-interface from dependencies")
        if not dry_run:
            content = re.sub(r'\s*"lms-interface"[,\s]*\n', '', content)
            content = re.sub(r"\s*'lms-interface'[,\s]*\n", '', content)

    # 2. Check if canvasapi is already in dependencies
    has_canvasapi = 'canvasapi' in content
    if not has_canvasapi:
        changes.append("Add canvasapi dependency")
        # We'll note this but won't auto-add (requires finding right place in deps list)

    # 3. Remove [tool.uv.sources] section if it exists
    if '[tool.uv.sources]' in content:
        changes.append("Remove [tool.uv.sources] section")
        if not dry_run:
            # Remove the section and its lms-interface line
            content = re.sub(
                r'\[tool\.uv\.sources\]\s*\nlms-interface\s*=\s*\{[^}]+\}\s*\n',
                '',
                content
            )

    if dry_run:
        print(f"  [DRY RUN] Would make the following changes:")
        for change in changes:
            print(f"    - {change}")
        if not has_canvasapi:
            print(f"  ⚠️  Note: You'll need to manually add these dependencies:")
            for dep in lms_deps:
                print(f"      {dep}")
    else:
        if changes:
            pyproject.write_text(content)
            print(f"  ✅ Made {len(changes)} change(s)")
            for change in changes:
                print(f"    - {change}")

        if not has_canvasapi:
            print(f"\n  ⚠️  ACTION REQUIRED: Add these dependencies manually:")
            for dep in lms_deps:
                print(f"      {dep}")

    return True


def verify_structure(repo_root: Path, dry_run: bool = False):
    """Verify the vendored structure is correct"""
    print(f"\n🔍 Verifying structure...")

    required_files = [
        "QuizGenerator/canvas/__init__.py",
        "QuizGenerator/canvas/canvas_interface.py",
        "QuizGenerator/canvas/classes.py",
        "QuizGenerator/generate.py",
        "pyproject.toml",
    ]

    all_good = True
    for file_path in required_files:
        full_path = repo_root / file_path
        if dry_run:
            print(f"  [DRY RUN] Would check: {file_path}")
        else:
            if full_path.exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ Missing: {file_path}")
                all_good = False

    return all_good


def main():
    parser = argparse.ArgumentParser(
        description="Vendor LMSInterface into QuizGenerator for PyPI release"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--lms-path",
        type=Path,
        help="Path to LMSInterface repository (default: ../LMSInterface)"
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    lms_path = args.lms_path or (repo_root.parent / "LMSInterface")

    print("=" * 60)
    print("🔧 QuizGenerator LMSInterface Vendoring Script")
    print("=" * 60)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")

    print(f"\nPaths:")
    print(f"  QuizGenerator: {repo_root}")
    print(f"  LMSInterface:  {lms_path}")

    # Verify LMSInterface exists
    lms_source = lms_path / "lms_interface"
    if not lms_source.exists():
        print(f"\n❌ Error: LMSInterface not found at {lms_path}")
        print(f"   Looked for: {lms_source}")
        print(f"\nUse --lms-path to specify a different location")
        return 1

    # Get version
    version = get_lms_version(lms_path)
    print(f"\n📌 LMSInterface version: {version}")

    # Execute vendoring steps
    target_dir = repo_root / "QuizGenerator" / "canvas"

    success = True
    success &= copy_lms_files(lms_source, target_dir, args.dry_run)
    success &= update_canvas_init(target_dir, version, args.dry_run)
    success &= update_generate_quiz_imports((repo_root / "QuizGenerator"), args.dry_run)
    success &= update_pyproject_toml(repo_root, lms_path, args.dry_run)

    if not args.dry_run:
        success &= verify_structure(repo_root, args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("✅ Dry run complete! Run without --dry-run to apply changes")
    elif success:
        print("✅ Vendoring complete!")
        print("\nNext steps:")
        print("  1. Review changes: git diff")
        print("  2. Check dependencies in pyproject.toml")
        print("  3. Test: uv sync && python QuizGenerator/generate.py --help")
        print("  4. Commit: git add QuizGenerator/canvas/ QuizGenerator/generate.py pyproject.toml")
    else:
        print("⚠️  Completed with warnings - please review output above")
        return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())