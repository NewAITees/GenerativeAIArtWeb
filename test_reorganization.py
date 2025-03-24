#!/usr/bin/env python3
"""
GenerativeAIArtWeb テスト再編成スクリプト
テストフォルダの構造を整理し、テストコードを簡素化します。
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
import re
from typing import Dict, List, Set

# プロジェクトのルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.absolute()

# ディレクトリ設定
SOURCE_DIR = PROJECT_ROOT / "test"
TARGET_DIR = PROJECT_ROOT / "tests"
BACKUP_DIR = PROJECT_ROOT / "test_backup"

# 保持するテストモジュール
MODULES_TO_KEEP = {
    "generator/test_sd3_inf.py",
    "prompt/test_llm_generator.py",
    "prompt/test_json_builder.py",
    "web/test_app.py",
}

# 削除するテストモジュール
MODULES_TO_REMOVE = {
    "generator/test_mmditx.py",
    "generator/test_other_impls.py",
    "generator/test_dit_embedder.py",
    "generator/test_sd3_integration.py",
}

def create_backup() -> None:
    """元のテストディレクトリをバックアップ"""
    print("バックアップを作成中...")
    
    for dir_path in [SOURCE_DIR, TARGET_DIR]:
        if dir_path.exists():
            backup_path = BACKUP_DIR / dir_path.name
            if backup_path.exists():
                shutil.rmtree(backup_path)
            print(f"バックアップ作成: {dir_path} -> {backup_path}")
            shutil.copytree(dir_path, backup_path)

def check_test_dependencies() -> Dict[Path, List[str]]:
    """テスト間の依存関係をチェック"""
    dependencies = {}
    
    for test_file in TARGET_DIR.glob('**/*.py'):
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            imports = re.findall(r'from\s+test[s]?\..*\s+import\s+.*', content)
            if imports:
                dependencies[test_file] = imports
    
    return dependencies

def verify_coverage() -> subprocess.CompletedProcess:
    """テストカバレッジが維持されているか確認"""
    return subprocess.run(
        ['pytest', '--cov=src', 'tests'],
        capture_output=True,
        text=True
    )

def migrate_tests() -> None:
    """テストを移行"""
    print("\nテストファイルを移行中...")
    
    # ターゲットディレクトリの作成
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 必要なサブディレクトリの作成
    for subdir in ["generator", "prompt", "utils", "web"]:
        (TARGET_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    # conftest.pyの移行
    if (SOURCE_DIR / "conftest.py").exists():
        shutil.copy2(SOURCE_DIR / "conftest.py", TARGET_DIR / "conftest.py")
        print("conftest.py を移行しました")
    
    # 保持するモジュールの移行
    for module_path in MODULES_TO_KEEP:
        source_file = SOURCE_DIR / module_path
        target_file = TARGET_DIR / module_path
        
        if source_file.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            print(f"移行完了: {module_path}")

def create_init_files() -> None:
    """__init__.pyファイルを作成"""
    print("\n__init__.pyファイルを作成中...")
    
    init_files = [
        TARGET_DIR / "__init__.py",
        TARGET_DIR / "generator" / "__init__.py",
        TARGET_DIR / "prompt" / "__init__.py",
        TARGET_DIR / "utils" / "__init__.py",
        TARGET_DIR / "web" / "__init__.py",
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.write_text('"""テストモジュール"""\n')
            print(f"作成完了: {init_file}")

def create_pytest_ini() -> None:
    """pytest.iniファイルを作成"""
    print("\npytest.iniを作成中...")
    
    content = """[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=src --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
"""
    
    with open(PROJECT_ROOT / "pytest.ini", "w", encoding="utf-8") as f:
        f.write(content)
    print("pytest.ini を作成しました")

def validate_migration() -> bool:
    """移行結果を検証"""
    print("\n移行結果を検証中...")
    
    success = True
    
    # 必要なディレクトリの存在確認
    for subdir in ["generator", "prompt", "utils", "web"]:
        if not (TARGET_DIR / subdir).exists():
            print(f"エラー: {subdir} ディレクトリが存在しません")
            success = False
    
    # 重要なファイルの存在確認
    required_files = [
        "conftest.py",
        "pytest.ini",
        "tests/__init__.py"
    ]
    
    for file_path in required_files:
        if not (PROJECT_ROOT / file_path).exists():
            print(f"エラー: {file_path} が存在しません")
            success = False
    
    # 保持すべきモジュールの確認
    for module_path in MODULES_TO_KEEP:
        if not (TARGET_DIR / module_path).exists():
            print(f"エラー: {module_path} が存在しません")
            success = False
    
    return success

def main() -> None:
    """メイン処理"""
    print("GenerativeAIArtWeb テスト再編成を開始します...")
    
    # バックアップ作成
    create_backup()
    
    # テスト移行
    migrate_tests()
    
    # __init__.pyファイルの作成
    create_init_files()
    
    # pytest.iniの作成
    create_pytest_ini()
    
    # 移行結果の検証
    if validate_migration():
        print("\n✓ テスト再編成が正常に完了しました")
        
        # 元のtest/ディレクトリの削除確認
        if SOURCE_DIR.exists():
            response = input(f"\n{SOURCE_DIR}を削除しますか？ (yes/no): ")
            if response.lower() == "yes":
                shutil.rmtree(SOURCE_DIR)
                print(f"{SOURCE_DIR}を削除しました")
    else:
        print("\n✗ テスト再編成中にエラーが発生しました")
        print("バックアップから復元することをお勧めします")

if __name__ == "__main__":
    main() 