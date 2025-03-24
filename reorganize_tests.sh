#!/bin/bash
# GenerativeAIArtWeb テスト再編成実行スクリプト

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}GenerativeAIArtWeb テスト再編成ツール${NC}"
echo "========================================"

# 作業ディレクトリをプロジェクトルートに変更
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || {
    echo -e "${RED}プロジェクトルートに移動できません${NC}"
    exit 1
}

# 必要なPythonパッケージの確認
echo -e "\n${YELLOW}必要なパッケージを確認中...${NC}"
python -m pip install --quiet pytest pytest-cov

# 開始前の状態確認
echo -e "\n${YELLOW}開始前の状態確認:${NC}"
if [ -d "./test" ]; then
    echo -e "- ${GREEN}test/ ディレクトリが存在します${NC}"
    TEST_FILES=$(find ./test -name "*.py" | wc -l)
    echo "  - Pythonファイル数: $TEST_FILES"
else
    echo -e "- ${RED}test/ ディレクトリが存在しません${NC}"
fi

if [ -d "./tests" ]; then
    echo -e "- ${GREEN}tests/ ディレクトリが存在します${NC}"
    TESTS_FILES=$(find ./tests -name "*.py" | wc -l)
    echo "  - Pythonファイル数: $TESTS_FILES"
else
    echo -e "- ${RED}tests/ ディレクトリが存在しません${NC}"
fi

# 確認
echo -e "\n${YELLOW}警告: このスクリプトはテストディレクトリを再編成します。${NC}"
echo "処理を続行する前に、未コミットの変更をコミットすることをお勧めします。"
read -p "続行しますか？ (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}処理を中止します${NC}"
    exit 1
fi

# テスト再編成スクリプトの実行
echo -e "\n${BLUE}テスト再編成を実行します...${NC}"
python test_reorganization.py

# テスト実行の確認
if [ -d "./tests" ]; then
    echo -e "\n${YELLOW}テストを実行して確認しますか？${NC}"
    read -p "(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}テストを実行しています...${NC}"
        python -m pytest tests -v
    fi
fi

# 終了
echo -e "\n${GREEN}テスト再編成が完了しました！${NC}"
echo "========================================"
echo -e "${BLUE}以下のステップを実行してください:${NC}"
echo "1. テスト実行に問題がないか確認する"
echo "2. 変更をGitにコミットする"
echo "3. テストカバレッジを確認する" 