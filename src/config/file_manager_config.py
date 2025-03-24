"""
ファイル管理のための設定モジュール
"""
import os
import stat
from pathlib import Path

# プロジェクトのルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# ファイル権限設定
FILE_PERMISSIONS = {
    'directory': stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,  # 755
    'file': stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH  # 644
}

# 画像設定
IMAGE_CONFIG = {
    'extensions': {'.png', '.jpg', '.jpeg', '.gif', '.bmp'},
    'default_format': 'PNG'
}

# ディレクトリ構造設定
DIRECTORY_STRUCTURE = {
    'root': 'outputs',
    'subdirs': ['images', 'metadata', 'presets']
}

# メタデータ設定
METADATA_CONFIG = {
    'extension': '.json',
    'encoding': 'utf-8',
    'temp_suffix': '.tmp'
}

# ファイル名設定
FILENAME_CONFIG = {
    'max_length': 100,
    'date_format': '%Y%m%d',
    'time_format': '%H%M%S',
    'separator': '_',
    'invalid_chars': r'[<>:"/\\|?*]'
}

# エラーメッセージ
ERROR_MESSAGES = {
    'permission_denied': "アクセス権限がありません: {path}",
    'file_not_found': "ファイルが見つかりません: {path}",
    'directory_creation_error': "ディレクトリの作成に失敗しました: {path}"
}

# ログメッセージ
LOG_MESSAGES = {
    'init_success': "FileManagerを初期化しました: {path}",
    'save_success': "ファイルを保存しました: {path}",
    'save_metadata_success': "メタデータを保存しました: {path}"
}