"""
ファイル管理機能のテスト
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

from src.utils.file_manager import FileManager
from src.models.file_manager import FileMetadata, FilenameConfig, FileExtension

@pytest.fixture
def file_manager(tmp_path):
    """Fixture providing a FileManager instance with a temporary directory."""
    return FileManager(output_dir=str(tmp_path))

@pytest.fixture
def sample_image():
    """Fixture providing a sample image."""
    # 100x100の黒い画像を作成
    return Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

@pytest.fixture
def sample_metadata():
    """Fixture providing sample metadata."""
    return FileMetadata(
        prompt="test prompt",
        steps=30,
        cfg_scale=4.5,
        sampler="euler",
        width=512,
        height=512,
        seed=42,
        custom_tags=["test", "sample"]
    )

def test_generate_filename(file_manager):
    """ファイル名生成のテスト"""
    prompt = "test image generation"
    config = FilenameConfig(
        prefix="test",
        include_date=True,
        include_time=True,
        extension=FileExtension.PNG
    )
    
    filename = file_manager.generate_filename(prompt, config)
    
    # 基本的な検証
    assert filename.endswith(".png")
    assert filename.startswith("test_")
    assert "test_image_generation" in filename

def test_generate_filename_without_date(file_manager):
    """日付なしのファイル名生成テスト"""
    prompt = "test image"
    config = FilenameConfig(
        prefix="test",
        include_date=False,
        include_time=False,
        extension=FileExtension.JPG
    )
    
    filename = file_manager.generate_filename(prompt, config)
    
    # 日付と時刻が含まれていないことを確認
    assert not any(part.isdigit() for part in filename.split("_"))
    assert filename.endswith(".jpg")

def test_save_image_with_metadata(file_manager, sample_image, sample_metadata):
    """画像とメタデータの保存テスト"""
    # 画像とメタデータを保存
    image_path = file_manager.save_image_with_metadata(
        sample_image,
        "test",
        sample_metadata
    )
    
    assert image_path is not None
    assert os.path.exists(image_path)
    
    # メタデータファイルの存在確認
    metadata_path = os.path.splitext(image_path)[0] + ".json"
    assert os.path.exists(metadata_path)
    
    # メタデータの内容を確認
    with open(metadata_path, "r", encoding="utf-8") as f:
        saved_metadata = json.load(f)
    
    # pydanticモデルで検証
    loaded_metadata = FileMetadata.model_validate(saved_metadata)
    assert loaded_metadata.prompt == "test prompt"
    assert loaded_metadata.steps == 30
    assert loaded_metadata.cfg_scale == 4.5
    assert loaded_metadata.seed == 42

def test_invalid_metadata(file_manager, sample_image):
    """無効なメタデータの保存テスト"""
    invalid_metadata = {
        "steps": -1,  # 無効な値
        "cfg_scale": "invalid"  # 無効な型
    }
    
    # 無効なメタデータでの保存は失敗するはず
    with pytest.raises(Exception):
        file_manager.save_image_with_metadata(
            sample_image,
            "test",
            invalid_metadata
        )

def test_sanitize_filename(file_manager):
    """ファイル名の正規化テスト"""
    test_cases = [
        ("hello world", "hello_world"),
        ("test/file*name", "test_file_name"),
        ("  spaces  ", "spaces"),
        ("test__multiple___underscores", "test_multiple_underscores"),
        ("test.txt", "test.txt"),
        ("<>:\"/\\|?*", "_"),
    ]
    
    for input_name, expected in test_cases:
        assert file_manager._sanitize_filename(input_name) == expected

def test_file_extension_validation():
    """ファイル拡張子のバリデーションテスト"""
    # 有効な拡張子
    config = FilenameConfig(extension="png")
    assert config.extension == FileExtension.PNG
    
    # ドット付きの拡張子
    config = FilenameConfig(extension=".jpg")
    assert config.extension == FileExtension.JPG
    
    # 大文字の拡張子
    config = FilenameConfig(extension="PNG")
    assert config.extension == FileExtension.PNG
    
    # 無効な拡張子はデフォルト値(PNG)を使用
    config = FilenameConfig(extension="invalid")
    assert config.extension == FileExtension.PNG 