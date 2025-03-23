"""
設定管理機能のテスト
"""

import os
import json
import pytest
from pathlib import Path
from src.utils.settings import SettingsManager
from src.models.settings import AppSettings, ImageGenerationSettings, WatermarkSettings

@pytest.fixture
def sample_settings():
    """Fixture providing a sample settings dictionary."""
    # pydanticモデルを使用してサンプル設定を作成
    settings = AppSettings(
        generation=ImageGenerationSettings(
            prompt="a cat in a garden",
            steps=30,
            cfg_scale=4.5,
            width=1024,
            height=1024,
            seed=42,
            sampler="euler"
        ),
        watermark=WatermarkSettings(
            enabled=True,
            text="Generated with AI",
            opacity=0.3
        )
    )
    return settings.model_dump()

@pytest.fixture
def settings_manager(tmp_path):
    """Fixture providing a SettingsManager instance with a temporary directory."""
    return SettingsManager(settings_dir=tmp_path)

def test_save_profile(settings_manager, sample_settings):
    """プロファイル保存のテスト"""
    # プロファイルを保存
    result = settings_manager.save_profile("test_profile", sample_settings)
    assert result is True
    
    # ファイルが作成されたことを確認
    profile_path = settings_manager.settings_dir / "test_profile.json"
    assert profile_path.exists()
    
    # 保存された内容を確認
    with open(profile_path, "r", encoding="utf-8") as f:
        saved_settings = json.load(f)
    
    # pydanticモデルで検証
    app_settings = AppSettings.model_validate(saved_settings)
    assert app_settings.generation.prompt == "a cat in a garden"
    assert app_settings.generation.steps == 30
    assert app_settings.watermark.enabled is True

def test_load_profile(settings_manager, sample_settings):
    """プロファイル読み込みのテスト"""
    # プロファイルを保存
    settings_manager.save_profile("test_profile", sample_settings)
    
    # プロファイルを読み込み
    loaded_settings = settings_manager.load_profile("test_profile")
    assert loaded_settings is not None
    
    # pydanticモデルで検証
    app_settings = AppSettings.model_validate(loaded_settings)
    assert app_settings.generation.prompt == "a cat in a garden"
    assert app_settings.generation.steps == 30
    assert app_settings.watermark.enabled is True

def test_load_nonexistent_profile(settings_manager):
    """存在しないプロファイルの読み込みテスト"""
    result = settings_manager.load_profile("nonexistent")
    assert result is None

def test_get_default_settings(settings_manager):
    """デフォルト設定取得のテスト"""
    default_settings = settings_manager.get_default_settings()
    
    # pydanticモデルで検証
    app_settings = AppSettings.model_validate(default_settings)
    assert app_settings.generation.steps == 40  # デフォルト値
    assert app_settings.generation.cfg_scale == 4.5  # デフォルト値
    assert app_settings.watermark.enabled is False  # デフォルト値

def test_invalid_settings(settings_manager):
    """無効な設定の保存テスト"""
    invalid_settings = {
        "generation": {
            "steps": -1,  # 無効な値
            "cfg_scale": 100.0  # 無効な値
        }
    }
    
    # 無効な設定の保存は失敗するはず
    result = settings_manager.save_profile("invalid", invalid_settings)
    assert result is False 