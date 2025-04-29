import os
import pytest
from pathlib import Path
import json
import tempfile
import shutil
from serve.utils.model_config import ModelConfig 


# @pytest.fixture
# def temp_config_dir():
#     with tempfile.TemporaryDirectory() as tmpdir:
#         yield Path(tmpdir)


@pytest.fixture
def temp_config_dir():
    """Fixture to create and clean up the models directory."""
    models_path = Path(__file__).parent / "modelconfig"
    models_path.mkdir(exist_ok=True)
    yield models_path
    # Clean up after tests
    # if models_path.exists():
    #     shutil.rmtree(models_path)

@pytest.fixture
def model_config(temp_config_dir):
    config_path = temp_config_dir / "config.json"
    print(config_path)
    return ModelConfig(config_path=config_path)

def test_ensure_config_exists(model_config):
    assert model_config.config_path.exists()
    
    with open(model_config.config_path, 'r') as f:
        config = json.load(f)
        assert isinstance(config, dict)

def test_update_model_info(model_config):
    model_name = "test_model"
    run_id = "test_run_123"
    alias = "champion"
    
    model_config.update_model_info(run_id, model_name, alias)
    
    loaded_config = model_config.load_model_config(model_name)
    assert loaded_config.run_id == run_id
    assert loaded_config.model_name == model_name
    assert loaded_config.alias == alias

def test_load_config(model_config):
    model_name = "test_model"
    run_id = "test_run_123"
    alias = "champion"
    
    model_config.update_model_info(run_id, model_name, alias)
    
    config = model_config.load_config()
    assert isinstance(config, dict)
    assert model_name in config
    assert config[model_name].run_id == run_id 