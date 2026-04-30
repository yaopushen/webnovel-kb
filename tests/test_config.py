import pytest


def test_package_import():
    """Test that the package can be imported."""
    import webnovel_kb
    assert webnovel_kb is not None


def test_config_module():
    """Test that config module can be imported and has expected attributes."""
    from webnovel_kb import config
    assert hasattr(config, 'DATA_DIR')
    assert hasattr(config, 'CHUNK_SIZE')
    assert hasattr(config, 'LLM_API_KEY')


def test_data_models_import():
    """Test that data_models module can be imported."""
    from webnovel_kb import data_models
    assert data_models is not None


def test_chunk_size_value():
    """Test that CHUNK_SIZE has a reasonable value."""
    from webnovel_kb.config import CHUNK_SIZE
    assert isinstance(CHUNK_SIZE, int)
    assert CHUNK_SIZE > 0
    assert CHUNK_SIZE < 10000


def test_chunk_overlap_value():
    """Test that CHUNK_OVERLAP has a reasonable value."""
    from webnovel_kb.config import CHUNK_OVERLAP
    assert isinstance(CHUNK_OVERLAP, int)
    assert CHUNK_OVERLAP >= 0
