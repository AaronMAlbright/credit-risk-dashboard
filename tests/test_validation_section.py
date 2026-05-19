from src.validation_section import render_validation_section


def test_validation_section_imports():
    assert callable(render_validation_section)
