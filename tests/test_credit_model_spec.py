from src.credit_model_spec import credit_model_spec_table, validation_boundary_table, model_spec_markdown


def test_credit_model_spec_has_six_channels():
    spec = credit_model_spec_table()
    assert len(spec) == 6
    assert set(spec["status"]) == {"Observed", "Proxy"}
    assert abs(spec["weight"].sum() - 1.0) < 1e-9


def test_validation_boundary_and_markdown():
    boundary = validation_boundary_table()
    assert list(boundary.columns) == ["Layer", "Scope", "Interpretation"]
    text = model_spec_markdown()
    assert "Credit Model Spec" in text
    assert "Production decision score" in text
