import pytest

from modelgym.utils import ModelSpace, process_model_spaces
from modelgym.models import LGBMClassifier, RFClassifier

def test_process_model_spaces():
    rf_model_space = ModelSpace(RFClassifier, name="RF")
    processed = process_model_spaces([LGBMClassifier, rf_model_space])
    assert len(processed) == 2
    assert all(map(lambda ms: isinstance(ms, ModelSpace), processed.values()))
    assert all(map(lambda kv: kv[0] == kv[1].name, processed.items()))

    assert processed["LGBMClassifier"].model_class == LGBMClassifier
    assert processed["RF"] == rf_model_space

def test_process_model_spaces_unwrap():
    processed = process_model_spaces(LGBMClassifier)
    assert isinstance(processed, dict)
    assert len(processed) == 1
    assert all(map(lambda ms: isinstance(ms, ModelSpace), processed.values()))
    assert all(map(lambda kv: kv[0] == kv[1].name, processed.items()))

    assert processed["LGBMClassifier"].model_class == LGBMClassifier

def test_process_model_space_value_errors():
    with pytest.raises(ValueError):
        process_model_spaces([RFClassifier()])
    with pytest.raises(ValueError):
        process_model_spaces(LGBMClassifier())
    with pytest.raises(ValueError):
        process_model_spaces([LGBMClassifier, LGBMClassifier])

