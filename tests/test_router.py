import pytest
from somax.router import WAXALRouter

def test_router_heuristic_fallback():
    # Test without a trained model file
    router = WAXALRouter(language="akan", model_dir="non_existent_dir")
    
    # "robust" markers
    assert router.classify("uhm chale me dwo") == "robust"
    assert router.classify("err okay") == "robust"
    assert router.classify("naa") == "robust"
    
    # "robust" short text
    assert router.classify("Hello") == "robust"
    
    # "logic" formal/long text
    formal_text = "The quick brown fox jumps over the lazy dog in the forest."
    assert router.classify(formal_text) == "logic"

def test_router_initialization():
    router = WAXALRouter(language="akan")
    assert router.language == "akan"
    assert not router.using_classifier # Should be False unless model trained
