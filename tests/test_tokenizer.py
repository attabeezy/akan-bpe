import pytest
from pathlib import Path
from somax.tokenizer import DualCoreTokenizer
from transformers import PreTrainedTokenizerFast

@pytest.fixture
def mock_tokenizer_file(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    
    # Train on some dummy data to make it functional
    tokenizer.train_from_iterator(["hello world", "this is a test"], trainer=trainer)
    
    tokenizer_path = tmp_path / "unified_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path

def test_dual_core_tokenizer_init(mock_tokenizer_file):
    tokenizer = DualCoreTokenizer(tokenizer_path=mock_tokenizer_file, language="akan")
    assert isinstance(tokenizer._tokenizer, PreTrainedTokenizerFast)
    assert tokenizer.router.language == "akan"

def test_dual_core_tokenizer_classify(mock_tokenizer_file):
    tokenizer = DualCoreTokenizer(tokenizer_path=mock_tokenizer_file)
    assert tokenizer.classify("uhm chale") == "robust"
    assert tokenizer.classify("This is a formal and long sentence for logic.") == "logic"

def test_dual_core_tokenizer_encode(mock_tokenizer_file):
    tokenizer = DualCoreTokenizer(tokenizer_path=mock_tokenizer_file)
    # The Dummy BPE will just split on whitespace and find 'hello' if it's there
    # But for a dummy, it might just return [UNK] for everything except our vocab
    tokens = tokenizer.encode("hello world")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
