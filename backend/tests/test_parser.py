import pytest
from services.parser import ParserService
from models.schema import AssumptionType

@pytest.fixture(scope="module")
def parser():
    return ParserService()

def test_parse_sentences(parser):
    text = "This is a sentence. And this is another one, clearly."
    sentences = parser.parse(text, min_sentence_chars=5)
    assert len(sentences) == 2
    assert sentences[0].text == "This is a sentence."
    assert sentences[1].text == "And this is another one, clearly."

def test_extract_assumptions(parser):
    text = "Because water is essential to life, we must protect it."
    sentences = parser.parse(text)
    assert len(sentences) == 1
    
    assumptions = parser.extract_assumptions(sentences[0])
    types = [a.type for a in assumptions]
    
    # "Because" is causal
    assert AssumptionType.CAUSAL in types
    # "must" is normative
    assert AssumptionType.NORMATIVE in types

def test_too_short_sentences(parser):
    text = "Hi. This is good. Ok."
    sentences = parser.parse(text, min_sentence_chars=5)
    assert len(sentences) == 1
    assert sentences[0].text == "This is good."
