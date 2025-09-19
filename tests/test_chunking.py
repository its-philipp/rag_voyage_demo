from src.chunking import simple_sentence_split, make_chunks_with_context


def test_simple_sentence_split_basic():
    text = "Hello world. This is a test! Right?"
    parts = simple_sentence_split(text)
    assert parts == ["Hello world.", "This is a test!", "Right?"]


def test_make_chunks_with_context_sliding_window():
    doc = {
        "doc_id": "d1",
        "title": "T",
        "text": "One. Two. Three. Four. Five.",
    }
    chunks = make_chunks_with_context(doc, max_sentences=2, overlap=1)
    # windows: [One,Two], [Two,Three], [Three,Four], [Four,Five]
    assert len(chunks) == 4
    assert chunks[0]["doc_id"] == "d1"
    assert chunks[0]["title"] == "T"
    assert chunks[0]["text"].startswith("One")
