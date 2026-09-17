from spiral.edits import EditProgressHint


def test_streamed_header_keeps_exact_filename_across_every_chunk_boundary():
    text = 'src/components/[id]/long_component.rs\n<<<<<<< SEARCH\n' + 'payload\n' * 600
    for size in (1, 2, 7, 31, 399, 2050):
        hint = EditProgressHint()
        paths = [value for start in range(0, len(text), size)
                 if (value := hint.feed(text[start:start + size]))]
        if size <= 399:
            assert paths == ['src/components/[id]/long_component.rs']
        else:
            assert set(paths) <= {'src/components/[id]/long_component.rs'}
        assert len(hint.tail) <= 2048


def test_prose_or_incomplete_header_does_not_claim_an_edit():
    hint = EditProgressHint()
    assert hint.feed('I may inspect notes.md or use code.py.\n') is None
    assert hint.feed('src/index.ts\n<<<<') is None
    assert hint.feed('<<< SEARCH\n') == 'src/index.ts'
    assert hint.feed('old code\n=======\nnew code\n>>>>>>> REPLACE\n') is None
    assert hint.feed('test/run.sh\n<<<<<<< SEARCH\n') == 'test/run.sh'
