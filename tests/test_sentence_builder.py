import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentence_builder import WordBuffer


def feed(buffer, sign, frames):
    added = []
    for _ in range(frames):
        word = buffer.update(sign)
        if word:
            added.append(word)
    return added


def test_word_added_after_stable_frames():
    buffer = WordBuffer(stable_frames=5)
    assert feed(buffer, "play", 4) == []
    assert feed(buffer, "play", 1) == ["play"]
    assert buffer.words == ["play"]


def test_same_word_not_repeated_while_held():
    buffer = WordBuffer(stable_frames=3)
    feed(buffer, "man", 20)
    assert buffer.words == ["man"]


def test_flickering_sign_is_not_added():
    buffer = WordBuffer(stable_frames=5)
    for sign in ["who", "tell", "who", "tell", "who", "tell"]:
        buffer.update(sign)
    assert buffer.words == []


def test_same_word_allowed_after_hand_lost():
    buffer = WordBuffer(stable_frames=3)
    feed(buffer, "city", 3)
    buffer.hand_lost()
    feed(buffer, "city", 3)
    assert buffer.words == ["city", "city"]


def test_different_signs_build_a_sentence():
    buffer = WordBuffer(stable_frames=3)
    feed(buffer, "who", 3)
    feed(buffer, "play", 3)
    feed(buffer, "orange", 3)
    assert buffer.sentence == "who play orange"


def test_uncertain_sign_does_not_unlock_repeat():
    buffer = WordBuffer(stable_frames=3)
    feed(buffer, "man", 3)
    buffer.update(None)
    feed(buffer, "man", 5)
    assert buffer.words == ["man"]


def test_remove_last_and_clear():
    buffer = WordBuffer(stable_frames=2)
    feed(buffer, "who", 2)
    feed(buffer, "tell", 2)
    buffer.remove_last()
    assert buffer.words == ["who"]
    buffer.clear()
    assert buffer.words == []
    assert buffer.sentence == ""


def test_progress_goes_up_to_one():
    buffer = WordBuffer(stable_frames=4)
    buffer.update("but")
    assert buffer.progress == 0.25
    feed(buffer, "but", 3)
    assert buffer.progress == 0.0   # accepted, so bar resets