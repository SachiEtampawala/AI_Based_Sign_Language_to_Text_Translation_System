class WordBuffer:
    """Turns a stream of per-frame sign predictions into a list of words.

    A word is added only when the same sign is held for `stable_frames`
    frames in a row. The same word is not added again until the signer
    changes to a different sign, or the hand leaves the camera view.
    """

    def __init__(self, stable_frames=10, max_words=30):
        self.stable_frames = stable_frames
        self.max_words = max_words
        self.words = []
        self._candidate = None   # sign currently being held
        self._count = 0          # how many frames in a row
        self._locked = None      # last word added (blocks repeats)

    def update(self, sign):
        """Call once per frame with the current sign.

        Use sign=None when the sign is uncertain.
        Returns the new word if one was just added, otherwise None.
        """
        if sign is None:
            self._candidate = None
            self._count = 0
            return None

        if sign == self._candidate:
            self._count += 1
        else:
            self._candidate = sign
            self._count = 1

        if self._count >= self.stable_frames and sign != self._locked:
            self.words.append(sign)
            self._locked = sign

            if len(self.words) > self.max_words:
                self.words.pop(0)

            return sign

        return None

    def hand_lost(self):
        """Call when no hand is visible. Allows the same word again."""
        self._candidate = None
        self._count = 0
        self._locked = None

    @property
    def progress(self):
        """How close the current sign is to being accepted (0.0 to 1.0)."""
        if self._candidate is None or self._candidate == self._locked:
            return 0.0
        return min(self._count / self.stable_frames, 1.0)

    def remove_last(self):
        if self.words:
            self.words.pop()
        self._locked = None

    def clear(self):
        self.words.clear()
        self._candidate = None
        self._count = 0
        self._locked = None

    @property
    def sentence(self):
        return " ".join(self.words)