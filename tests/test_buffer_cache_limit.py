from lmk.engine import BUFFER_CACHE_LIMIT_BYTES, limit_buffer_cache


class FakeMx:
    def __init__(self):
        self.limits = []

    def set_cache_limit(self, n):
        self.limits.append(n)
        return 0


def test_the_buffer_cache_is_capped_at_4_gib():
    mx = FakeMx()
    limit_buffer_cache(mx)
    assert mx.limits == [4 * 1024**3] == [BUFFER_CACHE_LIMIT_BYTES]
