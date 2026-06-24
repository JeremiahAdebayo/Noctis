from cache import Cache


class UserService:
    def __init__(self):
        self.cache = Cache()

    def get_user_score(self, user_id: str) -> int:
        """
    Retrieve the user's score, using the cache when possible.
    A cached score of ``0`` is a valid value and must be returned.
    """
        cached = self.cache.get(user_id)

        # ``Cache.get`` returns ``None`` when the key is missing, so we must
        # explicitly check for ``None`` rather than relying on truthiness.
        if cached is not None:
            return cached

        score = self._compute_score(user_id)
        self.cache.set(user_id, score)

        return score

    def _compute_score(self, user_id: str) -> int:
        if user_id.startswith("new"):
            return 0

        return 100