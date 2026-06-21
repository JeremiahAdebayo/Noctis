from cache import Cache


class UserService:
    def __init__(self):
        self.cache = Cache()

    def get_user_score(self, user_id: str) -> int:
        cached = self.cache.get(user_id)

        if cached:
            return cached

        score = self._compute_score(user_id)
        self.cache.set(user_id, score)

        return score

    def _compute_score(self, user_id: str) -> int:
        if user_id.startswith("new"):
            return 0

        return 100