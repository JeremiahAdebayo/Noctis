from auth import AuthService
from storage import Storage


class API:
    def __init__(self):
        self.storage = Storage()

    def login(self, username, password):
        auth = AuthService(self.storage)
        return auth.login(username, password)

    def profile(self, token):
        auth = AuthService(self.storage)

        user = auth.authenticate(token)

        if user is None:
            return {
                "status": 401
            }

        return {
            "status": 200,
            "email": user["email"]
        }