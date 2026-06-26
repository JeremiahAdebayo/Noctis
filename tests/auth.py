from session import SessionManager


class AuthService:
    def __init__(self, storage):
        self.storage = storage
        self.sessions = SessionManager()

    def login(self, username, password):
        user = self.storage.get_user(username)

        if user is None:
            return None

        if user["password"] != password:
            return None

        token = self.sessions.create_session(username)

        return token

    def authenticate(self, token):
        username = self.sessions.get_username(token)

        if username is None:
            return None

        return self.storage.get_user(username)