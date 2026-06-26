import secrets


class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, username):
        token = secrets.token_hex(8)
        self.sessions[token] = username
        return token

    def get_username(self, token):
        return self.sessions.get(token)