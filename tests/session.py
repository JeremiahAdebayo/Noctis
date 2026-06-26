import secrets
class SessionManager:
    """Manages user sessions with token-to-username mapping.

    The session store is shared across all instances of ``SessionManager`` so that
    tokens created during a login persist for subsequent API calls that instantiate
    a new ``SessionManager`` (e.g., via ``AuthService``)."""

    # Class‑level dictionary that holds token → username mappings for all instances.
    _sessions: dict = {}

    def __init__(self):
        # No per‑instance state is required; the shared ``_sessions`` dict is used.
        pass

    def create_session(self, username):
        """Create a new session token for *username* and store it.

        Returns the generated token string.
        """
        token = secrets.token_hex(8)
        # Store the token in the shared session dictionary.
        self.__class__._sessions[token] = username
        return token

    def get_username(self, token):
        """Retrieve the username associated with *token*.

        Returns ``None`` if the token is not found.
        """
        return self.__class__._sessions.get(token)