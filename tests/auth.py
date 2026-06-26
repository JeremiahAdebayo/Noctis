from session import SessionManager


class AuthService:
    def __init__(self, storage):
        self.storage = storage
        self.sessions = SessionManager()

    def login(self, username, password):
        """Authenticate a user and create a session token.

    The original implementation relied solely on ``SessionManager.create_session``
    to persist the mapping between the generated token and the username.  In the
    current ``SessionManager`` implementation the ``create_session`` method does
    **not** store this mapping – it only returns a token.  Consequently, subsequent
    calls to ``authenticate`` (which uses ``SessionManager.get_username``) cannot
    resolve the token to a username, resulting in a ``401 Unauthorized`` response.

    This method now:
    1. Validates the supplied credentials.
    2. Generates a token via ``create_session``.
    3. Explicitly stores the ``token -> username`` relationship in the
       ``SessionManager``'s internal dictionary (named ``sessions`` or ``_sessions``
       depending on the implementation).
    4. Returns the token.
    """
        user = self.storage.get_user(username)
        if user is None or user.get("password") != password:
            return None

        # Generate a token using the SessionManager.
        token = self.sessions.create_session(username)

        # Persist the token‑username mapping. ``SessionManager`` stores its data in a
        # dictionary attribute.  The attribute name may be ``sessions`` (as used in
        # ``get_username``) or ``_sessions`` (a common private‑name convention).  We
        # handle both possibilities to remain robust.
        if hasattr(self.sessions, "sessions"):
            self.sessions.sessions[token] = username
        elif hasattr(self.sessions, "_sessions"):
            self.sessions._sessions[token] = username
        else:
            # Fallback: if the manager provides a ``set`` method, use it.
            try:
                self.sessions.set(token, username)  # type: ignore[attr-defined]
            except Exception:
                # If no storage mechanism is available, raise a clear error.
                raise RuntimeError("SessionManager does not expose a mutable session store")

        return token

    def authenticate(self, token):
        username = self.sessions.get_username(token)

        if username is None:
            return None

        return self.storage.get_user(username)