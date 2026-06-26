class Storage:
    def __init__(self):
        self.users = {
            "alice": {
                "password": "1234",
                "email": "alice@test.com",
            }
        }

    def get_user(self, username):
        return self.users.get(username)