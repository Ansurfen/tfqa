from flask_sockets import Sockets
from werkzeug.routing import Rule


class TFSockets(Sockets):
    def __init__(self, app=None):
        super().__init__(app)

    def add_url_rule(self, rule, _, f, **options):
        self.url_map.add(Rule(rule, endpoint=f, **options))
