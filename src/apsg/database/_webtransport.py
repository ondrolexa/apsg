"""
Dependency-free HTTP transport for the websdb REST API.

Uses only the standard library (``urllib``), mirroring the reference client shipped
with the websdb project itself. This module has no apsg/geology-specific knowledge -
it only knows how to authenticate, build requests, and map HTTP status codes onto a
small exception hierarchy. ``apsg.database._websdbsession.WebSDBSession`` is the layer
that knows websdb's endpoint paths/shapes.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request


class WebsdbError(Exception):
    """Base class for websdb transport/API errors. Plain ``Exception`` subclass, so
    catching it needs no extra dependency (no ``requests``/``httpx`` involved).
    """

    def __init__(self, status, body):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


class WebSDBAuthError(WebsdbError):
    """401 - missing/invalid credentials or an expired token."""


class WebSDBNotFoundError(WebsdbError):
    """404 - resource doesn't exist, or the caller isn't a member of the project
    (websdb deliberately returns 404 rather than 403 for non-members, so it can't be
    told apart from a genuinely missing resource).
    """


class WebSDBPermissionError(WebsdbError):
    """403 - caller is a project member but lacks the required role."""


class WebSDBConflictError(WebsdbError):
    """409 - duplicate name, already-a-member, already-paired, etc."""


class ReadOnlySessionError(Exception):
    """Raised locally, before any HTTP call, when a write method is called on a
    ``WebSDBSession`` opened with ``mode="r"``.
    """


class ProjectResolutionError(Exception):
    """Raised by ``WebSDBSession.__init__`` when ``project=`` (given by name) matches
    zero or more than one of the caller's projects.
    """


_STATUS_EXCEPTIONS = {
    401: WebSDBAuthError,
    403: WebSDBPermissionError,
    404: WebSDBNotFoundError,
    409: WebSDBConflictError,
}


class _WebsdbTransport:
    """Minimal urllib-based HTTP client for the websdb REST API.

    Configuration falls back to the ``WEBSDB_URL``/``WEBSDB_USERNAME``/
    ``WEBSDB_PASSWORD``/``WEBSDB_TOKEN`` environment variables when not passed
    explicitly, matching the websdb project's own reference client.
    """

    def __init__(self, base_url=None, username=None, password=None, token=None):
        self.base_url = (
            base_url or os.environ.get("WEBSDB_URL", "http://localhost:8080")
        ).rstrip("/")
        self.username = username or os.environ.get("WEBSDB_USERNAME")
        self.password = password or os.environ.get("WEBSDB_PASSWORD")
        self.token = token or os.environ.get("WEBSDB_TOKEN")

    def _headers(self):
        if not self.token:
            if not (self.username and self.password):
                raise WebSDBAuthError(
                    401,
                    "No token and no username/password to auto-login with",
                )
            self.login(self.username, self.password)
        return {"Authorization": f"Bearer {self.token}"}

    def request(self, method, path, json_body=None, params=None, auth=True):
        url = self.base_url + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + urllib.parse.urlencode(clean)
        headers = {"Accept": "application/json"}
        if auth:
            headers.update(self._headers())
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        return self._send(req)

    @staticmethod
    def _send(req):
        """Sole network-touching seam - the one thing tests patch."""
        try:
            with urllib.request.urlopen(req) as resp:
                if resp.status == 204:
                    return None
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = raw.decode(errors="replace")
            exc_cls = _STATUS_EXCEPTIONS.get(e.code, WebsdbError)
            raise exc_cls(e.code, body) from None
        except urllib.error.URLError as e:
            raise WebsdbError(0, str(e.reason)) from None

    def login(self, username=None, password=None):
        """``POST /api/login``. Sets ``self.token`` and returns the raw response."""
        result = self.request(
            "POST",
            "/api/login",
            json_body={
                "username": username or self.username,
                "password": password or self.password,
            },
            auth=False,
        )
        self.token = result["access_token"]
        return result
