from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests


class BrokerClient(ABC):
    provider_name = "broker"
    base_url = ""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        redirect_uri: str = "",
        code: str = "",
        access_token: str = "",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.redirect_uri = redirect_uri
        self.code = code
        self.access_token = access_token

    def login(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "not_configured", "provider": self.provider_name}
        return {
            "status": "configured",
            "provider": self.provider_name,
            "message": f"{self.provider_name.title()} credentials loaded. Complete OAuth flow to get a valid session.",
            "auth_url": self.build_auth_url(),
        }

    def build_auth_url(self) -> str:
        if not self.api_key:
            raise ValueError("api_key is required to build auth URL")
        params = {
            "client_id": self.api_key,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "" if not self.redirect_uri else "default",
        }
        return f"{self.auth_url_base}?{urlencode(params)}"

    def get_access_token(self) -> Dict[str, Any]:
        if not self.access_token:
            return {
                "status": "not_available",
                "provider": self.provider_name,
                "message": "Set the access token in settings or complete the broker auth flow.",
            }
        return {
            "status": "ok",
            "provider": self.provider_name,
            "access_token": self.access_token,
        }

    def exchange_code_for_token(self) -> Dict[str, Any]:
        if not self.api_key or not self.api_secret or not self.code:
            return {
                "status": "not_configured",
                "provider": self.provider_name,
                "message": "api_key, api_secret, and code are required for token exchange.",
            }

        payload = {
            "grant_type": "authorization_code",
            "code": self.code,
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "redirect_uri": self.redirect_uri,
        }

        try:
            response = requests.post(self.token_url, data=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            self.access_token = data.get("access_token")
            return {
                "status": "ok",
                "provider": self.provider_name,
                "access_token": self.access_token,
                "data": data,
            }
        except requests.RequestException as exc:
            return {
                "status": "error",
                "provider": self.provider_name,
                "message": str(exc),
            }

    def _session(self) -> requests.Session:
        session = requests.Session()
        headers = {"Accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        session.headers.update(headers)
        return session

    def _request_json(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.access_token:
            return {
                "provider": self.provider_name,
                "status": "not_connected",
                "message": "A valid access token is required to call this endpoint.",
            }

        try:
            url = f"{self.base_url}{endpoint}"
            response = self._session().request(method=method, url=url, params=params, json=payload, timeout=20)
            response.raise_for_status()
            return {
                "provider": self.provider_name,
                "status": "ok",
                "data": response.json(),
            }
        except requests.RequestException as exc:
            return {
                "provider": self.provider_name,
                "status": "error",
                "message": str(exc),
            }

    auth_url_base = ""
    token_url = ""

    @abstractmethod
    def get_profile(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_funds(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_holdings(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_orders(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_instruments(self) -> Dict[str, Any]:
        raise NotImplementedError
