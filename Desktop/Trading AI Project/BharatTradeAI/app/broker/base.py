from abc import ABC, abstractmethod


class BrokerBase(ABC):
    provider_name = "base"

    @abstractmethod
    def login(self):
        raise NotImplementedError

    @abstractmethod
    def get_access_token(self):
        raise NotImplementedError

    @abstractmethod
    def get_profile(self):
        raise NotImplementedError

    @abstractmethod
    def get_funds(self):
        raise NotImplementedError

    @abstractmethod
    def get_holdings(self):
        raise NotImplementedError

    @abstractmethod
    def get_positions(self):
        raise NotImplementedError

    @abstractmethod
    def get_orders(self):
        raise NotImplementedError

    @abstractmethod
    def get_instruments(self):
        raise NotImplementedError
