from abc import ABC, abstractmethod

from nemo_gym.orchestration.api import SubmitConfig


class BaseExecutor(ABC):
    @abstractmethod
    def run(self, config: SubmitConfig) -> None: ...
