from abc import ABC, abstractmethod
from typing import Dict

class Backend(ABC):
    @abstractmethod
    def codegen(self, loop_source, metadata: Dict) -> str:
        """Emit target code as a string."""
        pass
