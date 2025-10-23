from abc import ABC, abstractmethod
from typing import Dict

class Backend(ABC):
    @abstractmethod
    def codegen(self, loop_source, metadata: Dict) -> str:
        """Emit target code as a string."""
        pass
    
    @staticmethod
    def create_backend(name: str, val_map: Dict = None):
        """Factory method to create backend instances based on the name."""
        if name == "triton":
            from .triton.backend import TritonBackend
            return TritonBackend(val_map)
        elif name == "ptx":
            from .ptx.backend import PTXBackend
            return PTXBackend(val_map)
        else:
            raise ValueError(f"Unknown backend: {name}")
