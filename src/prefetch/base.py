"""
Base prefetching strategy interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class PrefetchStrategy(ABC):
    """Abstract base class for prefetching strategies."""
    
    @abstractmethod
    def should_prefetch(self, object_id: str, cache_state: dict) -> List[str]:
        """Determine which objects to prefetch."""
        pass
