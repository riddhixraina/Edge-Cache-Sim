"""
Static cache partitioning strategy.
"""

from typing import List, Dict, Any


class StaticPartitioner:
    """Static equal partitioning of cache capacity across nodes."""
    
    def __init__(self, total_capacity: int, num_nodes: int):
        self.total_capacity = total_capacity
        self.num_nodes = num_nodes
    
    def partition(self) -> List[int]:
        """Return capacity allocation for each node."""
        capacity_per_node = self.total_capacity // self.num_nodes
        return [capacity_per_node] * self.num_nodes
