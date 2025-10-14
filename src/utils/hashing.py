"""
Consistent hashing implementation for request routing.

This module provides consistent hashing functionality for distributing
requests across multiple edge nodes with minimal redistribution on node changes.
"""

from typing import List, Dict, Any, Tuple
import hashlib
import bisect


class ConsistentHasher:
    """
    Consistent hashing implementation for CDN request routing.
    
    Provides stable request-to-node mapping with minimal redistribution
    when nodes are added or removed from the system.
    """
    
    def __init__(self, nodes: List[Any], virtual_nodes_per_node: int = 150) -> None:
        """
        Initialize consistent hasher.
        
        Args:
            nodes: List of nodes to distribute requests across
            virtual_nodes_per_node: Number of virtual nodes per physical node
        """
        self.nodes = nodes
        self.virtual_nodes_per_node = virtual_nodes_per_node
        self.ring: List[Tuple[int, Any]] = []  # (hash_value, node)
        
        self._build_ring()
    
    def _build_ring(self) -> None:
        """Build the consistent hash ring."""
        self.ring.clear()
        
        for node in self.nodes:
            for i in range(self.virtual_nodes_per_node):
                # Create virtual node identifier
                virtual_node_id = f"{node.node_id}_{i}"
                
                # Hash the virtual node
                hash_value = self._hash(virtual_node_id)
                
                # Add to ring
                self.ring.append((hash_value, node))
        
        # Sort ring by hash value
        self.ring.sort(key=lambda x: x[0])
    
    def _hash(self, key: str) -> int:
        """
        Hash a key to a 32-bit integer.
        
        Args:
            key: String to hash
            
        Returns:
            32-bit hash value
        """
        return int(hashlib.md5(key.encode()).hexdigest(), 16) & 0xFFFFFFFF
    
    def get_node(self, key: str) -> Any:
        """
        Get the node responsible for a given key.
        
        Args:
            key: The key to route
            
        Returns:
            Node responsible for this key
        """
        if not self.ring:
            raise RuntimeError("Hash ring is empty")
        
        # Hash the key
        key_hash = self._hash(key)
        
        # Find the first node with hash >= key_hash
        # This uses binary search for O(log n) performance
        index = bisect.bisect_left(self.ring, (key_hash, None))
        
        # Wrap around if we're at the end
        if index >= len(self.ring):
            index = 0
        
        return self.ring[index][1]
    
    def add_node(self, node: Any) -> None:
        """
        Add a new node to the hash ring.
        
        Args:
            node: Node to add
        """
        if node not in self.nodes:
            self.nodes.append(node)
            self._build_ring()
    
    def remove_node(self, node: Any) -> None:
        """
        Remove a node from the hash ring.
        
        Args:
            node: Node to remove
        """
        if node in self.nodes:
            self.nodes.remove(node)
            self._build_ring()
    
    def get_distribution(self, num_samples: int = 10000) -> Dict[int, int]:
        """
        Analyze the distribution of requests across nodes.
        
        Args:
            num_samples: Number of sample keys to test
            
        Returns:
            Dictionary mapping node_id to request count
        """
        distribution = {node.node_id: 0 for node in self.nodes}
        
        for i in range(num_samples):
            key = f"sample_key_{i}"
            node = self.get_node(key)
            distribution[node.node_id] += 1
        
        return distribution
    
    def get_load_balance_ratio(self, num_samples: int = 10000) -> float:
        """
        Calculate load balance ratio (min_load / max_load).
        
        Args:
            num_samples: Number of sample keys to test
            
        Returns:
            Load balance ratio (1.0 = perfectly balanced)
        """
        distribution = self.get_distribution(num_samples)
        
        if not distribution:
            return 1.0
        
        loads = list(distribution.values())
        min_load = min(loads)
        max_load = max(loads)
        
        return min_load / max_load if max_load > 0 else 1.0
    
    def get_ring_info(self) -> Dict[str, Any]:
        """
        Get information about the hash ring.
        
        Returns:
            Dictionary containing ring statistics
        """
        return {
            "num_nodes": len(self.nodes),
            "virtual_nodes_per_node": self.virtual_nodes_per_node,
            "total_virtual_nodes": len(self.ring),
            "ring_size": len(self.ring),
            "load_balance_ratio": self.get_load_balance_ratio(),
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"ConsistentHasher(nodes={len(self.nodes)}, ring_size={len(self.ring)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
