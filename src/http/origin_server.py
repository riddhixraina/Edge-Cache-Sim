"""
Mock origin server for HTTP simulation.
"""

from flask import Flask, send_file
import os
import random
import time


class OriginServer:
    """Mock origin server for realistic HTTP simulation."""
    
    def __init__(self, port: int = 8080):
        self.app = Flask(__name__)
        self.port = port
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/objects/<object_id>')
        def serve_object(object_id):
            """Simulate origin server with realistic latency."""
            # Simulate network latency (50-200ms)
            time.sleep(random.uniform(0.05, 0.2))
            
            # Generate or retrieve actual file
            filepath = f"data/objects/{object_id}.bin"
            if not os.path.exists(filepath):
                # Generate synthetic object
                size = random.randint(1024, 10*1024*1024)  # 1KB-10MB
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'wb') as f:
                    f.write(os.urandom(size))
            
            return send_file(filepath)
    
    def run(self):
        """Start the origin server."""
        self.app.run(port=self.port, debug=False)
