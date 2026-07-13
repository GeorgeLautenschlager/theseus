from typing import List


class WorkingMemory:
    """A simple in-memory storage system for short-term recall"""
    def __init__(self):
        self.memory: List[str] = []
    
    def remember(self, item: str):
        """Add an item to the working memory."""
        self.memory.append(item)
    
    def recall(self) -> List[str]:
        """Retrieve the current contents of the working memory."""
        return self.memory
    
    def flush(self):
        """Clear the working memory."""
        self.memory.clear()