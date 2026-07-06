from abc import ABC, abstractmethod

class AIService (ABC):

    @abstractmethod
    def curriculum_analyze (self, curriculum: str) -> str:
        pass
    
    @abstractmethod
    def curriculum_generate(self, batch_size: int = 10) -> str:
        pass