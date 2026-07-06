from abc import ABC, abstractmethod

class AIService (ABC):

    @abstractmethod
    def curriculum_analyze (self, curriculum: str) -> str:
        pass
    
    @abstractmethod
    def curriculum_generate(self) -> str:
        pass