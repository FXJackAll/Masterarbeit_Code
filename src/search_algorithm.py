from abc import ABC, abstractmethod


# TODO find better name for "Search_Pattern"
class Search_Algorithm(ABC):

    @abstractmethod
    def construct_solution(self, search_parameters: dict) -> dict:
        """
        Args:
            search_parameters:

        Returns: solution
        """
        pass
