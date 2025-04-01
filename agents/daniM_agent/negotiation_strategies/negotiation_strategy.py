from abc import ABC, abstractmethod
from geniusweb.issuevalue.Bid import Bid
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from agents.daniM_agent.utils.opponent_model import OpponentModel
from geniusweb.progress.ProgressTime import ProgressTime

class NegotiationStrategy(ABC):
    # @abstractmethod
    # def get_best_bid(self, all_bids: list[Bid], profile: LinearAdditiveUtilitySpace, opponent_model: OpponentModel) -> Bid:
    #     """
    #     Gets best bid based on current strategy
    #     """

    @abstractmethod
    def score_bid(self, bid: Bid, agent) -> float:
        """
        Gets best bid based on current strategy
        """