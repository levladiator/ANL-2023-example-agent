from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from geniusweb.issuevalue.Bid import Bid


class AggressiveEarlyOffersStrategy(NegotiationStrategy):
    """
    Aggressive early offers strategy for negotiation.
    Focuses on making aggressive offers early in the negotiation process.
    """
    def __init__(self):
        pass

    def score_bid(self, bid: Bid, agent) -> float:
        """
        Scores the bid according to our own utility.
        Ensures the same bid is not sent twice in a row by incrementing the last_bid_sent_index.
        """
        agent.last_bid_sent_index += 1
        return float(agent.profile.getUtility(bid))
