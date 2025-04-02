from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from geniusweb.issuevalue.Bid import Bid


class AggressiveEarlyOffersStrategy(NegotiationStrategy):
    def __init__(self):
        pass

    def score_bid(self, bid: Bid, agent) -> float:
        agent.last_bid_sent_index += 1
        return float(agent.profile.getUtility(bid))
