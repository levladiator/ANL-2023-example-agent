from collections import OrderedDict

from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from geniusweb.issuevalue.Bid import Bid


class LogrollingStrategy(NegotiationStrategy):
    def __init__(self):
        self.delta = 0.15

    def score_bid(self, bid: Bid, agent) -> float:
        if agent.opponent_model is None:
            return float(agent.profile.getUtility(bid))

        search_objective = self.get_search_objective(agent)

        for issue, value in search_objective.items():
            if bid.getValue(issue) != value:
                return 0

        return self.score_bid_for_social_welfare(bid, agent)

    def get_search_objective(self, agent) -> dict[str, str]:
        own_weights: dict[str, float] = {key: float(value) for key, value in agent.profile.getWeights().items()}
        own_weights = OrderedDict(
            sorted(own_weights.items(), key=lambda item: item[1], reverse=True)
        )

        opponent_weights: dict[str, float] = dict()
        for issue, issue_estimator in agent.opponent_model.issue_estimators.items():
            opponent_weights[issue] = issue_estimator.weight
        opponent_weights = OrderedDict(sorted(opponent_weights.items(), key=lambda item: item[1]))

        own_target_issue: str = next(iter(own_weights.keys()))
        opponent_target_issue: str = ""
        for issue in opponent_weights.keys():
            if issue != own_target_issue:
                opponent_target_issue = issue
                break

        own_target_value: str = ""
        own_values_and_utilities = agent.profile.getUtilities()[own_target_issue]
        max_own_utility = -1
        for value in agent.domain.getValues(own_target_issue):
            if own_values_and_utilities.getUtility(value) > max_own_utility:
                own_target_value = value
                max_own_utility = own_values_and_utilities.getUtility(value)

        opponent_target_value_tracker = agent.opponent_model.issue_estimators[opponent_target_issue].value_trackers
        opponent_target_value: str = ""
        max_target_value_count = -1
        for value, value_estimator in opponent_target_value_tracker.items():
            if value_estimator.count > max_target_value_count:
                max_target_value_count = value_estimator.count
                opponent_target_value = value

        search_objective = {own_target_issue: own_target_value, opponent_target_issue: opponent_target_value}
        return search_objective
