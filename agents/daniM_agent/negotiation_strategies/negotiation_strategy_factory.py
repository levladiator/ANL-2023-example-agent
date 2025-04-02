from agents.daniM_agent.enums import Fairness, Stance, NegotiationType
from agents.daniM_agent.negotiation_strategies.negotiation_strategy import NegotiationStrategy
from agents.daniM_agent.negotiation_strategies.continued_concessions_strategy import ContinuedConcessionsStrategy
from agents.daniM_agent.negotiation_strategies.logrolling_strategy import LogrollingStrategy


class NegotiationStrategyFactory:
    @staticmethod
    def select_strategy(opponent_negotiation_model: NegotiationType, opponent_stance: Stance,
                        opponent_fairness: Fairness) -> NegotiationStrategy:
        """
        Selects a negotiation strategy based on the opponent's negotiation model, stance, and fairness.
        """
        if opponent_negotiation_model == NegotiationType.CONCEDER:
            if opponent_stance == Stance.GREEDY or opponent_fairness == Fairness.UNFAIR:
                return LogrollingStrategy()
            else:  # opponent_stance == Stance.NEUTRAL or opponent_stance == Stance.GENEROUS:
                return ContinuedConcessionsStrategy()

        elif opponent_negotiation_model == NegotiationType.HARDLINER:
            if opponent_stance == Stance.GREEDY or opponent_fairness == Fairness.UNFAIR:
                return LogrollingStrategy()
            elif opponent_stance == Stance.NEUTRAL:
                return LogrollingStrategy()
            else:  # opponent_stance == Stance.GENEROUS:
                return ContinuedConcessionsStrategy()

        else:  # opponent_negotiation_model == NegotiationType.RANDOM
            return LogrollingStrategy()
