"""CPU-only contracts for the proposed shared-candidate diagnostic, not measurements.

The author composite B is deliberately an input: its paper-to-runtime recipe is
unresolved. This module neither fabricates B nor loads a model/starts training.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AnswerCondition:
    """Already-tokenized continuation; positions are relative to continuation.

    I supplies isolated answer IDs; P supplies the complete fixed response IDs
    and only its final-answer positions. Real-tokenizer alignment must be checked
    separately before model scoring; synthetic token IDs do not establish it.
    """
    token_ids: tuple[int, ...]
    answer_positions: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.token_ids, tuple) or not isinstance(self.answer_positions, tuple):
            raise ValueError('Use immutable tuples')
        if not self.token_ids or any(type(x) is not int or x < 0 for x in self.token_ids):
            raise ValueError('Invalid continuation token IDs')
        if not self.answer_positions or any(type(x) is not int for x in self.answer_positions):
            raise ValueError('Missing final-answer span')
        if tuple(sorted(set(self.answer_positions))) != self.answer_positions:
            raise ValueError('Answer positions must be increasing and unique')
        if self.answer_positions[0] < 0 or self.answer_positions[-1] >= len(self.token_ids):
            raise ValueError('Truncated or out-of-range final-answer span')


def score_fixed_conditions(condition, intervention_ids, score):
    """Callback returns final-answer mean logprob for a frozen model/condition.

    No generation callback exists. Image/question/model/region/source identities
    belong to the separately frozen run manifest. This only enforces text identity.
    """
    names = ('original',) + tuple(intervention_ids)
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError('Unique intervention names required')
    values = {}
    for name in names:
        value = float(score(condition, name))
        if not math.isfinite(value):
            raise ValueError('Non-finite model score')
        values[name] = value
    return values


def diagnostic_rewards(correct, base, margin_isolated, margin_prefix):
    """Six requested combinations + fixed-m=0 B control; no author_exact claim.

    B must be computed ONCE from a separately verified recipe and then frozen
    across I/P. No invalid-format penalty, routing or clipping is silently added.
    """
    if correct not in (0, 1):
        raise ValueError('C must be strict binary correctness')
    if not all(math.isfinite(x) for x in (correct, base, margin_isolated, margin_prefix)):
        raise ValueError('Non-finite reward input')
    if any(abs(x) > 1 for x in (margin_isolated, margin_prefix)):
        raise ValueError('CED margins must be in [-1,1]')
    result = {'C': float(correct), 'B': float(base), 'B_fixed_m0': .5 * base}
    for suffix, margin in [('I', margin_isolated), ('P', margin_prefix)]:
        gate = .5 * (1 + math.tanh(margin / .20))
        result['C_' + suffix] = correct * gate + .10 * margin
        result['B_' + suffix] = base * gate + .10 * margin
    return result
