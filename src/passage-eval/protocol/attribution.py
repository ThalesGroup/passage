"""Declared S2 component controls, distinct from evidence that they were run."""
from collections.abc import Mapping

VARIANTS = frozenset({'direct', 'bounded_repair', 'prediction_free', 'exact_fallback',
                      'exact_classical', 'budgeted_classical', 'oracle_diagnostic'})


def component_control(value):
    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping) or not set(value) <= {
            'variant', 'guide', 'checkpoint_id', 'machinery_id', 'primary_learned'}:
        raise ValueError('component_control requires the finite attribution schema')
    result = dict(value)
    variant, guide = result.get('variant'), result.get('guide')
    if variant not in VARIANTS or guide not in {'learned', 'prediction_free', 'none', 'oracle'}:
        raise ValueError('component_control requires a named variant and guide')
    for key in ('checkpoint_id', 'machinery_id'):
        if key in result and (not isinstance(result[key], str) or not result[key].strip()):
            raise ValueError(f'{key} must be a nonempty identity')
    primary = result.get('primary_learned', variant in {'direct', 'bounded_repair'})
    if not isinstance(primary, bool):
        raise ValueError('primary_learned must be boolean')
    if variant in {'direct', 'bounded_repair', 'exact_fallback'} and (
            guide != 'learned' or not result.get('checkpoint_id')):
        raise ValueError('learned controls require the same identified checkpoint and learned guide')
    if variant in {'bounded_repair', 'prediction_free', 'exact_fallback'} and not result.get('machinery_id'):
        raise ValueError('search controls require an identified decoder/repair machinery')
    if variant == 'prediction_free' and (guide != 'prediction_free' or result.get('checkpoint_id') or primary):
        raise ValueError('prediction_free control cannot use a learned guide or checkpoint')
    if (variant in {'exact_fallback', 'oracle_diagnostic'} or guide == 'oracle') and primary:
        raise ValueError('exact fallback and oracle diagnostics cannot be primary learned rows')
    if variant == 'oracle_diagnostic' and guide != 'oracle':
        raise ValueError('oracle_diagnostic must explicitly declare oracle assistance')
    if guide == 'oracle' and variant != 'oracle_diagnostic':
        raise ValueError('oracle assistance must be an explicitly diagnostic procedure')
    result['primary_learned'] = primary
    return result
