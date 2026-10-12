"""Select one native agent_proxy leaf; leave trajectory assembly/splitting intact."""

import json
from pathlib import Path

from miles.rollout.session.v2.postprocessor_hub import default_postprocess


def longest(leaf_samples, session_metadata):
    """Keep the longest token sequence, with deterministic ties, and save every leaf."""
    chosen = max(leaf_samples, key=lambda sample: (
        len(sample.tokens), len(sample.metadata['leaf']['path_node_ids']), sample.metadata['leaf']['node_id']
    ))
    selected_ids = set(chosen.metadata['leaf']['path_node_ids'])
    nodes = session_metadata['tree']['nodes']
    excluded = [node for node in nodes if node['id'] not in selected_ids]
    stats = dict(
        policy='longest-token-path', leaves=len(leaf_samples), selected_leaf=chosen.metadata['leaf']['node_id'],
        selected_turns=len(selected_ids), total_turns=len(nodes), discarded_leaves=len(leaf_samples)-1,
        discarded_unique_turns=len(excluded),
        discarded_unique_completion_tokens=sum(n['completion_span'][1]-n['completion_span'][0] for n in excluded),
        discarded_leaf_sequence_tokens=sum(len(s.tokens) for s in leaf_samples if s is not chosen),
    )
    session_metadata['branch_selection'] = stats
    agent = session_metadata.get('agent') or {}
    if agent.get('trial_dir'):
        output = Path(agent['trial_dir']) / 'session-branches.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        # Messages are native debug traces; no HTTP headers or credentials are stored.
        output.write_text(json.dumps(dict(
            session_id=agent.get('miles_session_id'), selection=stats,
            nodes=[{k:v for k,v in n.items() if k != 'turn_args'} for n in nodes],
            leaves=[dict(leaf=s.metadata['leaf'], tokens=len(s.tokens), loss_tokens=sum(s.loss_mask),
                         messages=s.metadata.get('messages', [])) for s in leaf_samples],
        ), ensure_ascii=False, indent=2))
    return [chosen]


def postprocess(leaf_samples, session_metadata):
    """Use native reward/mask processing, then count the selected path's model calls."""
    samples = default_postprocess(leaf_samples, session_metadata)
    for sample in samples:
        path = sample.metadata['leaf']['path_node_ids']
        sample.metadata['agent_total_num_turns'] = sample.metadata.get('num_turns')
        sample.metadata['num_turns'] = len(path)
        sample.metadata['branch_selection'] = session_metadata['branch_selection']
    return samples
