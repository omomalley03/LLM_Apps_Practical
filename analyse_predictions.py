"""
Analysis script to show examples of correct predictions, errors, and bad format cases.

Usage:
    # NLU (turn-level) analysis
    python analyse_predictions.py \
        --refs $BDIR/data_preparation/data/multiwoz21/refs/test/test_v2.1.json \
        --hyps hyps/test/gpt2_exp2/model.200000/belief_states.json \
        --field nlu_belief_state \
        --n 5

    # CC-DST (dialogue-level) analysis
    python analyse_predictions.py \
        --refs $BDIR/data_preparation/data/multiwoz21/refs/test/test_v2.1.json \
        --hyps hyps/test/gpt2_exp2_cc_dst/model.200000/belief_states.json \
        --field dst_belief_state \
        --n 5
"""

import json
import argparse
import re
from collections import defaultdict


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_belief_state(bs_str):
    """Parse 'domain slot value <SEP> ...' into a set of 'domain slot value' strings."""
    if not bs_str or not bs_str.strip():
        return set()
    items = set()
    for item in bs_str.split('<SEP>'):
        item = item.strip()
        if item:
            items.add(item)
    return items


def extract_predicted_bs(predicted_str):
    """Extract the belief state from between <bos> and <eos> tokens."""
    match = re.search(r'<bos>(.*?)<eos>', predicted_str)
    if match:
        return match.group(1).strip()
    # fallback: try to find anything after <bos>
    match = re.search(r'<bos>(.*)', predicted_str)
    if match:
        return match.group(1).strip()
    return predicted_str.strip()


def load_refs(refs_file, field):
    """Load reference belief states into a flat dict: example_id -> belief state string."""
    refs = load_json(refs_file)
    ref_map = {}
    for dial_id, turns in refs.items():
        for turn_key, turn_data in turns.items():
            turn_num = int(turn_key.split('-')[1])
            example_id = f"{dial_id}-{turn_num}"
            ref_map[example_id] = turn_data[field]
    return ref_map


def load_hyps(hyps_file):
    """Load hypotheses into a flat dict: example_id -> predicted belief state string."""
    hyps = load_json(hyps_file)
    hyp_map = {}
    for item in hyps:
        example_id = item['example_id']
        predicted = extract_predicted_bs(item['predicted_belief_state'])
        hyp_map[example_id] = predicted
    return hyp_map


def classify(ref, hyp):
    """Classify a prediction as correct, missing slots, extra slots, or wrong values."""
    ref_set = parse_belief_state(ref)
    hyp_set = parse_belief_state(hyp)

    if ref_set == hyp_set:
        return 'correct', set(), set()

    missing = ref_set - hyp_set   # in ref but not in hyp
    extra = hyp_set - ref_set     # in hyp but not in ref
    return 'error', missing, extra


def analyse(refs_file, hyps_file, field, n_examples):
    print(f"References: {refs_file} (field: {field})")
    print(f"Hypotheses: {hyps_file}")
    print()

    ref_map = load_refs(refs_file, field)
    hyp_map = load_hyps(hyps_file)

    correct = []
    errors = []
    bad_format = []
    missing_slots_examples = []   # ref has slots, hyp predicted nothing
    extra_slots_examples = []     # hyp hallucinated slots not in ref

    for example_id, ref in ref_map.items():
        if example_id not in hyp_map:
            continue
        hyp = hyp_map[example_id]

        # Check for bad format (items with fewer than 3 parts)
        is_bad = False
        for item in hyp.split('<SEP>'):
            item = item.strip()
            if item and len(item.split()) < 3:
                bad_format.append({
                    'example_id': example_id,
                    'ref': ref,
                    'hyp': hyp,
                    'bad_item': item
                })
                is_bad = True
                break

        result, missing, extra = classify(ref, hyp)

        if result == 'correct':
            correct.append({'example_id': example_id, 'ref': ref, 'hyp': hyp})
        else:
            errors.append({
                'example_id': example_id,
                'ref': ref,
                'hyp': hyp,
                'missing': missing,
                'extra': extra
            })
            if ref and not hyp:
                missing_slots_examples.append(example_id)
            if hyp and not ref:
                extra_slots_examples.append(example_id)

    total = len(correct) + len(errors)
    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total turns:      {total}")
    print(f"Correct:          {len(correct)} ({100*len(correct)/total:.1f}%)")
    print(f"Errors:           {len(errors)} ({100*len(errors)/total:.1f}%)")
    print(f"Bad format:       {len(bad_format)}")
    print(f"Missed all slots: {len(missing_slots_examples)}")
    print(f"Hallucinated:     {len(extra_slots_examples)}")
    print()

    # --- Correct examples ---
    print(f"{'='*60}")
    print(f"CORRECT PREDICTIONS (first {n_examples})")
    print(f"{'='*60}")
    for ex in correct[:n_examples]:
        print(f"  ID:  {ex['example_id']}")
        print(f"  REF: {ex['ref']}")
        print(f"  HYP: {ex['hyp']}")
        print()

    # --- Error examples ---
    print(f"{'='*60}")
    print(f"ERRORS (first {n_examples})")
    print(f"{'='*60}")
    for ex in errors[:n_examples]:
        print(f"  ID:      {ex['example_id']}")
        print(f"  REF:     {ex['ref']}")
        print(f"  HYP:     {ex['hyp']}")
        if ex['missing']:
            print(f"  MISSING: {ex['missing']}")
        if ex['extra']:
            print(f"  EXTRA:   {ex['extra']}")
        print()

    # --- Error breakdown by type ---
    print(f"{'='*60}")
    print(f"ERROR BREAKDOWN BY SLOT TYPE")
    print(f"{'='*60}")
    missing_counts = defaultdict(int)
    extra_counts = defaultdict(int)
    for ex in errors:
        for item in ex['missing']:
            parts = item.split()
            if len(parts) >= 2:
                slot_key = f"{parts[0]} {parts[1]}"  # domain slot
                missing_counts[slot_key] += 1
        for item in ex['extra']:
            parts = item.split()
            if len(parts) >= 2:
                slot_key = f"{parts[0]} {parts[1]}"
                extra_counts[slot_key] += 1

    print(f"  Most commonly MISSED slots (in ref, not in hyp):")
    for slot, count in sorted(missing_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {slot}: {count}")
    print()
    print(f"  Most commonly HALLUCINATED slots (in hyp, not in ref):")
    for slot, count in sorted(extra_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {slot}: {count}")
    print()

    # --- Bad format examples ---
    if bad_format:
        print(f"{'='*60}")
        print(f"BAD FORMAT EXAMPLES (first {n_examples})")
        print(f"{'='*60}")
        for ex in bad_format[:n_examples]:
            print(f"  ID:       {ex['example_id']}")
            print(f"  BAD ITEM: '{ex['bad_item']}'")
            print(f"  FULL HYP: {ex['hyp']}")
            print(f"  REF:      {ex['ref']}")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs', required=True, help='Path to refs JSON file')
    parser.add_argument('--hyps', required=True, help='Path to hypothesis JSON file')
    parser.add_argument('--field', required=True,
                        choices=['nlu_belief_state', 'dst_belief_state'],
                        help='Which reference field to score against')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of examples to print per category')
    args = parser.parse_args()

    analyse(args.refs, args.hyps, args.field, args.n)


if __name__ == '__main__':
    main()