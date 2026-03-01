"""
Preprocessing script for Experiments 2 and 3.

Experiment 2: Full dialogue history as input, turn-level belief state as target (NLU).
Experiment 3: Full dialogue history as input, cumulative belief state as target (DST).

Usage:
    # Experiment 2 (history input, turn-level target):
    python prepare_dst_data.py \
        --input_file $BDIR/data_preparation/data/multiwoz21/processed/train/version_1/data.json \
        --output_dir data_preparation/data/multiwoz21/processed/train/exp2 \
        --mode exp2

    # Experiment 3 (history input, cumulative target):
    python prepare_dst_data.py \
        --input_file $BDIR/data_preparation/data/multiwoz21/processed/train/version_1/data.json \
        --refs_file $BDIR/data_preparation/data/multiwoz21/refs/dev/dev_v2.1.json \
        --output_dir data_preparation/data/multiwoz21/processed/train/exp3 \
        --mode exp3

Run for train, dev, and test splits separately.
"""

import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def build_dialogue_history(examples):
    """
    Given the flat list of examples, reconstruct the full dialogue history
    for each turn by grouping by dialogue ID and accumulating turns in order.

    Returns a dict: example_id -> full history string up to and including that turn.
    """
    # Group examples by dialogue, preserving order
    dialogues = defaultdict(list)
    for ex in examples:
        dial_id = ex['example_id'].split('-')[0]
        turn_id = int(ex['example_id'].split('-')[1])
        dialogues[dial_id].append((turn_id, ex))

    # Sort each dialogue's turns by turn number
    for dial_id in dialogues:
        dialogues[dial_id].sort(key=lambda x: x[0])

    history_map = {}  # example_id -> full history string

    for dial_id, turns in dialogues.items():
        accumulated_turns = []
        for turn_id, ex in turns:
            example_id = ex['example_id']
            dst_input = ex['dst_input']

            # The current turn's dst_input already contains "<SYS> ... <USR> ..."
            # We want to prepend all previous turns to it
            if accumulated_turns:
                full_history = ' '.join(accumulated_turns) + ' ' + dst_input
            else:
                full_history = dst_input

            history_map[example_id] = full_history
            accumulated_turns.append(dst_input)

    return history_map


def build_cumulative_belief_states(refs_file):
    """
    Load the reference annotations file and extract the cumulative
    (dialogue-level) belief state for each turn.

    Returns a dict: example_id -> cumulative belief state string.
    """
    refs = load_json(refs_file)
    cumulative_map = {}

    for dial_id, turns in refs.items():
        for turn_key, turn_data in turns.items():
            # turn_key is like "turn-0", "turn-1", etc.
            turn_num = int(turn_key.split('-')[1])
            example_id = f"{dial_id}-{turn_num}"
            cumulative_map[example_id] = turn_data['dst_belief_state']

    return cumulative_map


def build_cumulative_belief_states_from_training(examples):
    """
    Build cumulative belief states directly from the training JSON,
    for use when no refs file is available (e.g. for the train split).

    At each turn we:
      1. Parse the turn-level belief state into a set of slot-value pairs
      2. Remove any pairs listed in nlu_correction (user corrections)
      3. Add the remaining pairs to the running cumulative state
      4. Sort alphabetically (to match the refs file format)

    Returns a dict: example_id -> cumulative belief state string.
    """
    SEP = ' <SEP> '

    def parse_belief_state(bs_str):
        """Parse 'domain slot value <SEP> ...' into a dict of
        'domain slot' -> 'value' pairs. Returns empty dict for empty string."""
        pairs = {}
        if not bs_str or not bs_str.strip():
            return pairs
        for item in bs_str.split('<SEP>'):
            item = item.strip()
            if not item:
                continue
            parts = item.split()
            if len(parts) >= 3:
                # domain slot value (value may be multiple words e.g. "guest house")
                key = f"{parts[0]} {parts[1]}"
                value = ' '.join(parts[2:])
                pairs[key] = value
            else:
                print(f"Warning: could not parse belief state item: '{item}'")
        return pairs

    def serialize_belief_state(pairs):
        """Serialize dict back to sorted 'domain slot value <SEP> ...' string."""
        items = sorted(f"{key} {value}" for key, value in pairs.items())
        return SEP.join(items)

    # Group and sort by dialogue, same as build_dialogue_history
    dialogues = defaultdict(list)
    for ex in examples:
        dial_id = ex['example_id'].split('-')[0]
        turn_id = int(ex['example_id'].split('-')[1])
        dialogues[dial_id].append((turn_id, ex))
    for dial_id in dialogues:
        dialogues[dial_id].sort(key=lambda x: x[0])

    cumulative_map = {}

    for dial_id, turns in dialogues.items():
        cumulative_pairs = {}  # running belief state as a dict

        for turn_id, ex in turns:
            example_id = ex['example_id']

            # Step 1: parse this turn's new slot-value pairs
            new_pairs = parse_belief_state(ex['belief_state'])

            # Step 2: apply corrections - remove any slot-value pair the user
            # corrected. e.g. nlu_correction = "hotel stay 5" means remove
            # "hotel stay" from the cumulative state entirely before adding new.
            correction = ex.get('nlu_correction', '')
            if correction and correction.strip():
                correction_pairs = parse_belief_state(correction)
                for key in correction_pairs:
                    cumulative_pairs.pop(key, None)

            # Step 3: add new pairs, overwriting any existing value for same slot
            cumulative_pairs.update(new_pairs)

            # Step 4: serialize sorted
            cumulative_map[example_id] = serialize_belief_state(cumulative_pairs)

    return cumulative_map


def prepare_exp2(examples, history_map):
    """
    Experiment 2: Replace dst_input with full dialogue history.
    Keep belief_state as turn-level (unchanged).
    """
    output = []
    skipped = 0
    for ex in examples:
        example_id = ex['example_id']
        if example_id not in history_map:
            skipped += 1
            continue
        output.append({
            'example_id': example_id,
            'dst_input': history_map[example_id],
            'nlu_correction': ex.get('nlu_correction', ''),
            'belief_state': ex['belief_state'],  # turn-level, unchanged
        })
    if skipped:
        print(f"Warning: skipped {skipped} examples not found in history map")
    return output


def prepare_exp3(examples, history_map, cumulative_map):
    """
    Experiment 3: Replace dst_input with full dialogue history.
    Replace belief_state with cumulative (dialogue-level) belief state.
    """
    output = []
    skipped = 0
    for ex in examples:
        example_id = ex['example_id']
        if example_id not in history_map:
            skipped += 1
            continue
        if example_id not in cumulative_map:
            # Some examples in train may not be in refs - skip them
            skipped += 1
            continue
        output.append({
            'example_id': example_id,
            'dst_input': history_map[example_id],
            'nlu_correction': ex.get('nlu_correction', ''),
            'belief_state': cumulative_map[example_id],  # cumulative
        })
    if skipped:
        print(f"Warning: skipped {skipped} examples (not in history or cumulative map)")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True,
                        help='Path to input data.json (train, dev, or test)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory (data.json will be written here)')
    parser.add_argument('--mode', required=True, choices=['exp2', 'exp3'],
                        help='exp2: history input + turn-level target; '
                             'exp3: history input + cumulative target')
    parser.add_argument('--refs_file', default=None,
                        help='Path to refs JSON file (required for exp3)')
    args = parser.parse_args()

    print(f"Loading input data from {args.input_file}")
    examples = load_json(args.input_file)
    print(f"Loaded {len(examples)} examples")

    print("Building dialogue history map...")
    history_map = build_dialogue_history(examples)

    if args.mode == 'exp2':
        print("Preparing Experiment 2 data (history input, turn-level targets)...")
        output = prepare_exp2(examples, history_map)

    elif args.mode == 'exp3':
        if args.refs_file:
            # Dev/test: use the refs file (ground truth cumulative annotations)
            print(f"Loading cumulative belief states from {args.refs_file}")
            cumulative_map = build_cumulative_belief_states(args.refs_file)
            print(f"Loaded cumulative belief states for {len(cumulative_map)} turns")
        else:
            # Train: no refs file available, so accumulate from turn-level data
            print("No refs file provided - building cumulative belief states from training data...")
            cumulative_map = build_cumulative_belief_states_from_training(examples)
            print(f"Built cumulative belief states for {len(cumulative_map)} turns")
        print("Preparing Experiment 3 data (history input, cumulative targets)...")
        output = prepare_exp3(examples, history_map, cumulative_map)

    output_path = Path(args.output_dir) / 'data.json'
    print(f"Writing {len(output)} examples to {output_path}")
    save_json(output, output_path)

    # Copy the preprocessing_config.yaml from the original directory
    # so the training script can find it (it looks for it next to data.json)
    src_config = Path(args.input_file).parent / 'preprocessing_config.yaml'
    dst_config = Path(args.output_dir) / 'preprocessing_config.yaml'
    if src_config.exists():
        shutil.copy(src_config, dst_config)
        print(f"Copied preprocessing_config.yaml to {args.output_dir}")
    else:
        print(f"Warning: could not find preprocessing_config.yaml at {src_config}")

    # Print a few examples so you can sanity check the output
    print("\n--- Sample output examples ---")
    for ex in output[:2]:
        print(json.dumps(ex, indent=2))
        print()


if __name__ == '__main__':
    main()