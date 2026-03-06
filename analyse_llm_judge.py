"""
Analyse LLM-as-a-Judge results to find cases where exact match fails
but the LLM judge considers the predictions semantically equivalent.

Usage:
    python analyse_llm_judge.py \
        --results_file hyps/test_128/gpt-4o-mini-llmj.json \
        --n 10
"""

import json
import argparse
from collections import Counter


def load_json(path):
    with open(path) as f:
        return json.load(f)


def analyse(results_file, n_examples):
    data = load_json(results_file)

    # First entry is the summary statistics
    summary = data[0]
    errors = data[1:]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Average Joint Accuracy:  {summary['average_joint_accuracy']}%")
    print(f"  Total turns:             {summary['total_turns']}")
    print(f"  Exact match failures:    {summary['exact_match_errors']}")
    print(f"  LLM/exact disagreements: {summary['llm_disagree_count']}")
    print()

    # Split into categories
    # consistent=false AND llm_match=true means:
    #   exact match said wrong, LLM said correct
    rescued = [e for e in errors if not e['consistent'] and e['llm_match']]

    # consistent=false AND llm_match=false means:
    #   exact match said wrong, LLM also said wrong (consistent=false here
    #   would only occur if exact_match was somehow true but llm said false,
    #   which shouldn't happen given the code logic - but we handle it anyway)
    llm_also_wrong = [e for e in errors if not e['consistent'] and not e['llm_match']]

    # consistent=true AND llm_match=false means:
    #   exact match said wrong, LLM also said wrong (both agree it's wrong)
    both_wrong = [e for e in errors if e['consistent'] and not e['llm_match']]

    print("=" * 60)
    print("BREAKDOWN OF EXACT MATCH FAILURES")
    print("=" * 60)
    print(f"  Rescued by LLM judge (exact=No, LLM=Yes): {len(rescued)}")
    print(f"  Both agree wrong     (exact=No, LLM=No):  {len(both_wrong)}")
    print(f"  Inconsistent other:                        {len(llm_also_wrong)}")
    print()

    # --- Rescued examples ---
    print("=" * 60)
    print(f"RESCUED BY LLM JUDGE — exact match wrong, LLM says correct")
    print(f"(first {n_examples} of {len(rescued)})")
    print("=" * 60)
    for ex in rescued[:n_examples]:
        ref = ex['reference_belief_state']
        hyp = ex['predicted_belief_state']
        print(f"  Example ID: {ex['example_id']}")
        print(f"  Reference:  {ref}")
        print(f"  Predicted:  {hyp}")

        # Show what specifically differs
        ref_set = set(ref)
        hyp_set = set(hyp)
        in_ref_not_hyp = ref_set - hyp_set
        in_hyp_not_ref = hyp_set - ref_set
        if in_ref_not_hyp:
            print(f"  In ref not hyp: {in_ref_not_hyp}")
        if in_hyp_not_ref:
            print(f"  In hyp not ref: {in_hyp_not_ref}")
        print()

    # --- Pattern analysis on rescued examples ---
    print("=" * 60)
    print("PATTERN ANALYSIS: Why did exact match fail on rescued examples?")
    print("=" * 60)

    # Collect all differing pairs to find common patterns
    diff_pairs = []
    for ex in rescued:
        ref_set = set(ex['reference_belief_state'])
        hyp_set = set(ex['predicted_belief_state'])
        in_ref = ref_set - hyp_set
        in_hyp = hyp_set - ref_set
        # Pair up differing items by slot name
        ref_by_slot = {}
        hyp_by_slot = {}
        for item in in_ref:
            parts = item.split('=', 1)
            if len(parts) == 2:
                ref_by_slot[parts[0]] = parts[1]
        for item in in_hyp:
            parts = item.split('=', 1)
            if len(parts) == 2:
                hyp_by_slot[parts[0]] = parts[1]
        for slot in ref_by_slot:
            if slot in hyp_by_slot:
                diff_pairs.append((slot, ref_by_slot[slot], hyp_by_slot[slot]))

    # Categorise differences
    case_diffs = []
    apostrophe_diffs = []
    spelling_diffs = []
    article_diffs = []
    other_diffs = []

    for slot, ref_val, hyp_val in diff_pairs:
        if ref_val.lower() == hyp_val.lower():
            case_diffs.append((slot, ref_val, hyp_val))
        elif ref_val.replace("'", "") == hyp_val.replace("'", "") or \
             hyp_val.replace("'", "") == ref_val.replace("'", ""):
            apostrophe_diffs.append((slot, ref_val, hyp_val))
        elif ref_val.replace('centre', 'center') == hyp_val or \
             hyp_val.replace('centre', 'center') == ref_val:
            spelling_diffs.append((slot, ref_val, hyp_val))
        elif ref_val.lstrip('the ') == hyp_val.lstrip('the ') or \
             hyp_val.lstrip('the ') == ref_val.lstrip('the '):
            article_diffs.append((slot, ref_val, hyp_val))
        else:
            other_diffs.append((slot, ref_val, hyp_val))

    print(f"  Capitalisation differences: {len(case_diffs)}")
    for slot, r, h in case_diffs[:3]:
        print(f"    [{slot}] ref='{r}' hyp='{h}'")
    print()
    print(f"  Apostrophe differences: {len(apostrophe_diffs)}")
    for slot, r, h in apostrophe_diffs[:3]:
        print(f"    [{slot}] ref='{r}' hyp='{h}'")
    print()
    print(f"  Spelling differences (centre/center etc): {len(spelling_diffs)}")
    for slot, r, h in spelling_diffs[:3]:
        print(f"    [{slot}] ref='{r}' hyp='{h}'")
    print()
    print(f"  Article differences (the/no-the): {len(article_diffs)}")
    for slot, r, h in article_diffs[:3]:
        print(f"    [{slot}] ref='{r}' hyp='{h}'")
    print()
    print(f"  Other differences: {len(other_diffs)}")
    for slot, r, h in other_diffs[:5]:
        print(f"    [{slot}] ref='{r}' hyp='{h}'")
    print()

    # --- Both wrong examples ---
    print("=" * 60)
    print(f"BOTH EXACT MATCH AND LLM AGREE: WRONG")
    print(f"(first {n_examples} of {len(both_wrong)})")
    print("=" * 60)
    for ex in both_wrong[:n_examples]:
        ref = ex['reference_belief_state']
        hyp = ex['predicted_belief_state']
        print(f"  Example ID: {ex['example_id']}")
        print(f"  Reference:  {ref}")
        print(f"  Predicted:  {hyp}")
        ref_set = set(ref)
        hyp_set = set(hyp)
        print(f"  Missing:    {ref_set - hyp_set}")
        print(f"  Extra:      {hyp_set - ref_set}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', required=True,
                        help='Path to LLM judge results JSON file')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of examples to print per category')
    args = parser.parse_args()
    analyse(args.results_file, args.n)


if __name__ == '__main__':
    main()