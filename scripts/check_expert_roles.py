"""Replay a narrow reported expert-role failure; never a general factuality judge."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from evaluation.expert_roles import assess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True)
    args=parser.parse_args()
    result=assess(json.loads(args.replay.read_text()))
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['requested_demo_behavior_passed'] else 1)


if __name__=='__main__':
    main()
