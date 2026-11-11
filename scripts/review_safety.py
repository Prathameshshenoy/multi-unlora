"""Prepare, seal and score private safety reviews without executing model calls.

Fill semantic judgments using packet.json only; commit the aggregate seal before
scoring/unblinding. Never publish source records, packets, keys or raw judgments.
"""
import argparse
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.safety_review import (PROTOCOL, blind_safety_packet, seal_safety_review,
                                      score_safety_review, validate_safety_binding)
from evaluation.freeform_review import packet_hash


def write_new(path, value):
    with path.open('x') as stream:
        path.chmod(0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('input', type=Path)
    prepare.add_argument('--directory', type=Path, required=True)
    prepare.add_argument('--reviewer', required=True)
    for command in ('seal', 'score'):
        child = sub.add_parser(command)
        child.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((ROOT/'.runtime').resolve()):
        raise ValueError('Private safety reviews must remain under .runtime')
    if args.command == 'prepare':
        records = json.loads(args.input.read_text())['records']
        packet, key = blind_safety_packet(records, secret=secrets.token_bytes(32))
        review = {'protocol': PROTOCOL, 'packet_sha256': packet_hash(packet), 'reviewer': args.reviewer,
                  'cases': [{'case_id': case['case_id'], 'answers': {
                      slot: {'outcome': 'PENDING' if answer['raw_model'] else 'unscorable',
                             'rationale': '' if answer['raw_model'] else 'Not an eligible raw model response.'}
                      for slot, answer in case['answers'].items()}} for case in packet['cases']]}
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        for name, value in (('packet', packet), ('arm-mapping', key), ('source-records', records), ('judgments', review)):
            write_new(directory/(name+'.json'), value)
        print(json.dumps({'packet_sha256': packet_hash(packet), 'n_cases': len(packet['cases'])}))
        return
    read = lambda name: json.loads((directory/(name+'.json')).read_text())
    packet, key, review = read('packet'), read('arm-mapping'), read('judgments')
    validate_safety_binding(read('source-records'), packet, key)
    if args.command == 'seal':
        result = seal_safety_review(packet, key, review)
        write_new(directory/'seal.json', result)
    else:
        result = score_safety_review(packet, key, review, read('seal'))
        write_new(directory/'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
