"""Prepare, seal and score blinded generated-answer reviews without model calls.

The private key maps answer slots to arms. Review packet + judgments first;
seal judgments before scoring/unblinding. No truth labels are guessed by this
script. Reviewer identity must explicitly distinguish AI from human review.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import secrets
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from evaluation.freeform_review import (blind_packet,judgment_template,packet_hash,
                                        seal_review,score_sealed_review,validate_generation_binding)


def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('input',type=Path)
    prepare.add_argument('--output',type=Path,required=True);prepare.add_argument('--reviewer',required=True)
    prepare.add_argument('--saved-role-report',action='store_true',help='Adapt previously consumed qualitative examples; never call this fresh confirmation')
    for name in ('seal','score'):
        child=sub.add_parser(name);child.add_argument('directory',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        data=json.loads(args.input.read_text())
        if args.saved_role_report:
            report=data.get('evaluation',data)
            records=[{'case_id':r['question_id'],'prompt':r['prompt'],
                'correct_references':[r['good_reference']],'incorrect_references':[r['bad_reference']],
                'outputs':{'baseline':r['complete']['baseline'],'corrected':r['complete']['unlora']}}
                for r in report['qualitative_heldout']]
        else:records=data['records']
        packet,key=blind_packet(records,secret=secrets.token_bytes(32))
        packet['input_scope']='Previously consumed qualitative cases; development/audit only.' if args.saved_role_report else data.get('scope','Scope must be established by the generating experiment protocol.')
        # Adding explicit scope changes the packet hash; bind the final bytes.
        key['packet_sha256']=packet_hash(packet)
        args.output.mkdir(parents=True,exist_ok=False)
        write_new(args.output/'packet.json',packet);write_new(args.output/'private-key.json',key)
        write_new(args.output/'source-records.json',records)
        write_new(args.output/'judgments.json',judgment_template(packet,reviewer=args.reviewer))
        print(json.dumps({'packet':str(args.output/'packet.json'),'packet_sha256':packet_hash(packet),'n_cases':len(packet['cases']),
                          'instruction':'Review packet and fill judgments without opening private-key.json. Seal before unblinding.'}))
        return
    root=args.directory
    packet=json.loads((root/'packet.json').read_text());key=json.loads((root/'private-key.json').read_text())
    review=json.loads((root/'judgments.json').read_text())
    validate_generation_binding(json.loads((root/'source-records.json').read_text()),packet,key)
    if args.command=='seal':
        seal=seal_review(packet,key,review);seal['sealed_at']=datetime.now(timezone.utc).isoformat()
        write_new(root/'seal.json',seal);print(json.dumps({'sealed':True,'packet_sha256':seal['packet_sha256'],'judgments_sha256':seal['judgments_sha256']}))
    else:
        report=score_sealed_review(packet,key,review,json.loads((root/'seal.json').read_text()))
        write_new(root/'summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
