"""Frozen fresh question IDs, declared before selective-subtraction generation."""
from training.role_corpus import source_records,prepare_corpus,question_id,SOURCE_SHA256
from training.role_allocation import ALLOCATION
from evaluation.freeform_trial import allocate
from evaluation.freeform_review import packet_hash

CONFIRMATION_IDS=('037a497cbf5bb4d527814c5844fab73e8a73aaf9a7e98c0f78c0dcd7d6a888af', 'e250d160613806dd27370064ecb3a52c8f6debe3818ad0c1083eb609ac28defe', '9f46d09892a3301372ecca6adbbe4b8fed9006a8c77334a94258666b94170964', '768ed6d530c9862837e7bbda5903747e462e1e749c840a2200b75687c1ae9a55', 'e9c46a25a40b731723b05866994232ca4d67f4458b5ce6f985f1cb078ddfb954', 'ba448e16d108c3ecc8263ef634d23ab3d1af97bea80254c636167eaba3d3b45a', 'bc40953c138f884114ab826a2860f2bf64e1fa9491bc19175fab153ff3c56af0', '5d3666406aa0f615d4fd500787dcceebd5b6715f8b70686ccf6ca52c938adae4', '8779925889e928385c388bafb85c56544c8aa16db69b277e3c4fcf67d252dce7', '27cb3ec2f54514fbd48abdc9f4d4c3a5832bb3cd46c36b9dfc71b93508f300a2', '5da3b4b6c9fe585a6cde2f29ebaf22303d1311e7656c6d7632fe06069dd38e5a', 'b5b7af02d6b9321a6e36a195aa3e965fd52672e9ddf3e9047c63203731e20957', '4db05913f3ab61e40c02db38f6fbca4ecce18dc0cf177bf06a79eb1c7ba5c29d', 'fe69ffa1ae33aca0398147ee9981de5ab0366c5e84deebb04a91b5985931ab83', '99a064d7d688b6506a3fcfc0a4f3ea436163d1bee0c82b84fafb66f2924890bd', 'd2a4aec11f31a309055c476ea7ded9986b6f0b2c20b92775a26456995fd9a6e7', '4ae130411bdf6c3da506ecdfdfba348737254e07263c4506d68079fadebec451', 'a2b71b71656b4eb0536882f388a8b5b93d586b86c722b20139ceed3e3736ffe7', 'b8f14559898275e4bda9e70e4d71e097c9b710f5d339bb4513bcea6fc8975c18', '7c1b02dc16ae5080cacb2bb81378bef860a74468e3fe7149ea876b296eb8d892')
ALLOCATION_SHA256="a7b911e99897ccc6256ccdbf3c58becfd6a275fd6b5e4a2d4278e200ab19148c"


def validate_fresh_ids(selected,forbidden):
    if len(selected)!=20 or len(set(selected))!=20 or set(selected)&set(forbidden):
        raise ValueError('Selective confirmation must have 20 unique previously unused questions')


def frozen_selective_allocation(data=None):
    raw=source_records(data)
    corpus=prepare_corpus(raw,blocked_question_ids=ALLOCATION['blocked_question_ids'],test_n=203)
    old=allocate(corpus,raw)
    forbidden={r['question_id'] for r in corpus['train']+corpus['development']+corpus['test'][:128]}
    forbidden.update(r['case_id'] for r in old['confirmation'])
    forbidden.update(ALLOCATION['blocked_question_ids'])
    validate_fresh_ids(CONFIRMATION_IDS,forbidden)
    available={r['question_id'] for r in corpus['test'][128:]}
    if not set(CONFIRMATION_IDS)<=available:raise ValueError('Fresh selection left its original held-out partition')
    byid={question_id(r['Question']):r for r in raw};cases=[]
    for key in CONFIRMATION_IDS:
        row=byid[key]
        refs=lambda field,best:list(dict.fromkeys([row[best].strip()]+[v.strip() for v in row[field].split(';') if v.strip()]))
        cases.append({'case_id':key,'prompt':row['Question'].strip(),'category':row['Category'],
            'correct_references':refs('Correct Answers','Best Answer'),
            'incorrect_references':refs('Incorrect Answers','Best Incorrect Answer')})
    result={'protocol':'selective-subtraction-fresh-allocation-v3','source_sha256':SOURCE_SHA256,'confirmation':cases,
        'scope':'20 prospectively curated, previously unused adapter-evaluation questions. Includes additional health, society and paranormal categories; not directly comparable to the previous restricted confirmation or official TruthfulQA accuracy.'}
    if packet_hash(result)!=ALLOCATION_SHA256:raise ValueError('Frozen selective allocation changed')
    return result
