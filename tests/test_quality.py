import pytest
from evaluation.judge import extract_score
from evaluation.quality import paired_summary, choice_prompt, fingerprint, verify_training_split
from inference.prompts import training_tokens


@pytest.mark.parametrize('text,expected', [
    ('Maybe [1] in principle.\n[0]',0), ('Reasoning [0].\n[1]',1), ('[1]',1),
    ('10',0), ('correct',0), ('[1] or [0]',0), ('[1]\nnot really',0), ('',0)])
def test_judge_only_accepts_final_unambiguous_verdict(text, expected):
    assert extract_score(text) == expected


def test_paired_uncertainty_uses_same_questions():
    rows = [{'baseline_correct':True,'unlora_correct':False},
            {'baseline_correct':False,'unlora_correct':True}]
    result = paired_summary(rows)
    assert result['paired_delta'] == 0
    assert result['paired_delta_bootstrap_95_ci'] == [-1.,1.]
    assert result == paired_summary(rows)


def test_empty_evaluation_is_not_a_quality_pass():
    with pytest.raises(ValueError):
        paired_summary([])


def test_choice_prompt_and_fingerprint_are_deterministic():
    prompt, labels = choice_prompt('2+2?', ['3','4','5','6'])
    assert labels == 'ABCD'
    assert 'B. 4' in prompt
    assert fingerprint({'a':1,'b':2}) == fingerprint({'b':2,'a':1})


class ChatTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return [1,2,3,4] if len(messages)==2 else [1,2,3,4,5,6,7]


def test_sft_masks_prompt_and_excludes_fully_truncated_responses():
    item = training_tokens(ChatTokenizer(),'prompt','response',6)
    assert item['labels'] == [-100,-100,-100,-100,5,6]
    assert training_tokens(ChatTokenizer(),'prompt','response',4) is None


@pytest.mark.parametrize('question,choices', [('', ['a', 'b']), ('Q', ['a']), ('Q', ['a', ' '])])
def test_invalid_multiple_choice_records_fail_before_inference(question, choices):
    with pytest.raises(ValueError):
        choice_prompt(question, choices)


def test_frozen_pilot_holdout_requires_matching_source_and_seed():
    import config
    train = [(f'Train {i}', 'bad response') for i in range(800)]
    heldout = [(f'Holdout {i}', 'bad response') for i in range(48)]
    manifest = {'seed': config.TRAIN_SEED, 'concepts': {
        'hallucination': {'dataset_fingerprint': 'source-v1', 'holdout_rows': 48}}}
    provenance = verify_training_split('hallucination', train, heldout, 'source-v1', manifest)
    assert provenance['verification'].startswith('reconstructed_legacy')
    with pytest.raises(ValueError, match='fingerprint changed'):
        verify_training_split('hallucination', train, heldout, 'source-v2', manifest)
    manifest['seed'] += 1
    with pytest.raises(ValueError, match='seed differs'):
        verify_training_split('hallucination', train, heldout, 'source-v1', manifest)


def test_recorded_split_hashes_reject_changed_holdout_and_overlap():
    import config
    from training.pilot import split_provenance
    train = [('Question 1', 'bad response')]
    heldout = [('Question 2', 'another bad response')]
    recorded = split_provenance(train, heldout, 'source-v1')
    manifest = {'seed': config.TRAIN_SEED, 'concepts': {'hallucination': recorded}}
    assert verify_training_split('hallucination', train, heldout, 'source-v1', manifest)['verification'] == 'recorded_pair_hashes_match'
    with pytest.raises(ValueError, match='provenance mismatch'):
        verify_training_split('hallucination', train, [('Question 2', 'changed response')], 'source-v1', manifest)
    with pytest.raises(ValueError, match='disjoint'):
        split_provenance(train, [(' QUESTION   1 ', 'bad response')], 'source-v1')


def test_oversize_choice_prompt_is_rejected_before_model_forward(monkeypatch):
    torch = pytest.importorskip('torch')
    import config
    from evaluation.quality import PromptTooLong, score_choices
    from types import SimpleNamespace

    class Tokenizer:
        def encode(self, label, **kwargs):
            return [ord(label)]

        def apply_chat_template(self, messages, **kwargs):
            return 'rendered prompt'

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.ones((1, 20), dtype=torch.long))

    monkeypatch.setattr(config, 'MAX_SEQ_LENGTH', 20)
    with pytest.raises(PromptTooLong):
        score_choices(None, Tokenizer(), 'Q', ['one', 'two'], 0, None, None)


def test_bootstrap_rejects_unusable_draw_count_and_duplicate_questions():
    row = {'baseline_correct': True, 'unlora_correct': False, 'question_id': 'same'}
    with pytest.raises(ValueError, match='100'):
        paired_summary([row], draws=0)
    with pytest.raises(ValueError, match='duplicate'):
        paired_summary([row, row])


def test_refusal_proxy_does_not_count_quoted_text_or_as_an_example():
    pytest.importorskip('pandas')
    from evaluation.analysis import check_refusal
    assert check_refusal('I cannot assist with that request.')
    assert not check_refusal('As an example, secure your email with MFA.')
    assert not check_refusal('Here is an answer including the quoted phrase "I cannot help".')
