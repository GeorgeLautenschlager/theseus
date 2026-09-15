from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from theseus.memory_experiment_eval import run_evaluation
from theseus.model_providers.lm_studio_provider import LmStudioProvider


def test_offline_evaluation_preserves_failures_and_protects_existing_trial(tmp_path):
    report = run_evaluation(tmp_path)
    assert report['mode'] == 'reference-extraction-offline'
    assert report['costs']['answer_chat_calls'] == 0
    assert report['costs']['embedding_calls_at_consolidation'] == 0
    deadline = report['results'][0]['systems']
    assert deadline['layered']['top1']['expected_text_present']
    assert not deadline['layered']['top1']['forbidden_text_present']
    assert deadline['raw_lexical']['top1']['forbidden_text_present']
    unknown = report['results'][-1]['systems']['layered']
    assert not unknown['top1']['expected_text_present']
    assert json.loads((tmp_path / 'report.json').read_text())['queries'] == 12
    with pytest.raises(ValueError, match='empty workdir'):
        run_evaluation(tmp_path)


def test_answer_evaluation_records_usage_and_rejects_unknown_citations(tmp_path):
    class Answerer:
        last_chat_usage = {'total_tokens': 10}

        def chat(self, *args, **kwargs):
            return '{"answer":"unknown","evidence_ids":["invented"]}'

    report = run_evaluation(tmp_path, answerer=Answerer())
    assert report['costs']['answer_chat_calls'] == 24
    assert len(report['costs']['reported_answer_usage']) == 24
    for row in report['results']:
        for system in row['systems'].values():
            assert not system['answer']['valid_citations']


@pytest.mark.parametrize('operation', ['chat', 'embed'])
def test_provider_usage_does_not_survive_missing_metadata_or_failure(operation):
    provider = LmStudioProvider()
    provider._client = MagicMock()
    if operation == 'chat':
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))])
        call = provider._client.chat.completions.create
        attr = 'last_chat_usage'
        expected = {'total_tokens': 9}
    else:
        response = SimpleNamespace(data=[SimpleNamespace(embedding=[1., 2.])])
        call = provider._client.embeddings.create
        attr = 'last_embedding_usage'
        expected = 9
    response.usage = SimpleNamespace(total_tokens=9)
    call.return_value = response
    getattr(provider, operation)('test')
    assert getattr(provider, attr) == expected
    del response.usage
    getattr(provider, operation)('test')
    assert getattr(provider, attr) is None
    setattr(provider, attr, expected)
    call.side_effect = RuntimeError('unavailable')
    with pytest.raises(RuntimeError):
        getattr(provider, operation)('test')
    assert getattr(provider, attr) is None
