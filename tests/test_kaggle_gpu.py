import json
from unittest.mock import Mock

from local_app.config import prepare_data
from local_app.kaggle_gpu import KaggleGpuCheck


class FakeCLI:
    def __init__(self, tmp_path, gpu):
        self.name = 'tester'
        self.gpu = gpu
        self.calls = []
        self.tmp_path = tmp_path

    def snapshot(self):
        return {'status': 'ready', 'username': self.name}

    def identity(self):
        return self.name

    def run(self, args, **kwargs):
        self.calls.append(args)
        if args[0] == 'quota':
            return json.dumps([{'resource': 'GPU', 'used': '0.32h', 'remaining': '29.68h', 'total': '30.00h'}]), 0
        if args[:2] == ['kernels', 'status']:
            return 'x has status "KernelWorkerStatus.COMPLETE"', 0
        if args[:2] == ['kernels', 'output']:
            directory = args[args.index('-p') + 1]
            from pathlib import Path
            Path(directory, 'gpu-check.json').write_text(json.dumps({'gpu': self.gpu}))
        return '', 0


def test_quota_alone_does_not_prove_gpu_access(tmp_path):
    cli = FakeCLI(tmp_path, None)
    check = KaggleGpuCheck(prepare_data(tmp_path), cli)
    check._check('tester')
    assert check.snapshot()['status'] == 'verification_required'
    assert check.snapshot()['quota'] is None
    assert any(args[:2] == ['kernels', 'delete'] for args in cli.calls)


def test_t4_probe_confirms_gpu_and_exposes_actual_quota(tmp_path):
    cli = FakeCLI(tmp_path, 'Tesla T4')
    check = KaggleGpuCheck(prepare_data(tmp_path), cli)
    check._check('tester')
    assert check.snapshot()['status'] == 'ready'
    assert check.snapshot()['quota']['remaining'] == '29.68h'
    cli.name = 'other-user'
    assert check.snapshot()['status'] == 'unchecked'


def test_gpu_probe_classifies_verification_failure(tmp_path):
    cli = FakeCLI(tmp_path, None)
    original = cli.run
    cli.run = Mock(side_effect=lambda args, **kw: ('403 phone verification required', 1)
                   if args[:2] == ['kernels', 'push'] else original(args, **kw))
    check = KaggleGpuCheck(prepare_data(tmp_path), cli)
    check._check('tester')
    assert check.snapshot()['status'] == 'verification_required'
