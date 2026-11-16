import math
import pytest

torch = pytest.importorskip('torch')
from evaluation.reachability import minimum_argmax_kl


def test_two_token_bound_matches_tied_distribution():
    base = torch.tensor([0., -4.375], dtype=torch.float64)
    expected = float((.5 * (math.log(.5) - base.log_softmax(0))).sum())
    assert minimum_argmax_kl(base, 1) == pytest.approx(expected)
    assert minimum_argmax_kl(base, 0) == pytest.approx(0.)
    assert minimum_argmax_kl(base + 1000, 1) == pytest.approx(expected)


def test_three_token_solution_matches_independent_probability_grid():
    base = torch.tensor([2., 1., -1.], dtype=torch.float64)
    p = base.softmax(0)
    best = math.inf
    for target_mass in torch.linspace(1/3, 1., 401):
        a = torch.linspace(0., float(1-target_mass), 401, dtype=torch.float64)
        q = torch.stack((a, 1-target_mass-a, torch.full_like(a, target_mass)), dim=-1)
        q = q[(q >= -1e-12).all(-1) & (q[:, :2] <= target_mass+1e-12).all(-1)].clamp_min(0)
        if q.numel():
            kl = torch.where(q > 0, q * (q.clamp_min(1e-300).log()-p.log()), 0.).sum(-1)
            best = min(best, float(kl.min()))
    exact = minimum_argmax_kl(base, 2)
    assert exact <= best+1e-10
    assert best-exact < 1e-4


def test_mask_cost_is_measured_against_full_base_distribution():
    base = torch.tensor([3., 1., 0., -torch.inf])
    support = torch.tensor([False, True, True, True])
    expected = -float(base.softmax(0)[1:3].sum().log())
    assert minimum_argmax_kl(base, 1, support=support) == pytest.approx(expected)
    assert math.isinf(minimum_argmax_kl(base, 0, support=support))
    assert math.isinf(minimum_argmax_kl(base, 3))


@pytest.mark.parametrize('base,target',[(torch.tensor([]),0),(torch.tensor([float('nan')]),0),(torch.zeros(2),True),(torch.zeros(2),2)])
def test_invalid_inputs_fail(base,target):
    with pytest.raises(ValueError):
        minimum_argmax_kl(base,target)
