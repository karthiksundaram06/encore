import torch

from encore.rl.reward import WorldEnsemble, diversity_bonus


def test_pessimistic_combine():
    U = torch.tensor([[[1.0, 0.0]], [[1.0, 2.0]], [[1.0, -2.0]]])
    single = WorldEnsemble.combine(U, "single", 1.0)
    pess = WorldEnsemble.combine(U, "ensemble", 1.0)
    assert torch.allclose(single, torch.tensor([[1.0, 0.0]]))
    assert pess[0, 0] == 1.0
    assert pess[0, 1] < 0


def test_diversity_bonus():
    emb = torch.eye(4)
    same = diversity_bonus(emb, torch.tensor([[0, 0, 0]]))
    diff = diversity_bonus(emb, torch.tensor([[0, 1, 2]]))
    assert torch.allclose(same, torch.zeros(1, 3))
    assert torch.allclose(diff, torch.ones(1, 3))


def test_grpo_advantage_zero_when_rewards_equal():
    r = torch.ones(2, 4)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_world_models_share_content_and_skip_it_in_checkpoints():
    import numpy as np

    from encore.world.model import ListenerModel

    content = torch.randn(20, 8)
    cold = torch.zeros(20, dtype=torch.bool)
    a = ListenerModel(20, content, cold, 16, 1, 2, 10, prior_scale=0.5)
    b = ListenerModel(20, content, cold, 16, 1, 2, 10, prior_scale=0.5)
    assert a.content.data_ptr() == b.content.data_ptr() == content.data_ptr()
    sd = a.state_dict()
    assert "content" not in sd and "is_cold" not in sd and "item_prior" in sd
    old = dict(sd, content=content, is_cold=cold)
    b.load_state_dict({k: v for k, v in old.items() if k not in ("content", "is_cold")})
    assert np.allclose(a.item_prior.numpy(), b.item_prior.numpy())
