"""Unit + end-to-end tests for PredictorAgent and PopulationAgent."""

from __future__ import annotations

import torch

from perfsim.agents import PopulationAgent, PredictorAgent
from perfsim.core import (
    Agent,
    AgentHandle,
    BestRespondRequest,
    BestRespondResponse,
    EvalLossRequest,
    EvalLossResponse,
    GetParamsRequest,
    GetParamsResponse,
    InProcessExecutor,
    PredictRequest,
    PredictResponse,
    SetParamsRequest,
    SetParamsResponse,
    UpdateRequest,
    UpdateResponse,
)
from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import MSELoss
from perfsim.models import LinearModel
from perfsim.environments.dynamics import GaussianShiftWorld
from perfsim.environments.dynamics.strategic_linear import StrategicLinearWorld


def _make_predictor(d: int = 3, agent_id: str = "predictor") -> PredictorAgent:
    model = LinearModel(in_features=d, out_features=1, bias=False)
    learner = ERMLearner(model, MSELoss(), max_iter=50)
    return PredictorAgent(learner, MSELoss(), agent_id=agent_id)


def _make_population(
    executor: InProcessExecutor, d: int = 3, agent_id: str = "population"
) -> PopulationAgent:
    n = 100
    x0 = torch.randn(n, d)
    y = torch.zeros(n, 1)
    world = StrategicLinearWorld(x0=x0, y=y, epsilon=0.5)
    scratch = LinearModel(in_features=d, out_features=1, bias=False)
    return PopulationAgent(world, scratch, executor, agent_id=agent_id)


class TestPredictorAgentProtocol:
    def test_satisfies_agent_protocol(self) -> None:
        agent = _make_predictor()
        assert isinstance(agent, Agent)

    def test_spec_lists_five_skills(self) -> None:
        spec = _make_predictor().spec
        names = {s.name for s in spec.skills}
        assert names == {"predict", "update", "get_params", "set_params", "eval_loss"}

    def test_handle_uses_id_and_role(self) -> None:
        agent = _make_predictor(agent_id="p-1")
        h = agent.spec.handle
        assert h == AgentHandle(id="p-1", role="predictor")

    def test_model_property_reads_from_learner(self) -> None:
        # The agent's model is read from learner.model; if you mutate the
        # learner's model in place (e.g., via set_params), the agent sees it.
        m = LinearModel(in_features=3, out_features=1, bias=False)
        learner = ERMLearner(m, MSELoss())
        agent = PredictorAgent(learner, MSELoss())
        assert agent.model is m
        m.set_params(torch.tensor([7.0, 8.0, 9.0]))
        assert torch.equal(agent.model.get_params(), torch.tensor([7.0, 8.0, 9.0]))


class TestPredictorAgentSkills:
    def test_predict_matches_direct_model_call(self) -> None:
        agent = _make_predictor()
        agent.model.set_params(torch.tensor([0.5, -0.3, 0.2]))
        x = torch.randn(4, 3)
        resp = agent.predict(PredictRequest(x=x))
        with torch.no_grad():
            expected = agent.model(x)
        assert torch.equal(resp.y, expected)

    def test_get_set_params_round_trip(self) -> None:
        agent = _make_predictor()
        theta = torch.tensor([1.0, 2.0, 3.0])
        agent.set_params(SetParamsRequest(theta=theta))
        resp = agent.get_params(GetParamsRequest())
        assert torch.equal(resp.theta, theta)

    def test_update_advances_model(self) -> None:
        agent = _make_predictor()
        initial = agent.model.get_params().clone()
        data = {"x": torch.randn(32, 3), "y": torch.randn(32, 1)}
        agent.update(UpdateRequest(data=data))
        final = agent.model.get_params()
        assert not torch.equal(initial, final)

    def test_eval_loss_matches_direct_call(self) -> None:
        agent = _make_predictor()
        agent.set_params(SetParamsRequest(theta=torch.tensor([0.5, 0.5, 0.5])))
        data = {"x": torch.randn(16, 3), "y": torch.randn(16, 1)}
        resp = agent.eval_loss(EvalLossRequest(data=data))
        with torch.no_grad():
            direct = MSELoss()(agent.model, data, reduction="mean")
        assert torch.allclose(resp.loss, direct)


class TestPredictorAgentThroughExecutor:
    def test_predict_via_invoke(self) -> None:
        ex = InProcessExecutor()
        agent = _make_predictor()
        h = ex.register(agent)
        agent.model.set_params(torch.tensor([0.5, -0.3, 0.2]))
        x = torch.randn(4, 3)
        resp = ex.invoke(h, "predict", PredictRequest(x=x))
        assert isinstance(resp, PredictResponse)
        assert resp.y.shape == (4, 1)

    def test_all_skills_dispatch(self) -> None:
        ex = InProcessExecutor()
        agent = _make_predictor()
        h = ex.register(agent)
        # predict
        assert isinstance(
            ex.invoke(h, "predict", PredictRequest(x=torch.zeros(2, 3))),
            PredictResponse,
        )
        # update
        assert isinstance(
            ex.invoke(
                h, "update",
                UpdateRequest(data={"x": torch.randn(8, 3), "y": torch.randn(8, 1)}),
            ),
            UpdateResponse,
        )
        # get_params / set_params
        gp = ex.invoke(h, "get_params", GetParamsRequest())
        assert isinstance(gp, GetParamsResponse)
        sp = ex.invoke(h, "set_params", SetParamsRequest(theta=torch.zeros_like(gp.theta)))
        assert isinstance(sp, SetParamsResponse)
        # eval_loss
        el = ex.invoke(
            h, "eval_loss",
            EvalLossRequest(data={"x": torch.zeros(4, 3), "y": torch.zeros(4, 1)}),
        )
        assert isinstance(el, EvalLossResponse)


class TestPopulationAgentSkills:
    def test_best_respond_returns_world_data(self) -> None:
        ex = InProcessExecutor()
        predictor = _make_predictor()
        pop = _make_population(ex)
        ph = ex.register(predictor)
        ex.register(pop)
        predictor.model.set_params(torch.tensor([1.0, 0.0, 0.0]))
        resp = pop.best_respond(BestRespondRequest(predictor_handle=ph))
        assert isinstance(resp, BestRespondResponse)
        assert resp.x.shape == (100, 3)
        assert resp.y.shape == (100, 1)

    def test_best_respond_uses_predictor_params(self) -> None:
        # x_t = x_0 + epsilon * w; epsilon=0.5 and only the first coord is set
        # to 4.0, so the shift along that coord is 0.5 * 4.0 = 2.0. We can
        # verify this by comparing population x_t to x_0 at two predictor
        # parameter settings.
        ex = InProcessExecutor()
        predictor = _make_predictor()
        pop = _make_population(ex)
        ph = ex.register(predictor)
        ex.register(pop)

        predictor.model.set_params(torch.zeros(3))
        resp_zero = pop.best_respond(BestRespondRequest(predictor_handle=ph))

        predictor.model.set_params(torch.tensor([4.0, 0.0, 0.0]))
        resp_shift = pop.best_respond(BestRespondRequest(predictor_handle=ph))

        diff = (resp_shift.x - resp_zero.x).mean(dim=0)
        assert torch.allclose(diff, torch.tensor([2.0, 0.0, 0.0]), atol=1e-5)


class TestEndToEndRoundViaExecutor:
    def test_single_round_predict_best_respond_update(self) -> None:
        # Drive one PP round entirely through the Executor:
        #   1. predictor.predict(x_0)   -> y_hat  (smoke)
        #   2. population.best_respond  -> data conditioned on predictor's theta
        #   3. predictor.update(data)   -> mutates predictor's theta
        ex = InProcessExecutor()
        predictor = _make_predictor()
        pop = _make_population(ex)
        ph = ex.register(predictor)
        poph = ex.register(pop)

        predictor.model.set_params(torch.tensor([1.0, -0.5, 0.3]))
        theta_before = predictor.model.get_params().clone()

        pred_resp = ex.invoke(
            ph, "predict", PredictRequest(x=torch.zeros(2, 3))
        )
        assert isinstance(pred_resp, PredictResponse)

        br_resp = ex.invoke(
            poph, "best_respond", BestRespondRequest(predictor_handle=ph)
        )
        assert isinstance(br_resp, BestRespondResponse)
        data = {"x": br_resp.x, "y": br_resp.y}

        ex.invoke(ph, "update", UpdateRequest(data=data))
        theta_after = predictor.model.get_params()
        # ERM on synthetic data must move the params (theta_before is non-zero
        # and the BR data is not at its argmin).
        assert not torch.equal(theta_before, theta_after)

    def test_pp_loop_converges_on_gaussian_shift(self) -> None:
        # Same convergence test as the closed-form gating test, but routed
        # through the Executor / agent shell instead of Simulator + Learner.
        d = 3
        A = 0.5 * torch.eye(d)
        b = torch.tensor([1.0, 0.5, -0.5])
        world = GaussianShiftWorld(A=A, b=b, sigma_noise=1e-4, batch_size=512)
        world.reset(seed=0)
        theta_star = world.closed_form_fp()

        ex = InProcessExecutor()
        model = LinearModel(in_features=d, out_features=1, bias=False)
        learner = ERMLearner(model, MSELoss(), max_iter=100)
        predictor = PredictorAgent(learner, MSELoss())
        scratch = LinearModel(in_features=d, out_features=1, bias=False)
        pop = PopulationAgent(world, scratch, ex)

        ph = ex.register(predictor)
        poph = ex.register(pop)

        for _ in range(10):
            br = ex.invoke(poph, "best_respond", BestRespondRequest(predictor_handle=ph))
            ex.invoke(ph, "update", UpdateRequest(data={"x": br.x, "y": br.y}))

        gp = ex.invoke(ph, "get_params", GetParamsRequest())
        assert torch.allclose(gp.theta, theta_star, atol=2e-2)
