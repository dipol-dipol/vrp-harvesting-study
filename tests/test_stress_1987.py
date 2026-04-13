from vrp.analysis.stress_1987 import (
    stress_calendar,
    stress_put_writer,
    STRESS_SPX_DROP,
    STRESS_VIX_JUMP,
)


def test_constants_match_spec():
    assert abs(STRESS_SPX_DROP - (-0.205)) < 1e-3
    assert STRESS_VIX_JUMP >= 20


def test_stress_calendar_short_front_hit():
    result = stress_calendar(front_0=20, second_0=22,
                              delta_front=30, delta_second=20,
                              direction="short_front")
    assert result["daily_pnl_return"] < -0.2


def test_stress_calendar_long_front_gain():
    result = stress_calendar(front_0=20, second_0=22,
                              delta_front=30, delta_second=20,
                              direction="long_front")
    assert result["daily_pnl_return"] > 0.2


def test_stress_put_writer_short_heavily_negative():
    r = stress_put_writer(S0=100, K_short=95, sigma0_pct=20,
                           S_shock=80, sigma_shock_pct=50,
                           T_remaining_days=20,
                           K_long=None, premium_collected=2.0)
    assert r["pnl_return"] < -0.05


def test_stress_put_writer_spread_less_bad():
    naked = stress_put_writer(S0=100, K_short=95, sigma0_pct=20,
                               S_shock=80, sigma_shock_pct=50,
                               T_remaining_days=20, K_long=None,
                               premium_collected=2.0)
    spread = stress_put_writer(S0=100, K_short=95, sigma0_pct=20,
                                S_shock=80, sigma_shock_pct=50,
                                T_remaining_days=20, K_long=85,
                                premium_collected=1.5)
    assert spread["pnl_return"] > naked["pnl_return"]
