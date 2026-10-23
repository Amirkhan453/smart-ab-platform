from ab_platform.engine.power import PowerAndSampleSizeEngine


def test_evan_miller_exact_match():
    """
    Проверяем точное совпадение с формулой Эвана Миллера (evanmiller.org/ab-testing/sample-size.html):
    1) Baseline = 20% (0.20), Relative MDE = 5% (delta = 0.01, p2 = 0.21), alpha = 0.05, power = 0.80:
       Точное значение = 25,255.085 -> ceil = 25,256 (round = 25,255).
    2) Дефолт сайта при открытии: Baseline = 20% (0.20), Absolute MDE = 5 п.п. (delta = 0.05, p2 = 0.25):
       Точное значение = 1,030.22 -> ceil = 1,031 (round = 1,030).
    3) Baseline = 3% (0.03), Relative MDE = 20% (delta = 0.006, p2 = 0.036):
       Точное значение = 13,050.17 -> ceil = 13,051 (round = 13,050).
    """
    res_rel_5pct = PowerAndSampleSizeEngine.calculate_sample_size(
        baseline_mean=0.20,
        mde_relative=0.05,
        is_binary=True,
        binary_method="evan_miller",
        alpha=0.05,
        power=0.80,
    )
    assert res_rel_5pct.n_per_variant == 25256

    res_abs_5pct = PowerAndSampleSizeEngine.calculate_sample_size(
        baseline_mean=0.20,
        mde_absolute=0.05,
        is_binary=True,
        binary_method="evan_miller",
        alpha=0.05,
        power=0.80,
    )
    assert res_abs_5pct.n_per_variant == 1031

    res_3pct = PowerAndSampleSizeEngine.calculate_sample_size(
        baseline_mean=0.03,
        mde_relative=0.20,
        is_binary=True,
        binary_method="evan_miller",
        alpha=0.05,
        power=0.80,
    )
    assert res_3pct.n_per_variant == 13051


def test_calculator_net_url_exact_match():
    """
    Проверяем совпадение с переданной ссылкой Calculator.net:
    type=2 & cl2=95 & ss2=20000 & pc2=3 & ps2= (пусто)
    На сайте Calculator.net ответ: Margin of Error = ±0.24% (±0.002364).
    """
    audit = PowerAndSampleSizeEngine.calculate_survey_margin_of_error(
        sample_size=20000,
        proportion=0.03,
        confidence_level=0.95,
        finite_population_n=None,
    )
    assert audit.margin_of_error_pct_points == 0.24
    assert abs(audit.margin_of_error_abs - 0.002364) < 1e-5

    audit_fpc = PowerAndSampleSizeEngine.calculate_survey_margin_of_error(
        sample_size=20000,
        proportion=0.03,
        confidence_level=0.95,
        finite_population_n=25000,
    )
    assert audit_fpc.fpc_factor_applied < 0.45
    assert audit_fpc.margin_of_error_abs < audit.margin_of_error_abs
