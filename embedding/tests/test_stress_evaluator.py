from embedding.stress_evaluator import StressEvaluator
from embedding.data import chooseData
from embedding.embedder import chooseEmbedder


def test_kruskal_stress():
    actual_data = chooseData('scurve')
    fitted_data = chooseEmbedder('isomap', actual_data)
    fitted_data.embed(use_cache=True)

    evaluator = StressEvaluator(actual_data.df, 100)
    assert evaluator.kruskal(fitted_data.em) > 0


def test_local_ranking_stress():
    actual_data = chooseData('scurve')
    fitted_data = chooseEmbedder('isomap', actual_data)
    fitted_data.embed(use_cache=True)

    evaluator = StressEvaluator(actual_data.df, 100)
    assert evaluator.local_ranking(fitted_data.em, 15, is_z_value=False) > 0


def test_middle_ranking_stress():
    actual_data = chooseData('scurve')
    fitted_data = chooseEmbedder('isomap', actual_data)
    fitted_data.embed(use_cache=True)

    evaluator = StressEvaluator(actual_data.df, 100)
    assert evaluator.middle_ranking(fitted_data.em, 15, is_z_value=False) > 0


def test_global_ranking_stress():
    actual_data = chooseData('scurve')
    fitted_data = chooseEmbedder('isomap', actual_data)
    fitted_data.embed(use_cache=True)

    evaluator = StressEvaluator(actual_data.df, 100)
    assert evaluator.middle_ranking(fitted_data.em, 15, is_z_value=False) > 0


if __name__ == '__main__':
    test_kruskal_stress()
    test_local_ranking_stress()
    test_middle_ranking_stress()
    test_global_ranking_stress()
