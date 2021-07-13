from embedding.stress_evaluator import StressEvaluator
from embedding.data import chooseData
from embedding.embedder import chooseEmbedder


def test_kruskal_stress():
    actual_data = chooseData('scurve')
    fitted_data = chooseEmbedder('isomap', actual_data)
    fitted_data.embed(use_cache=True)

    evaluator = StressEvaluator(actual_data.df, 100)
    assert evaluator.kruskal(fitted_data.em) > 0


if __name__ == '__main__':
    test_kruskal_stress()
