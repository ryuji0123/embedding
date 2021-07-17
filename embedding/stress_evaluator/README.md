# StressEvaluator

## Outline
- StressEvaluator
- Kruskal Stress
- Ranking Local Stress
- Ranking Global Stress
- References

## StressEvaluator
You can measure the goodness-of-fit non-metrically by using `StressEvaluator Class`:
```python
from embedding.stress_evaluator import StressEvaluator


evaluator = StressEvaluator(actual_df, n_representative_points)
```
`actual_df`: DataFrame to be attached.    
`n_representative_points`: The number of representative points which is used to measure Ranking Stresses.  
`n_neighbor`: The number of nearby points around representative points which is also used to measure Ranking Stresses.  

## Kruskal Stress
This stress is defined by Kruskal for measuring the goodness of fit of NMDS.  
You can measure this stress like this:
```python
from embedding.stress_evaluator import StressEvaluator


evaluator = StressEvaluator(actual_df, n_representative_points)
kruskal_stress = evaluator.kruskal(fitted_df)
```
`fitted_df`: Attached DataFrame.  

## Ranking Local Stress
This stress is related to the ranking of distances between one representative point and nearby points.  
The representative points are determined to be as scattered as possible according to the k-means ++ algorithm.  
This evaluates local relationships around representative points.  
You can measure Ranking Local Stress like this:
```python
from embedding.stress_evaluator import StressEvaluator


evaluator = StressEvaluator(actual_df, n_representative_points)
ranking_local_stress = evaluator.rankingLocal(fitted_df)
```

## Ranking Global Stress
This stress is related to the ranking of distances between representative points.  
This evaluates a relatively wide range of relationships.     
You can measure Ranking Global Stress like this:
```python
from embedding.stress_evaluator import StressEvaluator


evaluator = StressEvaluator(actual_df, n_representative_points)
ranking_global_stress = evaluator.rankingGlobal(fitted_df)
```

## References
- Kruskal's stress: [Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis](https://link.springer.com/article/10.1007/BF02289565)  
- Ranking Stress: [非計量多次元尺度構成法への期待と新しい視点](https://www.ism.ac.jp/editsec/toukei/pdf/49-1-133.pdf)
