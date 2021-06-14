# Dataset

## Outline
- Choose Data
- Pokemon Data
- Scurve Data or Swissroll Data
- Basic Cluster Data
- Clustered Scurve Data and Clustered Swissroll Data
- Json Document Data
- Visualize
- References

## Choose Data
You can get any dataset instances when you import `chooseData` function and specify the `data_key`.  
For example, you can get `pokemon_data` like this:
```python
from embedding.data import chooseData


pokemon_data = chooseData('pokemon')
```
See `research-embedding/embeding/data/const.py` about all `data_key`.
## Pokemon Data
You can get pokemon dataset as in the example above.  
The default explanatory variables and objective variable are the Race-Value and is-legendary of each pokemon.

## Scurve Data or Swissroll Data
This is for generating s-curve or swiss-roll dataset using the scikit-learn module.  
The distribution of points on the plane of s-curve or swiss-roll is the uniform distribution.  
The coordinates of each points are non-negative values.
```python
from embedding.data import chooseData


scurve_data = chooseData('scurve')
swissroll_data = chooseData('swissroll')
```
## Basic Cluster Data
This is for generating an artificial dataset with any number of clusters following a multi-dimensional normal distribution.
```python
from embedding.data import chooseData


basic_cluster_data = chooseData('basic_cluster')
```
## Clustered Scurve Data and Clustered Swissroll Data
This is for making the basic cluster data into s-curve or swiss-roll.  
The coordinates of each points are non-negative values.
```python
from embedding.data import chooseData


clustered_scurve_data = chooseData('clustered_scurve')
clustered_swissroll_data = chooseData('clustered_swissroll')
```

## Json Document Data
This is for making the distance matrix from documents in json format.
```python
from embedding.data import chooseData


json_document_data = chooseData('json_document')
```

## Visualize
You can visualize the following data using `plotly.express` module like this:
- `scurve_data`
- `swissroll_data`
- `clustered_scurve_data`
- `clustered_swissroll_data`
- `json_document`

```python
import plotly.express as px

from embedding.data import chooseData


scurve_data = chooseData('scurve')

fig = px.scatter_3d(scurve_data.df, x='col0', y='col1', z='col2', color=scurve_data.color)
fig.update_traces(
    marker=dict(size=2, line=dict(width=2, color=scurve_data.color)),
    selector=dict(mode='markers')
    )
fig.show()
```

## References
Current supported datasets which are downloaded are:

|  File |  URL |
| ----  | ---- |
|  pokemon.csv  |  [pokemon](https://www.kaggle.com/rounakbanik/pokemon)  |
|  tmdb_5000_movies.csv  |  [tmdb_5000_movies](https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv)  |
