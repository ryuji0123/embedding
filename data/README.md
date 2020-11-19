# dataset
Current supported datasets are:

|  File |  URL |
| ----  | ---- |
|  pokemon.csv  |  [pokemon](https://www.kaggle.com/rounakbanik/pokemon)  |
|  tmdb_5000_movies.csv  |  [tmdb_5000_movies](https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv)  |

## artificial dataset
Generated artificial dataset with any number of clusters.

```python
# Input number of dimensions, number of clusters, number of data points
all_points, cols = generate_artificial_data(n_dim=3, 
                                 n_cluster=4, 
                                 n_points=399)

fig = px.scatter_3d(
    all_points, x=0, y=1, z=2,
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.update_traces(marker=dict(size=2, line=dict(width=2, color='DarkSlateGrey')),
                   selector=dict(mode='markers')
                   )
fig.show()
```
