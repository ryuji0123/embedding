import React, { useState } from 'react';
import './App.css';

const AppForm = () => {
  
  const dataOptions = ["pokemon", "artificial", "scurve", "swissroll"]
  const embedderOptions = ["isomap", "kernel_pca", "laplacian_eigenmaps", "locally_linear", "n_mds", "t_sne"]
  const reducerOptions = ["ica", "pca"]
  
  const [data, setData] = useState(dataOptions[0]);
  const [embedder, setEmbedder] = useState(embedderOptions[0]);
  const [reducer, setReducer] = useState(reducerOptions[0]);

  const handleSubmit = (event: any) => {
    alert(`A form was submitted: ${data}, ${embedder}, ${reducer}`);

    fetch(`http://localhost:8000/api/?data=${data}&embedder=${embedder}&reducer=${reducer}`, {
        method: 'GET'
      }).then(function(response) {
        console.log(response)
        return response.json();
      }).then(function (obj) {
        console.log(obj)
      });

    event.preventDefault();
  }

  return (
    <form onSubmit={handleSubmit}>
      <label>Data:
        <select value={data} onChange={event => setData(event.target.value)}>
          { dataOptions.map( item => <option value={item}>{item}</option>)}
        </select>
      </label>
      <br />
      <label>Embedder:
        <select value={embedder} onChange={event => setEmbedder(event.target.value)}>
          { embedderOptions.map( item => <option value={item}>{item}</option>)}
        </select>
      </label>
      <br />
      <label>Reducer: 
        <select value={reducer} onChange={event => setReducer(event.target.value)}>
          { reducerOptions.map( item => <option value={item}>{item}</option>)}
        </select>
      </label>
      <br />
      <input type="submit" value="Execute" />
    </form>
  );
}
  
export default AppForm;