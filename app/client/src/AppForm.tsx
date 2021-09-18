import React, { useState } from 'react';
import script from './python/script.py';
import './App.css';

declare global {
  interface Window {
      pyodide: any;
      languagePluginLoader: any;
  }
}

const AppForm = () => {
  const [output, setOutput] = useState("loading...");
  const runPythonScript = (code: string) => {
    window.pyodide.loadPackage(['numpy', 'pandas', 'scikit-learn']).then(() => {
      const output = window.pyodide.runPython(code);
      setOutput(output);
    })
  }
  
  const dataOptions = ["pokemon", "artificial", "scurve", "swissroll"]
  const embedderOptions = ["isomap", "kernel_pca", "laplacian_eigenmaps", "locally_linear", "n_mds", "t_sne"]
  const reducerOptions = ["pca", "mds"]
  
  const [data, setData] = useState(dataOptions[0]);
  const [embedder, setEmbedder] = useState(embedderOptions[0]);
  const [reducer, setReducer] = useState(reducerOptions[0]);

  const handleSubmit = (event: any) => {
    fetch(`http://localhost:8000/api/?data=${data}&embedder=${embedder}`, {
        method: 'GET'
      }).then(function(response) {
        return response.json();
      }).then(function (obj) {
        console.log(obj)
        
        console.log("loaded")
        window.languagePluginLoader.then(() => {
          fetch(script)
            .then(src => src.text())
            .then(str => str.replace("EMBEDDING", JSON.stringify(obj)))
            .then(str => str.replace("REDUCER", reducer))
            .then(runPythonScript)
        })
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
      <h2>Python Output</h2>
      { output }
    </form>
  );
}
  
export default AppForm;