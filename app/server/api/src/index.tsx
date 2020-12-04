import React from 'react';
import ReactDOM from 'react-dom';
import Hello from "./components/Hello";

console.log('Hello TypeScript!');
ReactDOM.render(<Hello content='Hello React!'/>, document.getElementById('app'));
