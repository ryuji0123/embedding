# Django and React Project for Demonstration

## Outline
  - Installation
  - Run Server
  - Run Client
  - Tests

## Installation
```
$ cd ~/research-embedding/app/
$ npm install --save-dev
```

## Run Server (Django)
After running research_embedding container, you can conduct the runserver script like this:
```
$ cd ~/research-embedding/app/server/
$ sh ./runserver.sh
```
Then, the server will run at `localhost:8000`

## Run Client (React)
After starting the server, execute the following commands in the container to launch the client.
```
$ cd /workspace/app/client/
$ npm start
```
Then, you can access the web app at `localhost:5000` using your browser.

## Tests
```
$ cd ~/research-embedding/app/server/
$ sh ./tests.sh
```
