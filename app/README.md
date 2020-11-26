# Django Project for Demonstration

## Outline
  - Installation
  - Compile TypeScript
  - Run Server
  - Tests

## Installation
```
$ cd ~/research-embedding/app/
$ npm install --save-dev
```

## Compile TypeScript
Before compiling the TypeScript files, you need to update the webpack.config.json. Add your files in entry block.

```
$ cd ~/research-embedding/app/
$ npm run build
```


## Run Server
After running research_embedding container, you can conduct the runserver script like this:
```
$ cd ~/research-embedding/app/
$ ./runserver.sh
```
Then, you can access the website at `localhost:8000` using your browser.

## Tests
```
$ cd ~/research-embedding/app/
$ ./tests.sh
```
