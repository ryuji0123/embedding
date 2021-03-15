# Research Embedding
## Outline
  - Installation
  - Build Docker Image & Run Docker Container
  - Exec Docker Container
  - Before Commit
  - Jupyterlab
  - Links

## Installation
 ```
 $ git clone --recursive git@github.com:murata-lab/research-embedding.git
 $ git submodule update --init
 ```
 
 If you want to develop or use app, you can see [README](https://github.com/murata-lab/research-embedding/tree/master/app) in app.
 
## Build Docker Image & Run Docker Container
 ```
 $ cd ~/research-embedding/
 $ ./build_docker_image.sh
 $ ./run_docker_container.sh
 ```

## Exec Docker Container
In another session, you can exec docker container like this:

```
$ cd ~/research-embedding/docker
$ ./exec_docker_container.sh
```

## Tests
```
$ ./python_build.sh
```

## Jupyterlab
 You can use .ipynb files with jupyterlab. If you want to use, create notebooks directory in docker container like this:
 ```
 $ mkdir /workspace/notebooks
 ```

 Since notebooks directory is in .gitignore, you do not warry about it when you add files and commit.
 Then, you can run jupyterlab in docker container with this command:
 ```
 $ ./jupyter_run.sh
 ```

## Links
[App](https://github.com/murata-lab/research-embedding/tree/master/app)

[Data](https://github.com/murata-lab/research-embedding/tree/master/embedding/data)
