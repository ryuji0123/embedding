# embedding
Outline
  - installation
  - build docker image & run docker container
  - exec docker container
  - before commit
  - jupyterlab

# installation
 ```
 $ git clone git@github.com:murata-lab/research-embedding.git
 ```
 
# build docker image & run docker container
 ```
 $ ./build_docker_image.sh
 $ cd ~/research-embedding/docker
 $ ./run_docker_container.sh
 ```

# exec docker container
In another session, you can exec docker container like this:

```
$ cd ~/research-embedding/docker
$ ./exec_docker_container.sh
```

# before commit
```
$ ./python_build.sh
```

# jupyterlab
 You can use .ipynb files with jupyterlab. If you want to use, create notebooks directory in docker container like this:
 ```
 $ mkdir /workspace/notebooks
 ```

 Since notebooks directory is in .gitignore, you do not warry about it when you add files and commit.
 Then, you can run jupyterlab in docker container with this command:
 ```
 $ ./jupyter_run.sh
 ```
