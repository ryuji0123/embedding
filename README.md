# embedding
Outline
  - installation
  - setup
  - jupyterlab

# installation
 ```
 git clone git@github.com:ryuji0123/embedding.git
 ```
 
# setup
 ```
 cd ~/embedding/docker
 docker build -t embedding .
 docker run -it --rm -p 18888:18888 -v ~/embedding:/workspace embedding bash
 ```
# jupyterlab
 You can use .ipynb files with jupyterlab. If you want to use, create notebooks direvtory like this:
 ```
 $ mkdir ~/embedding/notebooks
 ```

 Since notebooks directory is in .gitignore, you do not warry about it when you add files and commit.
 Then, you can run jupyterlab with this command:
 ```
 $ jupyter lab --port 18888 --ip=0.0.0.0 --allow-root
 ```
