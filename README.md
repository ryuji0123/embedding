# embedding
Outline
  - installation
  - setup

# installation
 ```
 git clone git@github.com:ryuji0123/embedding.git
 ```
 
 # setup
 ```
 cd ~/embedding/docker
 docker build -t embedding .
 docker run -it --rm -v ~/embedding:/workspace embedding bash
 ```
