# AudioSearch
Python implementation of "An Industrial-Strength Audio Search Algorithm"

https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf

Dependencies:
* numpy
* scipy
* pandas
* matplotlib (only for plotting)
* librosa (only for loading audio)
* pymongo (only if MongoDB is used for the database)

A conda environment with these packages can be built automatically from **audiosearchminenv.yml**

`conda env create -f audiosearchminenv.yml`