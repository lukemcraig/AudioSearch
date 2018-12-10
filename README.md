# AudioSearch
Python implementation of "An Industrial-Strength Audio Search Algorithm"

https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf

Usage:

positional arguments:
* d             the root directory of the library of mp3s to insert or test

optional arguments:
*  --insert      to insert into the database instead of testing
*  --plot        whether to plot the algorithm
* -processes p  the number of processes to use during insertion
*  -noise n      noise type (White or Pub)


Dependencies:
* numpy
* scipy
* pandas
* matplotlib (only for plotting)
* librosa (only for loading audio)
* mutagen (for parsing mp3 metadata)
* pymongo (only if MongoDB is used for the database)

A conda environment with these packages can be built automatically from **audiosearchminenv.yml**

`conda env create -f audiosearchminenv.yml`

