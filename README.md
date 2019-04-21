# AudioSearch
Python implementation of "[An Industrial-Strength Audio Search Algorithm](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)"

Created for my [term paper](https://github.com/lukemcraig/AudioSearch/blob/master/craiglm_TermPaper_CS5100_Fall_2018.pdf) in CS 5110 - Design and Analysis of Algorithms, Fall 2018.
>  One of the things I look for most in a term paper is its simplicity in explaining complex ideas. This year, the Best Term Paper Award goes to Luke Craig for his paper titled "Robust Audio Fingerprinting Using Combinatorial Hashing and Temporal Offsets". Many congratulations to him.
> 
> -- Dr. Raghuveer Mohan

Some neat figures in my paper:
 
![figure 7-2](https://github.com/lukemcraig/AudioSearch/blob/master/figures/max%20filter.png)

![figure 6-2](https://github.com/lukemcraig/AudioSearch/blob/master/figures/target%zone.png)

![figure 8-3](https://github.com/lukemcraig/AudioSearch/blob/master/figures/pub%noise.png)

------
## Usage:

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

