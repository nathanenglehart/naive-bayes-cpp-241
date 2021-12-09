# Naive Bayesian Classifier Algorithm
## Author
Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021)

## Usage
To clone and run this classifier so that it can be run on a dataset, please run the following. 

```
git clone git@github.com:nathanenglehart/naive-bayes-cpp-241.git
cd naive-bayes-cpp-241
make
```

The program is meant to be run as the below, where train and test are the paths to the train and test csv files.

```
./naive-bayes-cli [train] [test] [options...]
```

For a help menu, please run:

```
./naive-bayes-cli -h
```

To run this program in verbose mode, please run:

```
./naive-bayes-cli [train] [test] -v 
```

## Notes
This is currently a work in progress. We are currently including the eigen3 linear algebra library folder within this program.

## References
David Barber, [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/171216.pdf), 2016. (pp. 237-245)<br>

Max Kuhn and Kjell Johnson, Applied Predictive Modeling, 2013. (pp. 353)<br>

Data Mining: Practical Machine Learning Tools and Techniques by Ian Whitten, 2016. (pp. 88)
