# Naive Bayesian Classifier Algorithm
## Author
Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021)

## Usage
To clone and run this classifier so that it can be run on a dataset, please run the following. 

```bash
git clone https://github.com/nathanenglehart/naive-bayes-cpp-241
cd naive-bayes-cpp-241
make
```

The program is meant to be run as the below, where train and test are the paths to the train and test csv files.

```bash
./naive-bayes-cli [train] [test] [options...]
```

For a help menu, please run:

```bash
./naive-bayes-cli -h
```

To run this program in verbose mode, please run:

```bash
./naive-bayes-cli [train] [test] -v 
```

## Install
To install this program to your posix standard system, please run the following.

```bash
git clone https://github.com/nathanenglehart/naive-bayes-cpp-241
cd naive-bayes-cpp-241
make
sudo cp naive-bayes-cli /usr/local/bin/naive-bayes-cli
sudo chmod 0755 /usr/local/bin/naive-bayes-cli
```

The program can then be run from any location on your system, as in the below.

```bash
naive-bayes-cli [train] [test] [options...]
```

## Uninstall
To uninstall this program from your system, run the following.

```bash
sudo rm /usr/local/bin/naive-bayes-cli
```

## Documentation

Detailed documentation can be found at [https://nathanenglehart.github.io/naive-bayes](https://nathanenglehart.github.io/naive-bayes). 

## Notes

This is currently a work in progress. We are currently including the eigen3 linear algebra library folder within this program for a simpler installation process.

## References

Barber, David. (2016). Bayesian Reasoning and Machine Learning. Cambridge University Press.

Fisher, R.A. (1988). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris). Irvine, CA: University of California, School of Information and Computer Science.
 
Kuhn, Max and Johnson, Kjell. (2013). Applied Predictive Modeling, 2013. Springer.
