#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <fstream>
#include "includes/eigen3/Eigen/Dense"
#include "includes/utils.h"
#include "includes/naive_bayes.h"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */


template<typename T> T load_csv(const std::string & sys_path)
{

  /* Returns csv file input as an Eigen matrix or vector. */

  std::ifstream in;
  in.open(sys_path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(in, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      while (std::getline(lineStream, cell, ',')) {
          values.push_back(std::stod(cell));
      }
      rows = rows + 1;
  }

  return Eigen::Map<const Eigen::Matrix<typename T::Scalar, T::RowsAtCompileTime, T::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

void driver(std::string sys_path_test, std::string sys_path_train, bool verbose)
{

  /* Driver for a naive bayes classifier example. */

  Eigen::MatrixXd test = load_csv<Eigen::MatrixXd>(sys_path_test);

  if(verbose == true)
  {
      std::cout << "Test Data: " << sys_path_test << "\n";
      std::cout << test << "\n\n";
  }

  Eigen::MatrixXd train = load_csv<Eigen::MatrixXd>(sys_path_train);

  if(verbose == true)
  {
      std::cout << "Train Data: " << sys_path_train << "\n";
      std::cout << train << "\n\n";
  }

  std::vector<int> predictions = naive_bayes_classifier(test, test.rows(), train, train.rows(), train.cols(),verbose);
  //std::vector<int> predictions = naive_bayes_classifier(test, test.rows(), train, train.rows(), train.cols());

  if(verbose == true)
  {
    int count = 0;
    for(auto v : predictions)
    {
        std::cout << "Vector " << count << " Classification = " << v << "\n";
        count++;
    }
    std::cout << "\n";
  }
}

int main()
{

  /* [Iris-virginica] => 0 [Iris-setosa] => 1 [Iris-versicolor] => 2 */

  driver("./data/iristest.csv","./data/iris.csv",true);

  //driver("./data/S1test.csv","./data/S1train.csv",true);
  //driver("./data/S2test.csv","./data/S2train.csv",true);

  return 0;
}
