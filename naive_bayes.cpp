#include <iostream>
#include <cmath>
#include "eigen3/Eigen/Dense"
#include <algorithm>
#include <vector>
#include <map>

/* Naive Bayes Classifier by Nathan Englehart, Autumn 2021 */

 int len(const Eigen::VectorXd& vector)
 {

   /* Computes the length of an input vector. */

   int sum = 0;

   for(auto v : vector)
   {
     sum += 1;
   }

   return sum;
 }

 double mean(const Eigen::VectorXd& vector)
 {

   /* Computes the mean of an input vector. */

   return vector.mean();
 }

 double standard_deviation(const Eigen::VectorXd& vector)
 {

   /* Computes the standard deviation of an input vector. */

   int size = len(vector)-1;
   double average = mean(vector);
   double standard_deviation = 0.0;

   for(auto v : vector)
   {
     standard_deviation += pow((v-average), 2);
   }

   return sqrt((double) (standard_deviation / size));
 }

 double gaussian_pdf(double x, double mean, double standard_deviation)
 {

   /* Computes the Gaussian probability distribution function for x. */

   double exponent = exp(-pow((x-mean),2) / (2 * pow(standard_deviation,2)));
   return (1 / (sqrt(2 * M_PI) * standard_deviation)) * exponent;
 }

 template <typename T> T get_eigen_index(Eigen::VectorXd vector, int index)
 {

   /* Returns the value of a vector at the given index. */

   int place = 0;
   for(auto v : vector)
   {
     if(place == index)
     {
       return v;
     }
     place++;
   }
 }

 template <typename T> T double_vector_list_lookup(std::vector<std::vector<T>> list, int first_index, int second_index)
 {

   /* Finds value located in a double vector list located at first index, second index. */

   int first_count = 0;
   for(auto v : list)
   {
     if(first_index == first_count)
     {
       int second_count = 0;
       for(auto w : v)
       {
         if(second_count == second_index)
         {
           return w;
         }
         second_count++;
       }
     }
     first_count++;
   }
 }

bool compare_classification(const Eigen::VectorXd& l, const Eigen::VectorXd& r)
{

  /* Compares the first index of one vector to the first index of another vector. */

  return l(0) < r(0);
}

Eigen::MatrixXd sorted_rows_by_classification(Eigen::MatrixXd X)
{

  /* Sorts the input matrix according to the first column entry. */

  std::vector<Eigen::VectorXd> vec;

  for(int64_t i = 0; i < X.rows(); ++i)
  {
    vec.push_back(X.row(i));
  }

  std::sort(vec.begin(), vec.end(), &compare_classification);

  for(int64_t i = 0; i < X.rows(); ++i)
  {
    X.row(i) = vec[i];
  }

  return X;
}

std::vector<int> class_indicies(Eigen::MatrixXd X, int size)
{

  /* Returns a vector containing the indicies at which the classification (first row entry) changes in a sorted matrix. */

  int idx = 0;

  std::vector<int> indicies;
  indicies.push_back(0);

  Eigen::VectorXd row = X.row(0);
  int prev_classification = get_eigen_index<int>(row,0);

  for(int i = 1; i < size; i++)
  {
    idx++;
    Eigen::VectorXd row = X.row(i);
    int classification = get_eigen_index<int>(row,0);
    if(classification != prev_classification)
    {
      indicies.push_back(idx);
    }
    prev_classification = classification;
  }

  return indicies;

}

std::vector<std::vector<double>> summarize_dataset(Eigen::MatrixXd dataset, int length)
{

  /* Calculate the mean, standard deviation, and length of each column in input dataset. */

  std::vector<std::vector<double>> summary;

  for(int i = 0; i < length; i++)
  {

    Eigen::VectorXd col = dataset.col(i);

    std::vector<double> entry;

    entry.push_back(mean(col));
    entry.push_back(standard_deviation(col));
    entry.push_back(len(col));

    summary.push_back(entry);
  }

  return summary;
}

std::vector<Eigen::MatrixXd> matricies_by_classification(Eigen::MatrixXd dataset, int size, int length)
{

  /* Sorts dataset by classification and returns vector consisting of sub-matricies by classification. */

  Eigen::MatrixXd sorted_dataset = sorted_rows_by_classification(dataset);
  std::vector<int> indicies = class_indicies(sorted_dataset,size);
  std::vector<Eigen::MatrixXd> ret;

  int indicies_array [indicies.size()];
  int idx = 0;
  for(auto v : indicies) { indicies_array[idx++] = v; }
  idx = 1;

  std::vector<Eigen::VectorXd> rows;

  for(int i = 0; i < size; i++)
  {

    if(indicies_array[idx] == i)
    {
      Eigen::MatrixXd entry(rows.size(),length);
      int j = 0;
      for(auto v : rows)
      {
        entry.row(j) = v;
        j++;
      }
      idx++;
      ret.push_back(entry);
      rows.clear();
    }

    Eigen::VectorXd row = sorted_dataset.row(i);
    rows.push_back(row);
  }
  Eigen::MatrixXd entry(rows.size(),length);
  int j = 0;
  for(auto v : rows)
  {
    entry.row(j) = v;
    j++;
  }
  ret.push_back(entry);

  return ret;
}

std::map<int, std::vector<std::vector<double>>> summarize_by_classification(Eigen::MatrixXd dataset, int size, int length)
{

  /* Returns the mean, standard deviation, and length of each column for each sub-matrix sorted by class. */

  std::vector<Eigen::MatrixXd> dict = matricies_by_classification(dataset, size, length);
  std::map<int, std::vector<std::vector<double>>> ret;

  int classification = 0;
  for( auto v : dict )
  {
    std::vector<std::vector<double>> summary = summarize_dataset(v, length);
    ret[classification] = summary;
    classification++;
  }

  return ret;
}

std::map<int, double> calculate_classification_probabilities(std::map<int, std::vector<std::vector<double>>> summaries, Eigen::VectorXd row, int size)
{

  /* Calculates the classification probabilities for a single vector using P(A|B) = P(B|A) * P(A), which is derived from Bayes Theorem, for each classification. */

  std::map<int, double> probabilities;
  std::map<int, std::vector<std::vector<double>>>::iterator it;
  int classification_value = 0;

  for(it=summaries.begin(); it != summaries.end(); ++it)
  {
    std::vector<std::vector<double>> entry = it->second;
    probabilities[classification_value] = double_vector_list_lookup<double>(entry,0,2) / size; // P(A)

    for(int i = 1; i < row.size(); i++)
    {
      double mean = double_vector_list_lookup<double>(entry,i,0);
      double standard_deviation = double_vector_list_lookup<double>(entry,i,1);
      double x = get_eigen_index<double>(row,i);
      probabilities[classification_value] *= gaussian_pdf(x,mean,standard_deviation); // P(B|A)
    }
    classification_value++;
  }

  return probabilities;
}

int predict(std::map<int, std::vector<std::vector<double>>> summaries, Eigen::VectorXd row, int size)
{

  /* Returns classification prediction. */

  std::map<int, double> probabilities = calculate_classification_probabilities(summaries, row, size);
  std::map<int, double>::iterator it;

  int best_label = -1;
  double best_probability = -1;

  for(it=probabilities.begin(); it != probabilities.end(); ++it)
  {
    if(best_label == -1 || it->second > best_probability)
    {
      best_probability = it->second;
      best_label = it->first;
    }
  }

  return best_label;
}

std::vector<int> naive_bayes_classifier(Eigen::MatrixXd validation, int validation_size, Eigen::MatrixXd training, int training_size, int length)
{

  /* Calculates the classification probabilities for each row in dataset and puts their predicted classification in a list. */

  std::map<int, std::vector<std::vector<double>>> summaries = summarize_by_classification(training, training_size, length);
  std::vector<int> predictions;

  for(int i = 0; i < validation_size; i++)
  {
    int output = predict(summaries, validation.row(i), validation_size);
    predictions.push_back(output);
  }

  return predictions;
}

int main()
{
  return 0;
}
