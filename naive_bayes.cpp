#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <fstream>
#include "includes/eigen3/Eigen/Dense"
#include "includes/utils.h"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */

int verbose_vector_count = 0;

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

std::vector<int> class_indicies(Eigen::MatrixXd X, int size)
{

  /* Returns a vector containing the indicies at which the classification (first row entry) changes in a sorted matrix. */

  int idx = 0;

  std::vector<int> indicies;
  indicies.push_back(0);

  Eigen::VectorXd row = X.row(0);
  int prev_classification = (int) get_eigen_index(row,0);

  for(int i = 1; i < size; i++)
  {
    idx++;
    Eigen::VectorXd row = X.row(i);
    int classification = (int) get_eigen_index(row,0);
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

int num_classifications = 0;

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

std::map<int, double> calculate_classification_probabilities(std::map<int, std::vector<std::vector<double>>> summaries, Eigen::VectorXd row, int size, bool verbose)
{

  /* Calculates the classification probabilities for a single vector using P(A|B) = P(B|A) * P(A), which is derived from Bayes Theorem, for each classification. */

  std::map<int, double> probabilities;
  std::map<int, std::vector<std::vector<double>>>::iterator it;
  int classification_value = 0;

  if(verbose == true)
  {
    std::cout << "Row " << verbose_vector_count++ << ": [ ";
    for(auto v : row)
    {
      std::cout << v << " ";
    }
    std::cout << "]\n";

  }

  for(it=summaries.begin(); it != summaries.end(); ++it)
  {

    std::vector<std::vector<double>> entry = it->second;
    probabilities[classification_value] = double_vector_list_lookup(entry,0,2) / (size); // P(A)

    for(int i = 1; i < row.size(); i++)
    {
      double mean = double_vector_list_lookup(entry,i,0);
      double standard_deviation = double_vector_list_lookup(entry,i,1);
      double x = get_eigen_index(row,i);
      probabilities[classification_value] *= gaussian_pdf(x,mean,standard_deviation); // P(B|A)
    }

    if(verbose == true)
    {
      std::cout << "Class: " << classification_value << " Probability: " << probabilities[classification_value] << "\n";
    }

    classification_value++;
  }

  if(verbose == true)
  {
    std::cout << "\n";
  }

  return probabilities;
}

int predict(std::map<int, std::vector<std::vector<double>>> summaries, Eigen::VectorXd row, int size, bool verbose)
{

  /* Returns classification prediction. */

  std::map<int, double> probabilities = calculate_classification_probabilities(summaries, row, size, verbose);
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

std::vector<int> gaussian_naive_bayes_classifier(Eigen::MatrixXd validation, int validation_size, Eigen::MatrixXd training, int training_size, int length, bool verbose)
{

  /* Calculates the classification probabilities for each row in dataset and puts their predicted classification in a list. */

  std::map<int, std::vector<std::vector<double>>> summaries = summarize_by_classification(training, training_size, length);
  std::vector<int> predictions;

  for(int i = 0; i < validation_size; i++)
  {
    int output = predict(summaries, validation.row(i), training_size, verbose);
    predictions.push_back(output);
  }

  return predictions;
}

std::vector<int> mle_naive_bayes_classifier(Eigen::MatrixXd validation, int validation_size, Eigen::MatrixXd training, int training_size, int length, bool verbose)
{

  /* Calculates the classification probabilities for each row in dataset and puts their predicted classification in a list. */


  std::vector<int> predictions;

  // calculate class frequencies, then
  // calculate overall classification probabilities i.e. compute q_j(y)

  std::map<int, std::vector<std::vector<double>>> summaries = summarize_by_classification(training, training_size, length);

  std::vector<int> unique_classifications;
  std::vector<int> unique_classifications_count; 
  std::vector<float> unique_classifications_probabilities;
  
  int c = 0;
  
  for(auto v : summaries)
  {
  	unique_classifications.push_back(c++);
  }
  
  c--;

  int class_count = 0;
  int total_count = 0;

  for(int i = 0; i < c; i++)
  {
  	for(auto v : summaries[i])
	{
		class_count++;
	}

	unique_classifications_count.push_back(class_count);
	total_count += class_count;
	class_count = 0;
  }		

  for(auto v : unique_classifications_count)
  {
  	unique_classifications_probabilities.push_back(v / total_count);
  }

  // record the features of each vector corresponding to classification

  std::vector<Eigen::MatrixXd> list = matricies_by_classification(Eigen::MatrixXd training, int training_size, int length)

  std::map<int,std::vector<std::map<int,int>>> dict;

  int current_class = 0;

  for(auto v : list)
  {
  	std::vector<std::map<int,int>> entry; 
	
	// get all occurences of a category (assuming category is from 1-10 and skipping those that are not in this range)

	for(int i = 0; i < v.cols()-1; i++)
	{
		

		Eigen::VectorXd col = train.col(i);
		
		// sort col

		// 
	}

	


  	map[current_class++] = entry;
  }

  // calculate classification probabilities for each feature e.g. compute p_j(x|y)

  return predictions;
}
