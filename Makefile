TARGETS=naive-bayes-cli
CXX=g++ -std=c++11
INC=-I./includes

all: $(TARGETS)

naive-bayes-cli: utils.o naive_bayes.o main.o
	$(CXX) $(INC) utils.o naive_bayes.o main.o -o naive-bayes-cli

naive_bayes.o: utils.o includes/naive_bayes.h naive_bayes.cpp
	$(CXX) $(INC) -c naive_bayes.cpp

utils.o: includes/utils.h utils.cpp
	$(CXX) $(INC) -c utils.cpp

clean:
	rm -rf $(TARGETS) *.o
