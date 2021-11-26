TARGETS=naive_bayes
CXX=g++ -std=c++11

all: $(TARGETS)

naive_bayes: naive_bayes.cpp
	$(CXX) -o naive_bayes naive_bayes.cpp

clean:
	rm -rf $(TARGETS) *.o
