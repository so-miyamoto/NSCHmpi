
CXX:=g++
CXXFLAGS:=-std=c++17 -O3 -Wall

a.out: main.o
	$(CXX) $(CXXFLAGS) $< -o $@

main.o: main.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f a.out main.o