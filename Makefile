all: main

main: main.o NeuralNetwork.o Layer.o Neuron.o
	c++ main.o NeuralNetwork.o Layer.o Neuron.o -o main

main.o: main.cpp
	c++ -c main.cpp

NeuralNetwork.o: NeuralNetwork.cpp NeuralNetwork.hpp Layer.hpp
	c++ -c NeuralNetwork.cpp

Layer.o: Layer.cpp Layer.hpp Neuron.hpp
	c++ -c Layer.cpp

Neuron.o: Neuron.cpp Neuron.hpp
	c++ -c Neuron.cpp

clean:
	rm -f *.o main

run: main
	./main