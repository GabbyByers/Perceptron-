#include <iostream>
#include <vector>

#include "SFML/Graphics.hpp"
#define _CRTDBG_MAP_ALLOC

using namespace std;

class neuron
{
public:
	int index = 0;
	char type = '\0'; // 'i' = input, 'h' = hidden, 'o' = output
	
	vector<float> weights;
	float activation = 0.0f;
	float bias = 0.0f;

	vector<float> gradient_weights;
	float gradient_activation = 0.0f;
	float gradient_bias = 0.0f;

	neuron() {}

	~neuron() {}

	float& operator[](int index) // directly indexing the neuron will return a reference to the weight at the provided index
	{
		return weights[index];
	}
};

class layer
{
public:
	int index = 0;
	int num_neurons = 0;
	vector<neuron> neurons;

	layer() {}

	~layer() {}

	neuron& operator[](int index) // directly indexing the layer will return a reference to the neuron at the provided index
	{
		return neurons[index];
	}
};

class neural_network
{
public:
	vector<layer> layers;
	int num_layers = 4;
	int neurons_per_layer[4] = { 784, 16, 16, 10 }; // 784 = 28 * 28 (28x28 pixels)

	neural_network()
	{
		// create layers
		for (int i = 0; i < num_layers; i++)
		{
			layer layer;
			layer.index = i;
			layer.num_neurons = neurons_per_layer[i];
			layers.push_back(layer);
		}

		// create neurons
		for (layer& layer : layers)
		{
			int num_neurons = layer.num_neurons;
			for (int i = 0; i < num_neurons; i++)
			{
				char type = 'h'; // hidden
				if (layer.index == 0)
				{
					type = 'i';  // input
				}
				else if (layer.index == num_layers - 1)
				{
					type = 'o';  // output
				}

				neuron neuron;
				neuron.type = type;
				neuron.index = i;
				layer.neurons.push_back(neuron);
			}
		}

		// create weights (random value)
		for (layer& _layer : layers)
		{
			if (_layer.index == 0) // skip the first layer
			{
				continue;
			}
			for (neuron& neuron : _layer.neurons)
			{
				// the number of weights is equal to the number of neurons in the previous layer
				layer previous_layer = layers[_layer.index - 1];
				int num_weights = previous_layer.num_neurons;
				for (int i = 0; i < num_weights; i++)
				{
					neuron.weights.push_back(random_float());
				}
			}
		}
	}

	~neural_network() {}

	float random_float()
	{
		return rand() / (float)RAND_MAX;
	}

	layer& operator[](int index) // directly indexing the neural network will return a reference to the layer at the provided index
	{
		return layers[index];
	}
};

int main()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	sf::RenderWindow window(sf::VideoMode(500, 500), "Hello SFML", sf::Style::Close);
	sf::Event event;

	{
		neural_network neural_network;
	}

	while (window.isOpen())
	{
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
		}

		window.clear(sf::Color(0, 0, 0));
		window.display();
	}

	return 0;
}