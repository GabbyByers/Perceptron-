#include <iostream>
#include <fstream>
#include <vector>
#include "SFML/Graphics.hpp"
using namespace std;

#include <chrono>
#include <thread>
using namespace std::this_thread;
using namespace std::chrono;


class mnist_database
{
public:
	int num_pixels = 60000 * 28 * 28;
	int num_labels = 60000;
	vector<unsigned char> images;
	vector<unsigned char> labels;

	mnist_database()
	{
		ifstream image_file;
		ifstream label_file;
		char byte = '\0';

		// load in the digit images
		image_file.open("IDX/train-images.idx3-ubyte", ios::binary);
		if (!image_file.is_open())
		{
			cout << "Oh no! I couldn't open this file! :c\n";
		}
		for (int i = 0; i < 16; i++) // Skip the first 16 bytes
		{
			image_file.read(&byte, 1);
		}
		for (int i = 0; i < num_pixels; i++) // Read the rest of the bytes
		{
			image_file.read(&byte, 1);
			images.push_back((unsigned char)byte);
		}

		// load in the digit labels
		label_file.open("IDX/train-labels.idx1-ubyte", ios::binary);
		if (!label_file.is_open())
		{
			cout << "Oh no! I couldn't open this file! :c\n";
		}
		for (int i = 0; i < 8; i++) // Skip the first 8 bytes
		{
			label_file.read(&byte, 1);
		}
		for (int i = 0; i < num_labels; i++) // Read the rest of the bytes
		{
			label_file.read(&byte, 1);
			labels.push_back((unsigned char)byte);
		}

		image_file.close();
		label_file.close();
	}

	void draw_image(sf::RenderWindow& window, int image_index)
	{
		int scale = 4;
		float x = 50.0f;
		float y = 50.0f;

		// digit
		sf::Vertex* pixels = new sf::Vertex[784];
		for (int i = 0; i < 28; i++)
		{
			for (int j = 0; j < 28; j++)
			{
				int index = (image_index * 784) + (i * 28) + j;
				int color = images[index];
				pixels[i * 28 + j] = sf::Vertex(sf::Vector2f(x + (j * scale), y + (i * scale)), sf::Color(color, color, color));
			}
		}

		sf::Transform transform;
		for (int i = 0; i < scale; i++)
		{
			for (int j = 0; j < scale; j++)
			{
				transform = sf::Transform();
				transform.translate(i, j);
				window.draw(pixels, 784, sf::Points, transform);
			}
		}

		delete[] pixels;

		// bounding box
		sf::Vertex box[5] =
		{
			sf::Vertex(sf::Vector2f(x, y), sf::Color::White),
			sf::Vertex(sf::Vector2f(x + scale * 28, y), sf::Color::White),
			sf::Vertex(sf::Vector2f(x + scale * 28, y + scale * 28), sf::Color::White),
			sf::Vertex(sf::Vector2f(x, y + scale * 28), sf::Color::White),
			sf::Vertex(sf::Vector2f(x, y), sf::Color::White)
		};

		window.draw(box, 5, sf::LinesStrip);

		// text
		sf::Font font;
		font.loadFromFile("TimesNewRoman.ttf");

		sf::Text text;
		text.setFont(font);
		text.setString("Value: " + to_string(labels[image_index]));
		text.setPosition(x, y + scale * 28);
		text.setCharacterSize(20);
		text.setFillColor(sf::Color::White);
		window.draw(text);
	}
};

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

	float sigmoid(float x)
	{
		return 1 / (1 + exp(-x));
	}

	float sigmoid_derivative(float x)
	{
		return sigmoid(x) * (1 - sigmoid(x));
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

	mnist_database mnist_database;

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
					type = 'i'; // input
				}
				else if (layer.index == num_layers - 1)
				{
					type = 'o'; // output
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

	void draw(sf::RenderWindow& window, int index)
	{
		mnist_database.draw_image(window, index);


		float x = 212.0f;
		float y = 50.0f;
		float size = 20.0f;

		vector<sf::Vertex> neuron_solid;
		for (int l = 0; l < 4; l++)
		{
			for (int n = 0; n < 16; n++)
			{
				if (l == 0)
				{
					if (n == 7) { continue; }
					if (n == 8) { continue; }
				}
				if (l == 3)
				{
					if (n == 0) { continue; }
					if (n == 1) { continue; }
					if (n == 2) { continue; }
					if (n == 13) { continue; }
					if (n == 14) { continue; }
					if (n == 15) { continue; }
				}
				int color = 100; //int color = 255 * layers[l][n].activation;
				float x0 = x + 10 * l * size;
				float y0 = y + 2 * n * size;
				neuron_solid.push_back(sf::Vertex(sf::Vector2f(x0, y0), sf::Color(color, color, color)));
				neuron_solid.push_back(sf::Vertex(sf::Vector2f(x0 + size, y0), sf::Color(color, color, color)));
				neuron_solid.push_back(sf::Vertex(sf::Vector2f(x0 + size, y0 + size), sf::Color(color, color, color)));
				neuron_solid.push_back(sf::Vertex(sf::Vector2f(x0, y0 + size), sf::Color(color, color, color)));
			}
		}

		vector<sf::Vertex> neuron_box;
		for (int l = 0; l < 4; l++)
		{
			for (int n = 0; n < 16; n++)
			{
				if (l == 0)
				{
					if (n == 7) { continue; }
					if (n == 8) { continue; }
				}
				if (l == 3)
				{
					if (n == 0) { continue; }
					if (n == 1) { continue; }
					if (n == 2) { continue; }
					if (n == 13) { continue; }
					if (n == 14) { continue; }
					if (n == 15) { continue; }
				}
				float x0 = x + 10 * l * size;
				float y0 = y + 2 * n * size;
				sf::Color color = sf::Color::White;
				neuron_box.push_back(sf::Vertex(sf::Vector2f(x0, y0), color));
				neuron_box.push_back(sf::Vertex(sf::Vector2f(x0 + size, y0), color));
				neuron_box.push_back(sf::Vertex(sf::Vector2f(x0 + size, y0), color));
				neuron_box.push_back(sf::Vertex(sf::Vector2f(x0 + size, y0 + size), color));
				neuron_box.push_back(sf::Vertex(sf::Vector2f(x0 + size, y0 + size), color));
				neuron_box.push_back(sf::Vertex(sf::Vector2f(x0, y0 + size), color));
				neuron_box.push_back(sf::Vertex(sf::Vector2f(x0, y0 + size), color));
				neuron_box.push_back(sf::Vertex(sf::Vector2f(x0, y0), color));
			}
		}

		window.draw(&neuron_solid[0], neuron_solid.size(), sf::Quads);
		window.draw(&neuron_box[0], neuron_box.size(), sf::Lines);
	}
};

int main()
{
	sf::RenderWindow window(sf::VideoMode(1000, 720), "Hello SFML", sf::Style::Close);
	sf::Event event;
	
	neural_network neural_network;
	int index = 0;

	while (window.isOpen())
	{
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
		}

		//sleep_for(milliseconds(50));

		window.clear(sf::Color(0, 0, 0));

		neural_network.draw(window, index);
		index++;

		window.display();
	}

	return 0;
}