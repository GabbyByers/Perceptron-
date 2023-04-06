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

	vector<sf::Image> sfml_images;
	sf::Font font;
	sf::Text text;

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
		draw_setup();
	}

	void draw_setup()
	{
		sfml_images.clear();
		for (int im = 0; im < num_labels; im++)
		{
			sf::Image image;
			image.create(28, 28, sf::Color::Black);

			for (int i = 0; i < 28; i++)
			{
				for (int j = 0; j < 28; j++)
				{
					int color = images[(im * 784) + (i * 28) + j];
					image.setPixel(j, i, sf::Color(color, color, color));
				}
			}

			sfml_images.push_back(image);
		}

		font.loadFromFile("TimesNewRoman.ttf");
		text.setFont(font);
		text.setPosition(50, 50 + 112);
		text.setCharacterSize(20);
		text.setFillColor(sf::Color::White);
	}

	void draw_image(sf::RenderWindow& window, int image_index)
	{
		int scale = 4;
		float x = 50.0f;
		float y = 50.0f;

		// digit
		sf::Image& image = sfml_images[image_index];\
		sf::Texture texture;
		texture.loadFromImage(image);\
		sf::Sprite sprite;
		sprite.setTexture(texture);\
		sprite.setPosition(x, y);
		sprite.setScale(scale, scale);\
		window.draw(sprite);

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
		text.setString("Value: " + to_string(labels[image_index]));
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

		draw_setup();
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

	void propagate(int image_index)
	{
		layer& input_layer = layers[0];
		for (int i = 0; i < 28; i++)
		{
			for (int j = 0; j < 28; j++)
			{
				float brightness = mnist_database.images[(image_index * 784) + (i * 28) + j] / 255.0f;
				input_layer[(i * 28) + j].activation = brightness;
			}
		}
	}

	vector<sf::Vertex> quads;
	vector<sf::Vertex> boxes;

	void draw_setup()
	{
		for (layer& layer : layers)
		{
			for (neuron& neuron : layer.neurons)
			{
				if (layer.index == 0)
				{
					if (neuron.index == 28)
					{
						break;
					}
				}

				int color = neuron.activation * 255;
				float x = layer.index * 200 + 212;
				float y = neuron.index * 20 + 50;

				if (layer.index == 1) { y += 6 * 20; }
				if (layer.index == 2) { y += 6 * 20; }
				if (layer.index == 3) { y += 9 * 20; }

				quads.push_back(sf::Vertex(sf::Vector2f(x, y),           sf::Color(100, 100, 100)));
				quads.push_back(sf::Vertex(sf::Vector2f(x + 10, y),      sf::Color(100, 100, 100)));
				quads.push_back(sf::Vertex(sf::Vector2f(x + 10, y + 10), sf::Color(100, 100, 100)));
				quads.push_back(sf::Vertex(sf::Vector2f(x, y + 10),      sf::Color(100, 100, 100)));

				boxes.push_back(sf::Vertex(sf::Vector2f(x, y),           sf::Color::White));
				boxes.push_back(sf::Vertex(sf::Vector2f(x + 10, y),      sf::Color::White));
				boxes.push_back(sf::Vertex(sf::Vector2f(x + 10, y),      sf::Color::White));
				boxes.push_back(sf::Vertex(sf::Vector2f(x + 10, y + 10), sf::Color::White));
				boxes.push_back(sf::Vertex(sf::Vector2f(x + 10, y + 10), sf::Color::White));
				boxes.push_back(sf::Vertex(sf::Vector2f(x, y + 10),      sf::Color::White));
				boxes.push_back(sf::Vertex(sf::Vector2f(x, y + 10),      sf::Color::White));
				boxes.push_back(sf::Vertex(sf::Vector2f(x, y),           sf::Color::White));
			}
		}
	}

	void draw(sf::RenderWindow& window, int index)
	{
		mnist_database.draw_image(window, index);

		propagate(index);

		layer& input_layer = layers[0];
		for (int i = 0; i < 28; i++)
		{
			int color = input_layer[(i * 28) + i].activation * 255;
			for (int j = 0; j < 4; j++)
			{
				quads[(i * 4) + j].color = sf::Color(color, color, color);
			}
		}

		//for (layer& layer : layers)
		//{
		//	if (layer.index == 0)
		//	{
		//		continue;
		//	}
		//	for (neuron& neuron : layer.neurons)
		//	{
		//		for (int i = 0; i < 4; i++)
		//		{
		//			quads
		//		}
		//	}
		//}

		window.draw(&quads[0], quads.size(), sf::Quads);
		window.draw(&boxes[0], boxes.size(), sf::Lines);
	}
};

int main()
{
	sf::RenderWindow window(sf::VideoMode(1000, 650), "Perceptron", sf::Style::Close);
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

		if (index >= 60000)
		{
			index = 0;
		}

		window.display();
	}

	return 0;
}