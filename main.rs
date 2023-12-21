use std::f64;
use rand::*;
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Clone for Neuron {
    fn clone(&self) -> Self {
        let mut new_weights = self.weights.clone();
        let mut rng = rand::thread_rng();
        for weight in &mut new_weights {
            *weight = rng.gen::<f64>();
        }

        Neuron {
            weights: new_weights,
            bias: self.bias.clone(),
        }
    }
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let mut weights = vec![0.0; num_inputs];
        let mut rng = rand::thread_rng();
        for weight in &mut weights {
            *weight = rng.gen::<f64>();
        }

        Neuron {
            weights,
            bias: 0.0,
        }
    }

    fn forward(&self, inputs: Vec<f64>) -> f64 {
        let mut output = self.weights[0] * inputs[0] + self.bias;
        for i in 1..self.weights.len() {
            output += self.weights[i] * inputs[i];
        }

        output
    }
}

fn main() {
    let num_inputs = 7;
    let mut neurons = vec![Neuron::new(num_inputs); 24];

    // Sample input data
    let inputs = vec![0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3];
    let target = 1.0;

    for neuron in &mut neurons {
        let output = neuron.forward(inputs.clone());
        println!("Neuron Output: {}", output);
    }
}
