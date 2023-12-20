use rand::Rng;

struct ConvLayer {
    weights: Vec<f64>,
    biases: Vec<f64>,
}

impl ConvLayer {
    fn new(input_size: usize, kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        ConvLayer {
            weights: (0..input_size - kernel_size + 1).map(|_| rng.gen()).collect(),
            biases: vec![rng.gen(); kernel_size],
        }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.; self.biases.len()];
        for i in 0..input.len() - self.weights.len() + 1 {
            let mut sum = 0.0;
            for j in 0..self.weights.len() {
                sum += self.weights[j] * input[i + j];
            }
            output[i / (input.len() - kernel_size + 1)] = sum;
        }
        output
    }

    fn backprop(&mut self, errors: &[f64], input: &[f64]) {
        for i in 0..self.weights.len() {
            self.weights[i] -= self.learning_rate * errors[i / (input.len() - kernel_size + 1)] * input[i];
        }
        for j in 0..self.biases.len() {
            self.biases[j] -= self.learning_rate * errors[j] as f64;
        }
    }
}

struct CNN {
    layers: Vec<Arc<Mutex<ConvLayer>>>,
}

impl CNN {
    fn new(input_size: usize, kernel_sizes: &[usize]) -> Self {
        let mut layers = vec![];
        for kernel_size in kernel_sizes {
            layers.push(Arc::new(Mutex::new(ConvLayer::new(input_size, *kernel_size))));
            input_size *= 2;
        }
        CNN { layers }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.; self.layers[0].lock().unwrap().biases.len()];
        for layer in self.layers.iter().skip(1) {
            let prev_output = self.forward_prev_layer(&self.layers[..layer.as_ref().drwaw().index()]);
            let conv_layer = layer.lock().unwrap();
            output = prev_output.iter().map(|&x| x).zip(conv_layer.forward(input)).collect();
        }
        self.layers[0].lock().unwrap().forward(input)
    }

    fn backprop(&mut self, errors: &[f64], input: &[f64]) {
        let mut last_layer = self.layers.last_mut().unwrap();
        last_layer.backprop(&errors, &input);
        for layer in (self.layers.iter()).rev() {
            if let Some(conv_layer) = layer.downcast_ref::<ConvLayer>() {
                conv_layer.backprop(&errors, &last_layer.lock().unwrap().forward(input));
            }
        }
    }

    fn forward_prev_layer(&self, layers: &[Arc<Mutex<ConvLayer>>]) -> Vec<f64> {
        let mut output = vec![0.; self.layers[0].lock().unwrap().biases.len()];
        for layer in layers.iter() {
            let conv_layer = layer.lock().unwrap();
            let prev_output = if layers.len() == 1 { input } else { output.clone() };
            output = prev_output.iter().map(|&x| x).zip(conv_layer.forward(&prev_output)).collect();
        }
        output
    }
}

struct ImageNetDataset {
    images: Vec<Arc<Image>>,
    labels: Vec<f64>,
}

impl ImageNetDataset {
    fn new(images: Vec<Arc<Image>>, labels: Vec<f64>) -> Self {
        ImageNetDataset { images, labels }
    }

    fn random_split(&self, split: f64) -> (ImageNetDataset, ImageNetDataset) {
        let mut rng = rand::thread_rng();
        let len = self.images.len();
        let split_idx = rng.gen::<usize>((0..len).into());
        (
            ImageNetDataset {
                images: self.images[..split_idx].to_vec(),
                labels: self.labels[..split_idx].to_vec()
            },
            ImageNetDataset {
                images: self.images[split_idx..].to_vec(),
                labels: self.labels[split_idx..].to_vec()
            }
        )
    }
}

struct CNNTrainer {
    model: Arc<Mutex<CNN>>,
    dataset: ImageNetDataset,
    learning_rate: f64,
}

impl CNNTrainer {
    fn new(model: Arc<Mutex<CNN>>, dataset: ImageNetDataset, learning_rate: f64) -> Self {
        CNNTrainer { model, dataset, learning_rate }
    }

    fn train(&mut self, epochs: u32) {
        for _ in 0..epochs {
            let (images, labels) = self.dataset.random_split(0.8);
            self.train_epoch(images, labels);
            println!("Epoch {}/{} completed!", _ + 1, epochs);
        }
    }

    fn train_epoch(&mut self, images: Vec<Arc<Image>>, labels: Vec<f64>) {
        let mut data = vec![(); images.len() * 32].map(|_| 0.0 as f64);
        for (image, label) in images.iter().zip(labels.iter()) {
            let pixels = image.pixels().to_vec();
            data[..32].clone_from_slice(&pixels);
            let output = self.model.lock().unwrap().forward(&data);
            let error = label - output;
            self.model.lock().unwrap().backprop(&vec![error], &pixels);
        }
    }
}

fn main() {
    let (images, labels) = ImageNetDataset::new(Arc::new(Image::new()));
    let model = Arc::new(Mutex::new(CNN::new()));
    let trainer = CNNTrainer::new(model.clone(), images.clone(), 0.1);
    trainer.train(5);
}
