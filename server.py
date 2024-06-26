import argparse
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import io
import base64
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

app = Flask(__name__)
socketio = SocketIO(app)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, hidden_neurons=128, output_neurons=10):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(28*28, hidden_neurons)  # Adjusted for 28x28 input
        self.output = nn.Linear(hidden_neurons, output_neurons)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

    def get_weight_bias_distributions(self):
        weights = self.hidden.weight.data.flatten().tolist()
        biases = self.hidden.bias.data.flatten().tolist()
        return weights, biases

    def get_activation_distribution(self, x):
        with torch.no_grad():
            x = x.view(-1, 28*28)
            hidden_activations = self.activation(self.hidden(x)).flatten().tolist()
        return hidden_activations

# Load Fashion-MNIST dataset
def load_fashion_mnist(limit_per_class=1000, validation_split=0.2):
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    class_counts = {i: 0 for i in range(10)}
    train_data = []
    train_targets = []
    val_data = []
    val_targets = []
    
    for img, label in full_dataset:
        if class_counts[label] < limit_per_class:
            inverted_img = 1 - img.numpy()  # Invert the image
            if np.random.rand() < validation_split:
                val_data.append(inverted_img.flatten())
                val_targets.append(label)
            else:
                train_data.append(inverted_img.flatten())
                train_targets.append(label)
            class_counts[label] += 1
        
        if all(count >= limit_per_class for count in class_counts.values()):
            break
    
    return (np.array(train_data), np.array(train_targets)), (np.array(val_data), np.array(val_targets))

# Training loop
def train(model, criterion, optimizer, train_dataloader, val_dataloader=None, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    training_metrics = {'epoch': [], 'training_loss': [], 'validation_loss': [], 'training_accuracy': [], 'validation_accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss /= len(train_dataloader)
        train_accuracy = 100. * train_correct / train_total
        
        # Validation
        val_loss = 0.0
        val_accuracy = 0.0
        if val_dataloader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_loss /= len(val_dataloader)
            val_accuracy = 100. * val_correct / val_total
        
        # Log metrics
        training_metrics['epoch'].append(epoch)
        training_metrics['training_loss'].append(train_loss)
        training_metrics['validation_loss'].append(val_loss)
        training_metrics['training_accuracy'].append(train_accuracy)
        training_metrics['validation_accuracy'].append(val_accuracy)
        
        if epoch % 1 == 0:
            log_message = f"Epoch {epoch}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
            if val_dataloader is not None:
                log_message += f", Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
            print(log_message)
            socketio.emit('log', {'message': log_message})
            socketio.emit('training_metrics', training_metrics)
    
    model.cpu()

@app.route('/')
def index():
    num_classes = 10  # Fashion-MNIST has 10 classes
    return render_template('index.html', num_classes=num_classes)

@app.route('/predict', methods=['POST'])
def predict():
    input_grid = request.json['inputGrid']
    input_tensor = torch.tensor(np.array(input_grid).flatten()[np.newaxis, :], dtype=torch.float32)
    with torch.no_grad():
        hidden_activations = model.activation(model.hidden(input_tensor.view(-1, 28*28))).numpy()
        output_activations = model(input_tensor).numpy()

    predicted_class = int(np.argmax(output_activations))
    result = {
        'predictedClass': predicted_class,
        'hiddenActivations': hidden_activations.tolist(),
        'outputActivations': output_activations.tolist()
    }
    return jsonify(result)

@app.route('/confusion_matrix', methods=['GET'])
def get_confusion_matrix():
    global model, train_data, train_targets, val_data, val_targets
    model.eval()
    with torch.no_grad():
        # Combine train and validation data for the confusion matrix
        all_data = np.concatenate((train_data, val_data), axis=0)
        all_targets = np.concatenate((train_targets, val_targets), axis=0)
        
        outputs = model(torch.tensor(all_data, dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)
    cm = confusion_matrix(all_targets, predicted.numpy())
    return jsonify({'confusionMatrix': cm.tolist()})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    image_data = request.json['imageData']
    image_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = 1 - img_array  # Invert the image
    return jsonify({'inputGrid': img_array.tolist()})

@app.route('/train', methods=['POST'])
def train_model():
    global model, train_data, train_targets, val_data, val_targets
    epochs = int(request.json['epochs'])

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_targets, dtype=torch.long))
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs)
    model.eval()
    return jsonify({'message': 'Training completed'})

@app.route('/clear', methods=['POST'])
def clear_model_data():
    global model
    model = SimpleNN(args.hidden_neurons, 10)
    socketio.emit('log', {'message': 'Model data cleared'})
    return jsonify({'message': 'Model data cleared'})

@app.route('/training_data', methods=['GET'])
def get_training_data():
    global train_data
    training_data = []
    for img in train_data:
        img = Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        training_data.append(img_str)
    return jsonify({'trainingData': training_data})

@app.route('/validation_data', methods=['GET'])
def get_validation_data():
    global val_data
    validation_data = []
    for img in val_data:
        img = Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        validation_data.append(img_str)
    return jsonify({'trainingData': validation_data})

@app.route('/train_single', methods=['POST'])
def train_single_example():
    global model
    data = request.json
    input_grid = np.array(data['inputGrid'])
    class_label = data['classLabel']

    input_tensor = torch.tensor(input_grid.flatten()[np.newaxis, :], dtype=torch.float32)
    target_tensor = torch.tensor([class_label], dtype=torch.long)

    train_dataset = TensorDataset(input_tensor, target_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, criterion, optimizer, train_dataloader, epochs=1)
    socketio.emit('log', {'message': f'Trained on single example of class {class_label}'})
    model.eval()
    return jsonify({'message': f'Trained on single example of class {class_label}'})

@app.route('/load_training_image', methods=['POST'])
def load_training_image():
    global train_data
    index = request.json['index']
    img = train_data[index].reshape(28, 28)
    input_grid = img.tolist()
    return jsonify({'inputGrid': input_grid})

@app.route('/load_validation_image', methods=['POST'])
def load_validation_image():
    global val_data
    index = request.json['index']
    img = val_data[index].reshape(28, 28)
    input_grid = img.tolist()
    return jsonify({'inputGrid': input_grid})

@app.route('/training_metrics', methods=['GET'])
def get_training_metrics():
    return jsonify(training_metrics)

@app.route('/network_visualization', methods=['POST'])
def get_network_visualization():
    input_grid = request.json['inputGrid']
    input_tensor = torch.tensor(np.array(input_grid).flatten()[np.newaxis, :], dtype=torch.float32)

    with torch.no_grad():
        hidden_activations = model.activation(model.hidden(input_tensor.view(-1, 28*28))).numpy()
        output_activations = model(input_tensor).numpy()

    hidden_weights = model.hidden.weight.data.numpy()
    output_weights = model.output.weight.data.numpy()

    return jsonify({
        'inputActivations': input_tensor.flatten().tolist(),
        'hiddenActivations': hidden_activations.tolist(),
        'outputActivations': output_activations.tolist(),
        'hiddenWeights': hidden_weights.tolist(),
        'outputWeights': output_weights.tolist()
    })

@app.route('/update_hidden_neurons', methods=['POST'])
def update_hidden_neurons():
    global model
    new_hidden_neurons = int(request.json['hiddenNeurons'])
    model = SimpleNN(hidden_neurons=new_hidden_neurons, output_neurons=10)
    model.eval()
    socketio.emit('log', {'message': f'Updated hidden neurons to {new_hidden_neurons}'})
    return jsonify({'message': f'Updated hidden neurons to {new_hidden_neurons}'})

@app.route('/distributions', methods=['POST'])
def get_distributions():
    input_grid = request.json['inputGrid']
    input_tensor = torch.tensor(np.array(input_grid).flatten()[np.newaxis, :], dtype=torch.float32)

    weights, biases = model.get_weight_bias_distributions()
    activations = model.get_activation_distribution(input_tensor)
    
    with torch.no_grad():
        output = model(input_tensor)
        confidence = torch.nn.functional.softmax(output, dim=1).flatten().tolist()

    def create_histogram(data, title):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=30)
        ax.set_title(title)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    weight_hist = create_histogram(weights, 'Weight Distribution')
    bias_hist = create_histogram(biases, 'Bias Distribution')
    activation_hist = create_histogram(activations, 'Activation Distribution')

    return jsonify({
        'weightHist': weight_hist,
        'biasHist': bias_hist,
        'activationHist': activation_hist,
        'confidence': confidence
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a simple neural network on the Fashion-MNIST dataset.')
    parser.add_argument('--hidden_neurons', type=int, default=128, help='Number of neurons in the hidden layer (default: 128)')
    parser.add_argument('--limit_per_class', type=int, default=200, help='Number of samples per class for training (default: 100)')

    args = parser.parse_args()

    (train_data, train_targets), (val_data, val_targets) = load_fashion_mnist(limit_per_class=args.limit_per_class)

    model = SimpleNN(hidden_neurons=args.hidden_neurons, output_neurons=10)
    model.eval()

    training_metrics = {'epoch': [], 'training_loss': [], 'validation_loss': [], 'training_accuracy': [], 'validation_accuracy': []}

    socketio.run(app, host='0.0.0.0', debug=True, port=8000)