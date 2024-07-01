$(document).ready(function() {
    const GRID_SIZE = 28;
    const CELL_SIZE = 10;
    let inputGrid = Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(1));
    let canvas = document.getElementById('inputCanvas');
    let ctx = canvas.getContext('2d');
    let extendedLineCanvas = document.getElementById('extendedLineCanvas');
    let extendedLineCtx = extendedLineCanvas.getContext('2d');
    
    let currentDataset = 'fashion_mnist';
    let currentModel = 'simple';

    let lossChart;
    let accuracyChart;

    let currentLabels = {};

    fetchCurrentState();
    predict();
    fetchLabels();
    updateSwitchButtonText();
    drawInputGrid();
    drawExtendedLine();
    initializeCharts();
    updateSwitchDatasetButtonText();
    prepareData();


    $('#switchModelButton').click(function() {
        const newModel = currentModel === 'simple' ? 'advanced' : 'simple';
        $.ajax({
            url: '/switch_model',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ modelType: newModel }),
            success: function(response) {
                console.log(response.message);
                fetchCurrentState();  // This will update currentModel and UI
                alert(`Switched to ${newModel} model`);
                predict();  // Update the prediction with the new model
            },
            error: function(xhr, status, error) {
                console.error("Error switching model:", error);
                alert("Failed to switch model. See console for details.");
            }
        });
    });

    function fetchLabels() {
        $.ajax({
            url: '/get_labels',
            method: 'GET',
            success: function(response) {
                currentLabels = response;
                updateUIWithLabels();
            },
            error: function(xhr, status, error) {
                console.error("Error fetching labels:", error);
            }
        });
    }

    function updateUIWithLabels() {
        // Update output layer labels
        $('#outputLayer .neuron-number').each(function(index) {
            $(this).text(currentLabels[index]);
        });
    
        // Update any other UI elements that use labels
        // For example, if you have a dropdown for selecting classes:
        // var $dropdown = $('#classDropdown');
        // $dropdown.empty();
        // Object.entries(currentLabels).forEach(([key, value]) => {
        //     $dropdown.append($('<option></option>').attr('value', key).text(value));
        // });
    }

    function updateSwitchButtonText() {
        const buttonText = currentModel === 'simple' ? 'Switch to Advanced Model' : 'Switch to Simple Model';
        $('#switchModelButton').html(`<i class="fas fa-exchange-alt"></i> ${buttonText}`);
    }

    $('#distributionsButton').click(function() {
        updateDistributions();
        $('#distributionsModal').modal('show');
    });

    function updateDistributions() {
        $.ajax({
            url: '/distributions',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid }),
            success: function(response) {
                console.log("Received distribution data:", response);
                $('#weightHistogram').attr('src', 'data:image/png;base64,' + response.weightHist);
                $('#biasHistogram').attr('src', 'data:image/png;base64,' + response.biasHist);
                $('#activationHistogram').attr('src', 'data:image/png;base64,' + response.activationHist);
                updateConfidenceChart(response.confidence);
                console.log("Updated histograms and confidence chart");
            },
            error: function(xhr, status, error) {
                console.error("Error fetching distributions:", error);
            }
        });
    }

    function fetchCurrentState() {
        $.ajax({
            url: '/get_current_state',
            method: 'GET',
            success: function(response) {
                currentDataset = response.currentDataset;
                currentModel = response.currentModel;
                updateUIWithCurrentState();
            },
            error: function(xhr, status, error) {
                console.error("Error fetching current state:", error);
            }
        });
    }

    function updateHiddenNeurons() {
        var hiddenNeurons = $('#hiddenNeuronsInput').val();
        $.ajax({
            url: '/update_hidden_neurons',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ hiddenNeurons: hiddenNeurons }),
            success: function(response) {
                console.log(response.message);
                alert('Hidden neurons updated. The model has been reset.');
                clear();
            }
        });
    }
    
    $('#updateHiddenNeuronsButton').click(updateHiddenNeurons);

    function drawNetworkVisualization(data) {
        var canvas = document.getElementById('networkVisualizationCanvas');
        var ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Set canvas size
        canvas.width = 1000;
        canvas.height = 600;
        
        if (currentModel === 'simple') {
            // Draw simple model
            var inputNeurons = 28 * 28;
            var hiddenNeurons = data.hiddenActivations.length;
            var outputNeurons = data.outputActivations.length;
            
            var inputLayer = Array.from({length: inputNeurons}, (_, i) => ({x: 50, y: 20 + i * (560 / inputNeurons)}));
            var hiddenLayer = Array.from({length: hiddenNeurons}, (_, i) => ({x: 500, y: 20 + i * (560 / hiddenNeurons)}));
            var outputLayer = Array.from({length: outputNeurons}, (_, i) => ({x: 950, y: 20 + i * (560 / outputNeurons)}));
            
            // Draw connections
            drawLayerConnections(ctx, inputLayer, hiddenLayer, data.hiddenWeights);
            drawLayerConnections(ctx, hiddenLayer, outputLayer, data.outputWeights);

            
            // Draw neurons
            drawNeurons(ctx, inputLayer, data.inputActivations);
            drawNeurons(ctx, hiddenLayer, data.hiddenActivations);
            drawNeurons(ctx, outputLayer, data.outputActivations);
        } else {
            // Draw advanced model
            var inputNeurons = 28 * 28;
            var conv1Neurons = data.conv1Activations ? Math.min(data.conv1Activations.length, 32) : 0;
            var conv2Neurons = data.conv2Activations ? Math.min(data.conv2Activations.length, 64) : 0;
            var fc1Neurons = data.hiddenActivations ? Math.min(data.hiddenActivations.length, 128) : 0;
            var outputNeurons = data.outputActivations ? data.outputActivations.length : 0;
            
            var inputLayer = Array.from({length: inputNeurons}, (_, i) => ({x: 50, y: 20 + i * (560 / inputNeurons)}));
            var conv1Layer = Array.from({length: conv1Neurons}, (_, i) => ({x: 250, y: 20 + i * (560 / conv1Neurons)}));
            var conv2Layer = Array.from({length: conv2Neurons}, (_, i) => ({x: 450, y: 20 + i * (560 / conv2Neurons)}));
            var fc1Layer = Array.from({length: fc1Neurons}, (_, i) => ({x: 650, y: 20 + i * (560 / fc1Neurons)}));
            var outputLayer = Array.from({length: outputNeurons}, (_, i) => ({x: 850, y: 20 + i * (560 / outputNeurons)}));
            
            // Draw connections
            drawLayerConnections(ctx, inputLayer, conv1Layer, data.conv1Weights);
            drawLayerConnections(ctx, conv1Layer, conv2Layer, data.conv2Weights);
            drawLayerConnections(ctx, conv2Layer, fc1Layer, data.fc1Weights);
            drawLayerConnections(ctx, fc1Layer, outputLayer, data.fc2Weights);

            // Draw neurons
            drawNeurons(ctx, inputLayer, data.inputActivations);
            if (data.conv1Activations) drawNeurons(ctx, conv1Layer, data.conv1Activations);
            if (data.conv2Activations) drawNeurons(ctx, conv2Layer, data.conv2Activations);
            if (data.hiddenActivations) drawNeurons(ctx, fc1Layer, data.hiddenActivations);
            if (data.outputActivations) drawNeurons(ctx, outputLayer, data.outputActivations);
        }
    }
    
    function drawLayerConnections(ctx, layer1, layer2, weights) {
        const connectionsPerNeuron = 5; // Adjust this value to increase/decrease density of connections
        
        //console.log("Weights:", weights); // Log the weights

        if (!weights || weights.length === 0) {
            console.log("No weights provided, using default");
            weights = Array(layer1.length * layer2.length).fill(0);
        }
        
        for (let i = 0; i < layer1.length; i++) {
            for (let j = 0; j < connectionsPerNeuron; j++) {
                const targetIndex = Math.floor(Math.random() * layer2.length);
                const weightIndex = i * layer2.length + targetIndex;
                const weight = weights[weightIndex] || 0;
                
                ctx.beginPath();
                ctx.moveTo(layer1[i].x, layer1[i].y);
                ctx.lineTo(layer2[targetIndex].x, layer2[targetIndex].y);
                
                ctx.strokeStyle = getWeightColor(weight);
                ctx.lineWidth = Math.abs(weight) * 2 + 0.5; // Adjust line width based on weight magnitude
                ctx.stroke();
            }
        }
    }
    
    function getWeightColor(weight) {
        // Amplify the weight to make colors more visible
        const amplifiedWeight = weight * 10;
        const normalizedWeight = Math.tanh(amplifiedWeight);
        
        let r, g, b;
        if (normalizedWeight < 0) {
            // Negative weights: blue
            r = 0;
            g = 0;
            b = Math.round(255 * (-normalizedWeight));
        } else {
            // Positive weights: red
            r = Math.round(255 * normalizedWeight);
            g = 0;
            b = 0;
        }
        
        // Increase base alpha to make lines more visible
        const alpha = Math.abs(normalizedWeight) * 0.5 + 0.2;
        
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    
    function drawNeurons(ctx, neurons, activations) {
        if (!activations) return;  // Skip if activations are undefined
        neurons.forEach((neuron, i) => {
            ctx.beginPath();
            ctx.arc(neuron.x, neuron.y, 3, 0, 2 * Math.PI);
            var activation = activations[i] || 0;  // Use 0 if activation is undefined
            ctx.fillStyle = getActivationColor(activation);
            ctx.fill();
        });
    }
    
    function getActivationColor(activation) {
        var r = Math.round(activation * 255);
        var g = 0;        
        var b = Math.round((1 - activation) * 255);
        return `rgb(${r},${g},${b})`;
    }
    
    function showNetworkVisualization() {
        $.ajax({
            url: '/network_visualization',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid }),
            success: function(response) {
                drawNetworkVisualization(response);
                $('#networkVisualizationModal').modal('show');
            }
        });
    }

    function updateConfidenceChart(confidence) {
        var ctx = document.getElementById('confidenceChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Array.from({length: confidence.length}, (_, i) => i),
                datasets: [{
                    label: 'Confidence',
                    data: confidence,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    $('#networkVisualizationButton').click(showNetworkVisualization);

    $('#uploadButton').click(function() {
        $('#imageUpload').click();
    });
    
    $('#imageUpload').change(function(e) {
        var file = e.target.files[0];
        var reader = new FileReader();
        reader.onload = function(event) {
            var img = new Image();
            img.onload = function() {
                var canvas = document.createElement('canvas');
                var ctx = canvas.getContext('2d');
                canvas.width = 28;
                canvas.height = 28;
                ctx.drawImage(img, 0, 0, 28, 28);
                var imageData = ctx.getImageData(0, 0, 28, 28);
                var data = imageData.data;
                var grayScaleData = [];
                for (var i = 0; i < data.length; i += 4) {
                    var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                    grayScaleData.push(avg / 255);
                }
                inputGrid = [];
                for (var i = 0; i < 28; i++) {
                    inputGrid.push(grayScaleData.slice(i * 28, (i + 1) * 28));
                }
                drawInputGrid();
                drawExtendedLine();
                predict();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });
    
    function drawInputGrid() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    
        // Draw the coordinate numbers
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        for (var i = 0; i < GRID_SIZE; i++) {
            ctx.fillText(i, i * CELL_SIZE + CELL_SIZE / 2 - 5, 10);  // Top coordinates
            ctx.fillText(i, 5, i * CELL_SIZE + CELL_SIZE / 2 + 5);  // Left coordinates
        }
    
        // Draw the grid cells
        for (var i = 0; i < GRID_SIZE; i++) {
            for (var j = 0; j < GRID_SIZE; j++) {
                var value = inputGrid[i][j];
                var color = `rgb(${Math.round(value * 255)}, ${Math.round(value * 255)}, ${Math.round(value * 255)})`;
                ctx.fillStyle = color;
                ctx.fillRect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
    
        // Draw grid lines
        ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)';  // Light grey with low opacity
        ctx.beginPath();
        for (var i = 0; i <= GRID_SIZE; i++) {
            ctx.moveTo(i * CELL_SIZE, 0);
            ctx.lineTo(i * CELL_SIZE, canvas.height);
            ctx.moveTo(0, i * CELL_SIZE);
            ctx.lineTo(canvas.width, i * CELL_SIZE);
        }
        ctx.stroke();
    }

    function drawExtendedLine() {
        extendedLineCtx.clearRect(0, 0, extendedLineCanvas.width, extendedLineCanvas.height);
        
        const pixelWidth = 1;
        const pixelHeight = 2;
        const pixelsPerRow = Math.floor(extendedLineCanvas.width / pixelWidth);
        const numRows = Math.ceil((GRID_SIZE * GRID_SIZE) / pixelsPerRow);
    
        for (var i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            var row = Math.floor(i / GRID_SIZE);
            var col = i % GRID_SIZE;
            var value = inputGrid[row][col];
            
            var colorValue = Math.round(value * 255);
            var color = `rgb(${colorValue}, ${colorValue}, ${colorValue})`;
            
            extendedLineCtx.fillStyle = color;
    
            var xPos = (i % pixelsPerRow) * pixelWidth;
            var yPos = Math.floor(i / pixelsPerRow) * pixelHeight;
    
            extendedLineCtx.fillRect(xPos, yPos, pixelWidth, pixelHeight);
        }
    }

    function updateActivations(hiddenActivations, outputActivations) {
        var hiddenHtml = '';
        
        if (currentModel === 'simple') {
            // Existing code for simple model
            for (var i = 0; i < hiddenActivations[0].length; i++) {
                var activation = hiddenActivations[0][i];
                var imageIndex = getImageIndex(activation);
                hiddenHtml += '<div class="neuron hidden-neuron" style="background-image: url(\'static/images/' + imageIndex + '.svg\');" data-index="' + i + '"><span>' + activation.toFixed(2) + '</span></div>';
            }
        } else {
            // For advanced model, show a summary
            hiddenHtml = `
                <div class="advanced-model-summary">
                    <ul>
                        <li>Conv1: 32 filters (3x3)</li>
                        <li>Conv2: 64 filters (3x3)</li>
                        <li>MaxPool: 2x2</li>
                        <li>Dropout: 25%</li>
                        <li>FC: 128 neurons</li>
                    </ul>
                </div>
            `;
        }
        $('#hiddenLayer').html(hiddenHtml);
    
        var outputHtml = '';
        for (var i = 0; i < outputActivations[0].length; i++) {
            var activation = outputActivations[0][i];
            var imageIndex = getImageIndex(activation);
            outputHtml += '<div class="neuron-container" style="text-align: center;">';
            outputHtml += '<div class="neuron output-neuron" style="background-image: url(\'static/images/' + imageIndex + '.svg\');" data-index="' + i + '"><span>' + activation.toFixed(2) + '</span></div>';
            outputHtml += '<div class="neuron-number">' + currentLabels[i] + '</div>'; // Use dynamic labels
            outputHtml += '</div>';
        }
        $('#outputLayer').html(outputHtml);
    
        addNeuronClickHandlers();
    }
    
    function getImageIndex(activation) {
        // Maps the activation value (now from -2 to 2) inversely to an image index (1-7)
        var normalizedValue = (activation + 2) / 4; // Normalize to [0,1]
        var invertedIndex = 1 - normalizedValue; // Invert the mapping
        var index = Math.round(invertedIndex * 6) + 1; // Scale to [1,7]
        return Math.max(1, Math.min(7, index)); // Ensures the index stays within the range of available images
    }

    function predict() {
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid }),
            success: function(response) {
                var predictedClass = response.predictedClass;
                var hiddenActivations = response.hiddenActivations;
                var outputActivations = response.outputActivations;
                $('#result').text('Result: ' + currentLabels[predictedClass]);
                updateActivations(hiddenActivations, outputActivations);
            },
            error: function(xhr, status, error) {
                console.error("Error in prediction:", error);
                $('#result').text('Prediction failed. See console for details.');
            }
        });
    }

    function clear() {
        inputGrid = Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(1));
        drawInputGrid();
        drawExtendedLine();
        $('#result').text('');
        $('#hiddenLayer').empty();
        $('#outputLayer').empty();
        predict();
    }

    function trainModel(epochs) {
        $.ajax({
            url: '/train',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ epochs: epochs }),
            success: function(response) {
                console.log(response.message);
                showValidationData(response.validationData);
                predict();
                hideTrainingProgress();
            }
        });
    }

    function clearModelData() {
        $.ajax({
            url: '/clear',
            method: 'POST',
            success: function(response) {
                console.log(response.message);
                predict();
            }
        });
    }

    function showTrainingData(trainingData) {
        console.log("Showing training data");
        var trainingHtml = '';
        for (var i = 0; i < trainingData.length; i++) {
            var img = trainingData[i];
            var imgHtml = '<img src="data:image/png;base64,' + img.image + '" width="28" height="28" data-index="' + i + '" title="Class: ' + currentLabels[img.label] + '">';
            trainingHtml += imgHtml;
        }
        $('#trainingData').html(trainingHtml);
    }

    $(document).on('click', '#trainingData img', function() {
        var index = $(this).data('index');
        loadTrainingImage(index);
    });

    function loadTrainingImage(index) {
        $.ajax({
            url: '/load_training_image',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ index: index }),
            success: function(response) {
                inputGrid = response.inputGrid;
                drawInputGrid();
                drawExtendedLine();
                predict();
            }
        });
    }
    
    $('#validationDataSection').addClass('data-hidden');
    $('#trainingDataSection').addClass('data-hidden');

    function showValidationData(validationData) {
        var validationHtml = '';
        for (var i = 0; i < validationData.length; i++) {
            var item = validationData[i];
            var borderClass = item.is_correct ? 'border-success' : 'border-danger';
            var imgHtml = `<div class="validation-image-container">
                               <img src="data:image/png;base64,${item.image}" 
                                    width="28" height="28" 
                                    data-index="${i}" 
                                    class="validation-image ${borderClass}"
                                    title="Predicted: ${currentLabels[item.predicted]}\nActual: ${currentLabels[item.actual]}">
                           </div>`;
            validationHtml += imgHtml;
        }
        $('#validationData').html(validationHtml);
    }

    $(document).on('click', '#validationData img', function() {
        var index = $(this).data('index');
        loadValidationImage(index);
    });
 
    $('#switchDatasetButton').click(function() {
        const newDataset = currentDataset === 'fashion_mnist' ? 'qmnist' : 'fashion_mnist';
        $.ajax({
            url: '/switch_dataset',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ dataset: newDataset }),
            success: function(response) {
                console.log(response.message);
                fetchCurrentState();  // This will update currentDataset and UI
                alert(`Switched to ${newDataset} dataset`);
                prepareData();  // Reload the data with the new dataset
            },
            error: function(xhr, status, error) {
                console.error("Error switching dataset:", error);
                alert("Failed to switch dataset. See console for details.");
            }
        });
    });
    
    function updateSwitchDatasetButtonText() {
        const buttonText = currentDataset === 'fashion_mnist' ? 'Switch to QMNIST' : 'Switch to Fashion-MNIST';
        $('#switchDatasetButton').text(buttonText);
    }

    function updateSwitchModelButtonText() {
        const buttonText = currentModel === 'simple' ? 'Switch to Advanced Model' : 'Switch to Simple Model';
        $('#switchModelButton').text(buttonText);
    }

    function updateUIWithCurrentState() {
        // Update dataset switch button
        updateSwitchDatasetButtonText();
    
        // Update model switch button
        updateSwitchModelButtonText();
    
        // You might want to update other UI elements based on the current state
        // For example, you might want to show/hide certain elements depending on the model
        // if (currentModel === 'simple') {
        //     $('.simple-model-element').show();
        //     $('.advanced-model-element').hide();
        // } else {
        //     $('.simple-model-element').hide();
        //     $('.advanced-model-element').show();
        // }
    
        // Fetch labels for the current dataset
        fetchLabels();
    }

    function loadValidationImage(index) {
        $.ajax({
            url: '/load_validation_image',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ index: index }),
            success: function(response) {
                inputGrid = response.inputGrid;
                drawInputGrid();
                drawExtendedLine();
                predict();
            }
        });
    }

    let dataLoaded = false;
    let showingTrainingData = true;
    
    $('#dataButton').click(function() {        
        if (!dataLoaded) {
            prepareData();
            dataLoaded = true;
            $(this).html('<i class="fas fa-exchange-alt"></i> Show Validation Data');
        } else {
            $('#trainingDataSection, #validationDataSection').toggleClass('data-hidden');
            if (showingTrainingData) {
                $(this).html('<i class="fas fa-exchange-alt"></i> Show Training Data');
            } else {
                $(this).html('<i class="fas fa-exchange-alt"></i> Show Validation Data');
            }
            showingTrainingData = !showingTrainingData;
        }
    });
    
    function prepareData() {
        $.ajax({
            url: '/training_data',
            method: 'GET',
            success: function(trainingResponse) {
                showTrainingData(trainingResponse.trainingData);
                $('#trainingDataSection').removeClass('data-hidden');
                
                $.ajax({
                    url: '/validation_data',
                    method: 'GET',
                    success: function(validationResponse) {
                        showValidationData(validationResponse.validationData);
                        $('#validationDataSection').addClass('data-hidden');
                    }
                });
            }
        });
    }

    function trainSingleExample(classLabel) {
        $.ajax({
            url: '/train_single',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid, classLabel: classLabel }),
            success: function(response) {
                console.log(response.message);
                showValidationData(response.validationData);
                predict();
            }
        });
    }

    function addNeuronClickHandlers() {
        $('.hidden-neuron').click(function() {
            var index = $(this).data('index');
            console.log('Hidden neuron ' + index + ' clicked');
            // Implement the desired functionality here
        });

        $('.output-neuron').click(function() {
            var index = $(this).data('index');
            trainSingleExample(index);
        });
    }


    function initializeCharts() {
        var ctxLoss = document.getElementById('lossChart').getContext('2d');
        var ctxAccuracy = document.getElementById('accuracyChart').getContext('2d');
        
        lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    fill: false
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true }
                }
            }
        });
        
        accuracyChart = new Chart(ctxAccuracy, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1,
                    fill: false
                },
                {
                    label: 'Validation Accuracy',
                    data: [],
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true }
                }
            }
        });
    }

    $('#showConfusionMatrixButton').click(function() {
        $.ajax({
            url: '/confusion_matrix',
            method: 'GET',
            success: function(response) {
                displayConfusionMatrix(response.confusionMatrix);
                $('#confusionMatrixModal').modal('show');
            }
        });
    });
    
    function displayConfusionMatrix(confusionMatrix) {
        var labels = Object.values(currentLabels);
        var data = [{
            z: confusionMatrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: 'Viridis'
        }];
        var layout = {
            title: 'Confusion Matrix',
            width: 800,  // Increased width
            height: 800, // Increased height
            xaxis: {
                title: 'Predicted',
                tickangle: -45, // Rotate x-axis labels for better readability
            },
            yaxis: {
                title: 'Actual',
            },
            margin: {
                l: 100, // Increased left margin for y-axis labels
                r: 50,
                b: 150, // Increased bottom margin for x-axis labels
                t: 100, // Increased top margin for title
                pad: 4
            },
            annotations: []
        };
    
        // Add text annotations to each cell
        for (var i = 0; i < confusionMatrix.length; i++) {
            for (var j = 0; j < confusionMatrix[i].length; j++) {
                var currentValue = confusionMatrix[i][j];
                if (currentValue > 0) {
                    layout.annotations.push({
                        x: labels[j],
                        y: labels[i],
                        text: currentValue,
                        font: {
                            color: 'white'
                        },
                        showarrow: false
                    });
                }
            }
        }
    
        Plotly.newPlot('confusionMatrixContainer', data, layout, {displayModeBar: false});
    }

    function updateCharts(metrics) {
        lossChart.data.labels = metrics.epoch;
        lossChart.data.datasets[0].data = metrics.training_loss;
        lossChart.data.datasets[1].data = metrics.validation_loss;
        lossChart.update();
        
        accuracyChart.data.labels = metrics.epoch;
        accuracyChart.data.datasets[0].data = metrics.training_accuracy;
        accuracyChart.data.datasets[1].data = metrics.validation_accuracy;
        accuracyChart.update();
    }

    canvas.addEventListener('mousedown', function(e) {
        var rect = canvas.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;
        var gridX = Math.floor(x / CELL_SIZE);
        var gridY = Math.floor(y / CELL_SIZE);
        if (gridX >= 0 && gridX < GRID_SIZE && gridY >= 0 && gridY < GRID_SIZE) {
            inputGrid[gridY][gridX] = 1 - inputGrid[gridY][gridX];
            drawInputGrid();
            drawExtendedLine();
            predict();
        }
    });

    $('#predictButton').click(predict);
    $('#clearButton').click(clear);
    $('#clearModelButton').click(clearModelData);

    $('#train1Button, #train10Button, #train100Button').click(function() {
        const epochs = parseInt($(this).attr('id').replace('train', '').replace('Button', ''));
        showTrainingProgress();
        trainModel(epochs);
    });


    // Socket.IO client
    var socket = io();

    socket.on('log', function(data) {
        var logWindow = document.getElementById('logWindow');
        var logEntry = document.createElement('p');
        logEntry.textContent = data.message;
        logWindow.appendChild(logEntry);
        logWindow.scrollTop = logWindow.scrollHeight;
    });

    socket.on('training_metrics', function(metrics) {
        updateCharts(metrics);
    });

    socket.on('training_progress', function(data) {
        updateTrainingProgress(data.epoch, data.total_epochs);
    });
    
    socket.on('training_complete', function() {
        hideTrainingProgress();
    });

    $('#helpButton').click(function() {
        $('#helpModal').modal('show');
    });

    $('#graphsButton').click(function() {
        $('#graphsModal').modal('show');
    });

    //ProgressBar
    function showTrainingProgress() {
        $('#trainingProgressContainer').show();
    }
    
    function hideTrainingProgress() {
        $('#trainingProgressContainer').hide();
    }
    
    function updateTrainingProgress(epoch, totalEpochs) {
        const progress = (epoch / totalEpochs) * 100;
        $('#trainingProgressBar').css('width', `${progress}%`).attr('aria-valuenow', progress);
        $('#trainingProgressText').text(`Epoch ${epoch} of ${totalEpochs}`);
    }

    //Drawing
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getMousePos(canvas, e);
        draw(e);  // This allows a single click to draw a point
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function getMousePos(canvas, e) {
        var rect = canvas.getBoundingClientRect();
        return [
            Math.floor((e.clientX - rect.left) / CELL_SIZE),
            Math.floor((e.clientY - rect.top) / CELL_SIZE)
        ];
    }
    
    function draw(e) {
        if (!isDrawing) return;
    
        var [currentX, currentY] = getMousePos(canvas, e);
    
        // Bresenham's line algorithm to ensure a continuous line
        var dx = Math.abs(currentX - lastX);
        var dy = Math.abs(currentY - lastY);
        var sx = (lastX < currentX) ? 1 : -1;
        var sy = (lastY < currentY) ? 1 : -1;
        var err = dx - dy;
    
        while (true) {
            setPixel(lastX, lastY);
    
            if ((lastX === currentX) && (lastY === currentY)) break;
            var e2 = 2 * err;
            if (e2 > -dy) { err -= dy; lastX += sx; }
            if (e2 < dx) { err += dx; lastY += sy; }
        }
    
        lastX = currentX;
        lastY = currentY;
    }
    
    function setPixel(x, y) {
        if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
            inputGrid[y][x] = 0;  // Set to black (0)
            drawInputGrid();
            drawExtendedLine();
        }
    }

    function stopDrawing() {
        if (isDrawing) {
            isDrawing = false;
            predict();  // Predict after drawing is complete
        }
    }

});