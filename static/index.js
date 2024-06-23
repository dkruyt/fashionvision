$(document).ready(function() {
    const GRID_SIZE = 28;
    const CELL_SIZE = 10;
    let inputGrid = Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(1));
    let canvas = document.getElementById('inputCanvas');
    let ctx = canvas.getContext('2d');
    let extendedLineCanvas = document.getElementById('extendedLineCanvas');
    let extendedLineCtx = extendedLineCanvas.getContext('2d');
    
    let trainingLossChart;
    let validationLossChart;
    let trainingAccuracyChart;
    let validationAccuracyChart;

    function updateDistributions() {
        $.ajax({
            url: '/distributions',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid }),
            success: function(response) {
                $('#weightHistogram').attr('src', 'data:image/png;base64,' + response.weightHist);
                $('#biasHistogram').attr('src', 'data:image/png;base64,' + response.biasHist);
                $('#activationHistogram').attr('src', 'data:image/png;base64,' + response.activationHist);
                updateConfidenceChart(response.confidence);
            }
        });
    }
    
    function drawNetworkVisualization(data) {
        var canvas = document.getElementById('networkVisualizationCanvas');
        var ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Set canvas size
        canvas.width = 1000;
        canvas.height = 1024;
        
        // Define neuron positions
        var inputNeurons = data.inputActivations.length;
        var hiddenNeurons = data.hiddenActivations[0].length;
        var outputNeurons = data.outputActivations[0].length;
        
        var inputLayer = Array.from({length: inputNeurons}, (_, i) => ({x: 150, y: 20 + i * (960 / inputNeurons)}));
        var hiddenLayer = Array.from({length: hiddenNeurons}, (_, i) => ({x: 500, y: 330 + i * (360 / hiddenNeurons)}));
        var outputLayer = Array.from({length: outputNeurons}, (_, i) => ({x: 850, y: 320 + i * (360 / outputNeurons)}));
        
        // Draw connections
        for (let i = 0; i < inputNeurons; i++) {
            for (let j = 0; j < hiddenNeurons; j++) {
                drawConnection(ctx, inputLayer[i], hiddenLayer[j], data.hiddenWeights[j][i]);
            }
        }
        
        for (let i = 0; i < hiddenNeurons; i++) {
            for (let j = 0; j < outputNeurons; j++) {
                drawConnection(ctx, hiddenLayer[i], outputLayer[j], data.outputWeights[j][i]);
            }
        }
        
        // Draw neurons
        drawNeurons(ctx, inputLayer, data.inputActivations);
        drawNeurons(ctx, hiddenLayer, data.hiddenActivations[0]);
        drawNeurons(ctx, outputLayer, data.outputActivations[0]);
    }
    
    function drawConnection(ctx, start, end, weight) {
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.strokeStyle = getWeightColor(weight);
        ctx.lineWidth = Math.abs(weight) * 2;
        ctx.stroke();
    }
    
    function drawNeurons(ctx, neurons, activations) {
        neurons.forEach((neuron, i) => {
            ctx.beginPath();
            ctx.arc(neuron.x, neuron.y, 7, 0, 2 * Math.PI);
            ctx.fillStyle = getActivationColor(activations[i]);
            ctx.fill();
        });
    }
    
    function getWeightColor(weight) {
        var r = weight > 0 ? 255 : 0;
        var b = weight < 0 ? 255 : 0;
        var g = 0;
        var a = Math.min(Math.abs(weight), 1);
        return `rgba(${r},${g},${b},${a})`;
    }
    
    function getActivationColor(activation) {
        var r = Math.round(activation * 255);
        var g = 0;        
        var b = Math.round(activation * 255);
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
    
    $('#distributionsButton').click(function() {
        updateDistributions();
        $('#distributionsModal').modal('show');
    });
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
                var color = 'rgb(' + Math.round(value * 255) + ', ' + Math.round(value * 255) + ', ' + Math.round(value * 255) + ')';
                ctx.fillStyle = color;
                ctx.fillRect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
    }

    function drawExtendedLine() {
        extendedLineCtx.clearRect(0, 0, extendedLineCanvas.width, extendedLineCanvas.height);
        for (var i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            var row = Math.floor(i / GRID_SIZE);
            var col = i % GRID_SIZE;
            var value = inputGrid[row][col];
            var color = 'rgb(' + Math.round(value * 255) + ', ' + Math.round(value * 255) + ', ' + Math.round(value * 255) + ')';
            extendedLineCtx.fillStyle = color;
            extendedLineCtx.fillRect(i * 2, 3, 1, 4);
        }
    }

    function updateActivations(hiddenActivations, outputActivations) {
        var hiddenHtml = '';
        for (var i = 0; i < hiddenActivations[0].length; i++) {
            var activation = hiddenActivations[0][i];
            var color = getColorMap(activation);
            hiddenHtml += '<div class="neuron hidden-neuron" style="background-color: ' + color + ';" data-index="' + i + '">' + activation.toFixed(3) + '</div>';
        }
        $('#hiddenLayer').html(hiddenHtml);

        var outputHtml = '';
        for (var i = 0; i < outputActivations[0].length; i++) {
            var activation = outputActivations[0][i];
            var color = getColorMap(activation);
            outputHtml += '<div class="neuron-container" style="text-align: center;">';
            outputHtml += '<div class="neuron output-neuron" style="background-color: ' + color + ';" data-index="' + i + '">' + activation.toFixed(3) + '</div>';
            const labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
            outputHtml += '<div class="neuron-number">' + labels[i] + '</div>'; // Use Fashion-MNIST class names
            outputHtml += '</div>';
        }
        $('#outputLayer').html(outputHtml);

        addNeuronClickHandlers();
    }

    function getColorMap(value) {
        value = Math.max(0, Math.min(1, value));
        var hue = (1 - value) * 240;
        return 'hsl(' + hue + ', 100%, 50%)';
    }

    function predict() {
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid }),
            success: function(response) {
                const labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
                var predictedClass = response.predictedClass;
                var hiddenActivations = response.hiddenActivations;
                var outputActivations = response.outputActivations;
                $('#result').text('Result: ' + labels[predictedClass]);
                updateActivations(hiddenActivations, outputActivations);
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
    }

    function prepareData() {
        $.ajax({
            url: '/training_data',
            method: 'GET',
            success: function(response) {
                showTrainingData(response.trainingData);
            }
        });
    }

    function trainModel(epochs) {
        $.ajax({
            url: '/train',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ epochs: epochs }),
            success: function(response) {
                console.log(response.message);
                predict();
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
        var trainingHtml = '';
        for (var i = 0; i < trainingData.length; i++) {
            var img = trainingData[i];
            var imgHtml = '<img src="data:image/png;base64,' + img + '" width="28" height="28" data-index="' + i + '">';
            trainingHtml += imgHtml;
        }
        $('#trainingData').html(trainingHtml);
    }

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

    function trainSingleExample(classLabel) {
        $.ajax({
            url: '/train_single',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid, classLabel: classLabel }),
            success: function(response) {
                console.log('Training on class ' + classLabel + ' completed');
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
        var ctx1 = document.getElementById('trainingLossChart').getContext('2d');
        var ctx2 = document.getElementById('validationLossChart').getContext('2d');
        var ctx3 = document.getElementById('trainingAccuracyChart').getContext('2d');
        var ctx4 = document.getElementById('validationAccuracyChart').getContext('2d');
        
        trainingLossChart = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
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
        
        validationLossChart = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
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
        
        trainingAccuracyChart = new Chart(ctx3, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
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
        
        validationAccuracyChart = new Chart(ctx4, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
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
        var labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
        var data = [{
            z: confusionMatrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: 'Viridis'
        }];
        var layout = {
            title: 'Confusion Matrix',
            xaxis: {title: 'Predicted'},
            yaxis: {title: 'Actual'}
        };
        Plotly.newPlot('confusionMatrixContainer', data, layout);
    }

    function updateCharts(metrics) {
        trainingLossChart.data.labels = metrics.epoch;
        trainingLossChart.data.datasets[0].data = metrics.training_loss;
        trainingLossChart.update();
        
        validationLossChart.data.labels = metrics.epoch;
        validationLossChart.data.datasets[0].data = metrics.validation_loss;
        validationLossChart.update();
        
        trainingAccuracyChart.data.labels = metrics.epoch;
        trainingAccuracyChart.data.datasets[0].data = metrics.training_accuracy;
        trainingAccuracyChart.update();
        
        validationAccuracyChart.data.labels = metrics.epoch;
        validationAccuracyChart.data.datasets[0].data = metrics.validation_accuracy;
        validationAccuracyChart.update();
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
    $('#prepareButton').click(prepareData);
    $('#train1Button').click(function() { trainModel(1); });
    $('#train10Button').click(function() { trainModel(10); });
    $('#train100Button').click(function() { trainModel(100); });
    $('#clearModelButton').click(clearModelData);

    $(document).on('click', '#trainingData img', function() {
        var index = $(this).data('index');
        loadTrainingImage(index);
    });

    // Socket.IO client
    var socket = io();

    socket.on('log', function(data) {
        var logWindow = document.getElementById('logWindow');
        logWindow.innerHTML += '<p>' + data.message + '</p>';
        logWindow.scrollTop = logWindow.scrollHeight;
    });

    socket.on('training_metrics', function(metrics) {
        updateCharts(metrics);
    });

    $('#helpButton').click(function() {
        $('#helpModal').modal('show');
    });

    $('#graphsButton').click(function() {
        $('#graphsModal').modal('show');
    });

    drawInputGrid();
    drawExtendedLine();
    initializeCharts();
});