let model;
let samples;
let current = 0;

let drawingMode = false;
let isDrawing = false;
let customImage = new Array(784).fill(0);

const networkEl = document.getElementById("network");
const predictionEl = document.getElementById("prediction");
const architectureEl = document.getElementById("architecture");
const infoEl = document.getElementById("info");

const prevBtn = document.getElementById("prev");
const nextBtn = document.getElementById("next");
const drawModeBtn = document.getElementById("drawMode");
const clearBtn = document.getElementById("clearCanvas");

prevBtn.onclick = prev;
nextBtn.onclick = next;
drawModeBtn.onclick = toggleDrawMode;
clearBtn.onclick = clearDrawing;

document.addEventListener("mouseup", () => {
  isDrawing = false;
});

document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight" && !drawingMode) next();
  if (event.key === "ArrowLeft" && !drawingMode) prev();
});

async function main() {
  model = await fetch("../models/784-15-15-15-10.json").then(r => r.json());
  samples = await fetch("../data/test_samples.json").then(r => r.json());

  architectureEl.textContent = model.sizes.join(" → ");

  draw();
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function feedforward(input) {
  let activation = input;
  const activations = [input];

  for (let layer = 0; layer < model.weights.length; layer++) {
    const weights = model.weights[layer];
    const biases = model.biases[layer];

    const next = [];

    for (let neuron = 0; neuron < weights.length; neuron++) {
      let z = biases[neuron][0];

      for (let i = 0; i < activation.length; i++) {
        z += weights[neuron][i] * activation[i];
      }

      next.push(sigmoid(z));
    }

    activation = next;
    activations.push(next);
  }

  return activations;
}

function draw() {
  const sample = drawingMode
    ? { image: customImage, label: "drawn" }
    : samples[current];

  const activations = feedforward(sample.image);
  const output = activations[activations.length - 1];
  const prediction = output.indexOf(Math.max(...output));

  predictionEl.textContent = prediction;

  infoEl.textContent = drawingMode
    ? "Draw your own digit"
    : `True: ${sample.label} · ${current + 1} / ${samples.length}`;

  drawNetwork(activations);
}

function drawNetwork(activations) {
  networkEl.innerHTML = "";

  activations.forEach((layerActivations, layerIndex) => {
    const layer = document.createElement("div");
    layer.className = "layer";

    const title = document.createElement("div");
    title.className = "layer-title";

    if (layerIndex === 0) {
      title.textContent = layerActivations.length === 784
        ? "Input 28×28"
        : `Input (${layerActivations.length})`;
    } else if (layerIndex === activations.length - 1) {
      title.textContent = `Output (${layerActivations.length})`;
    } else {
      title.textContent = `Hidden ${layerIndex} (${layerActivations.length})`;
    }

    layer.appendChild(title);

    if (layerIndex === 0 && layerActivations.length === 784) {
      layer.appendChild(drawInputGrid(layerActivations));
    } else if (layerIndex === activations.length - 1) {
      layer.appendChild(drawOutputLayer(layerActivations));
    } else {
      layer.appendChild(drawNeuronLayer(layerActivations));
    }

    networkEl.appendChild(layer);
  });
}

function drawInputGrid(image) {
  const grid = document.createElement("div");
  grid.className = "input-grid";

  image.forEach((value, index) => {
    const pixel = document.createElement("div");
    pixel.className = "pixel";

    const brightness = Math.floor(value * 255);
    pixel.style.background = `rgb(${brightness}, ${brightness}, ${brightness})`;

    pixel.style.boxShadow = value > 0.35
      ? `0 0 ${value * 10}px rgba(124, 156, 255, ${value})`
      : "none";

    if (drawingMode) {
    pixel.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;

        isDrawing = true;
        pixel.setPointerCapture(event.pointerId);
        paint(index);
    });

    pixel.addEventListener("pointerenter", (event) => {
        if (!isDrawing) return;
        if ((event.buttons & 1) !== 1) return;

        paint(index);
    });
    }

    grid.appendChild(pixel);
  });

  return grid;
}

function paint(index) {
  const x = index % 28;
  const y = Math.floor(index / 28);

  const brush = [
    [0, -1, 0.45],
    [-1, 0, 0.45],
    [0, 0, 1.0],
    [1, 0, 0.45],
    [0, 1, 0.45],
  ];

  for (const [dx, dy, value] of brush) {
    const nx = x + dx;
    const ny = y + dy;

    if (nx < 0 || nx >= 28 || ny < 0 || ny >= 28) continue;

    const newIndex = ny * 28 + nx;
    customImage[newIndex] = Math.min(1, customImage[newIndex] + value);
  }

  draw();
  /*
  customImage[index] = 1;
  draw();
  */
}

function drawNeuronLayer(activations) {
  const neurons = document.createElement("div");
  neurons.className = "neurons";

  const maxVisible = 30;

  const visible = activations.length <= maxVisible
    ? activations
    : [...activations.slice(0, 14), null, ...activations.slice(-14)];

  visible.forEach(value => {
    if (value === null) {
      const dots = document.createElement("div");
      dots.className = "dots";
      dots.textContent = "⋮";
      neurons.appendChild(dots);
      return;
    }

    const neuron = document.createElement("div");
    neuron.className = "neuron";

    neuron.style.opacity = 0.25 + value * 0.75;
    neuron.style.background = `rgba(124, 156, 255, ${0.15 + value * 0.85})`;
    neuron.style.boxShadow = value > 0.5
      ? `0 0 ${value * 18}px rgba(124, 156, 255, ${value})`
      : "none";

    neurons.appendChild(neuron);
  });

  return neurons;
}

function drawOutputLayer(output) {
  const outputLayer = document.createElement("div");
  outputLayer.className = "neurons";

  const maxValue = Math.max(...output);

  output.forEach((value, digit) => {
    const row = document.createElement("div");
    row.className = "output-neuron";

    const digitEl = document.createElement("div");
    digitEl.className = "digit";
    digitEl.textContent = digit;

    const neuron = document.createElement("div");
    neuron.className = "neuron";

    neuron.style.opacity = 0.25 + value * 0.75;
    neuron.style.background = `rgba(124, 156, 255, ${0.15 + value * 0.85})`;
    neuron.style.boxShadow = value > 0.5
      ? `0 0 ${value * 18}px rgba(124, 156, 255, ${value})`
      : "none";

    const track = document.createElement("div");
    track.className = "track";

    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${(value / maxValue) * 100}%`;

    track.appendChild(fill);

    row.appendChild(digitEl);
    row.appendChild(neuron);
    row.appendChild(track);

    outputLayer.appendChild(row);
  });

  return outputLayer;
}

function toggleDrawMode() {
  drawingMode = !drawingMode;

  drawModeBtn.textContent = drawingMode
    ? "Test mode"
    : "Draw mode";

  prevBtn.disabled = drawingMode;
  nextBtn.disabled = drawingMode;

  draw();
}

function clearDrawing() {
  customImage = new Array(784).fill(0);

  if (!drawingMode) {
    drawingMode = true;
    drawModeBtn.textContent = "Test mode";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
  }

  draw();
}

function next() {
  current = (current + 1) % samples.length;
  draw();
}

function prev() {
  current = (current - 1 + samples.length) % samples.length;
  draw();
}

main();