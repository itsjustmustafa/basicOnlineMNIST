let model;
let scale = 10;
let graphic;
let drawButton, eraseButton;
let drawingRadius;
let modelRunning = false;

function setDrawing() {
    graphic.fill(255);
    drawButton.style = "background: rgb(50,50,50);"
    eraseButton.style = "background: rgb(150,150,150);"
    drawingRadius = 2;
}

function setErasing() {
    graphic.fill(0);
    eraseButton.style = "background: rgb(50,50,50);"
    drawButton.style = "background: rgb(150,150,150);"
    drawingRadius = 4;
}

function setup() {
    size = 28 * scale;
    createCanvas(size, size);
    graphic = createGraphics(28, 28);
    graphic.background(0);
    // graphic.fill(255);
    graphic.noStroke();


    drawButton = document.createElement('button');
    eraseButton = document.createElement('button');
    drawButton.innerText = "✏️";
    eraseButton.innerText = "🧽";
    mainElement = document.getElementsByTagName("main")[0];
    mainElement.appendChild(drawButton);
    mainElement.appendChild(eraseButton);
    drawButton.addEventListener('click', setDrawing);
    eraseButton.addEventListener('click', setErasing);
    setDrawing();
}


function draw() {
    image(graphic, 0, 0, width, height);
}

const compute = async (tensor) => {
    try {
        const prediction = await model.predict(tensor);
        return prediction.data();
    } catch (err) {
        throw err;
    }
};


async function mouseDragged(event) {
    mouseDist = max(abs(movedX), abs(movedY));
    for (let i = 0; i < mouseDist; i += scale * 0.3) {
        mouseLerpX = map(i, 0, mouseDist, mouseX, pmouseX);
        mouseLerpY = map(i, 0, mouseDist, mouseY, pmouseY);
        // graphic.fill(255);
        graphic.ellipse(mouseLerpX / scale, mouseLerpY / scale, drawingRadius);
    }
    if (mouseX < width && mouseX > 0 && mouseY < height && mouseY > 0) {
        await predict();
    }
}

function mouseReleased() {
    if (mouseX < width && mouseX > 0 && mouseY < height && mouseY > 0) {
        predict();
    }
}

function maxIndex(arr) {
    index = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > arr[index]) {
            index = i;
        }
    }
    return index;
}


async function predict() {
    if (modelRunning) {
        return null;
    }
    modelRunning = true;
    const imageData = [];
    for (let j = 0; j < height; j += scale) {
        for (let i = 0; i < width; i += scale) {
            imageData.push(get(i, j)[0]);
        }
    }


    const input = tf.tensor1d(imageData).reshape([1, 28, 28]).toFloat().div(255);

    predictionDiv = document.getElementById("prediction");
    predictionDiv.innerHTML = "";
    predictionDiv.style = "width: 4em;"

    console.log("pre");
    try {
        const prediction = await compute(input);
        console.log("got the prediction");
        console.log(prediction);
        let answer = maxIndex(prediction);
        for (let p = 0; p < prediction.length; p++) {
            let prob = prediction[p];
            let brightness = floor(constrain(map(Math.log(prob), -3, 0, 255, 0), 0, 255));
            textBrightness = 0;
            if (brightness < 100) {
                textBrightness = 255;
            }
            let bgColour = color(brightness);
            let textColour = color(textBrightness);
            styling = "background-color:" + bgColour.toString() + ";";
            styling += "color:" + textColour.toString() + ";";
            const paraDigit = document.createElement('p');
            paraDigit.innerText = p.toString()
            if (answer == p) {
                paraDigit.innerText += " <---";
            }
            paraDigit.style = styling;
            predictionDiv.appendChild(paraDigit);
            console.log("Yo")
        }
    } catch (error) {
        console.error(error);
    }
    console.log("post");
    modelRunning = false;
}

async function loadModel() {
    model = await tf.loadLayersModel('models/mnist_tfjs_model/model.json');
    model.summary();
}
loadModel();