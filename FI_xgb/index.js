async function runExample() {

    var x = [];
    
    x[0] = parseFloat(document.getElementById('box1').value);
    x[1] = parseFloat(document.getElementById('box2').value);
    x[2] = parseFloat(document.getElementById('box3').value);
    x[3] = parseFloat(document.getElementById('box4').value);
    x[4] = parseFloat(document.getElementById('box5').value);
    x[5] = parseFloat(document.getElementById('box6').value);
    x[6] = parseFloat(document.getElementById('box7').value);

     
    let tensorX = new ort.Tensor('float32', x, [1, 7]);
    let feeds = { input: tensorX }; // 'input' must match your ONNX model's input name

     
    let session = await ort.InferenceSession.create('rfc_FI.onnx');

  
    let result = await session.run(feeds);

     
    let outputData = result.output_label.data;

     
    let predictionOutputDiv = document.getElementById('prediction-output');

     
    predictionOutputDiv.innerHTML = `Prediction: ${parseFloat(outputData).toFixed(2)}`;
}
