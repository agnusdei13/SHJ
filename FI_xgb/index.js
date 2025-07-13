async function runExample() {
    let x = [];
    for (let i = 0; i < 12; i++) {
        x[i] = parseFloat(document.getElementById(`box${i+1}`).value);
    }

    const tensorX = new ort.Tensor('float32', x, [1, 12]);
    const session = await ort.InferenceSession.create('xgb_FI.onnx');
    
    const inputName = session.inputNames[0];
    const feeds = {};
    feeds[inputName] = tensorX;

    const result = await session.run(feeds); 
    
    const outputLabelTensor = result['output_label'];

    if (outputLabelTensor) {
        console.log("Output 'output_label' type:", outputLabelTensor.type);
        
        let outputData = outputLabelTensor.data[0];

        outputData = parseInt(outputData, 10); 
        
        const predictions = document.getElementById('predictions');
        predictions.innerHTML = `
            <hr> Got an output tensor with values: <br/>
            <table>
                <tr>
                    <td>FI prediction (Label)</td>
                    <td id="td0">${outputData}</td>
                </tr>
            </table>
        `;
    } else {
        console.error("The 'output_label' output was not found in the model results.");
        const predictions = document.getElementById('predictions');
        predictions.innerHTML = `<hr> Error: Could not find prediction label output.`;
    }
}
