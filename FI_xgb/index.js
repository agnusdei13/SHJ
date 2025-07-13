async function runExample() {

    const tensorX = new ort.Tensor('float32', x, [1, 12]);
    const session = await ort.InferenceSession.create('xgb_FI.onnx');
    
    const inputName = session.inputNames[0]; // This is fine
    const feeds = {};
    feeds[inputName] = tensorX;

    const result = await session.run(feeds);
    
    const probabilityOutput = result['output_probability'];
    
    if (probabilityOutput) {
        console.log("Probability output type:", probabilityOutput.type);
        
        let outputData = probabilityOutput.data[0]; 
        
        outputData = parseFloat(outputData).toFixed(2);
        
        const predictions = document.getElementById('predictions');
        predictions.innerHTML = `
            <hr> Got an output tensor with values: <br/>
            <table>
                <tr>
                    <td>FI prediction</td>
                    <td id="td0">${outputData}</td>
                </tr>
            </table>
        `;
    } else {
        // Handle the case where the expected output is not found
        console.error("The 'output_probability' output was not found in the model results.");
        const predictions = document.getElementById('predictions');
        predictions.innerHTML = `<hr> Error: Could not find prediction output.`;
    }
}
