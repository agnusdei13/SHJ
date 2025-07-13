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
    const probabilityOutput = result['output_probability'];
    
    if (probabilityOutput) {
        console.log("Probability output type:", probabilityOutput.type);
        let outputData = probabilityOutput.data[0]; 
        outputData = parseFloat(outputData).toFixed(2);
        
        const predictions = document.getElementById('predictions');
        predictions.innerHTML = '
            <hr> Got an output tensor with values: <br/>
            <table>
                <tr>
                    <td>FI prediction</td>
                    <td id="td0">${outputData}</td>
                </tr>
            </table>
        ';
}
