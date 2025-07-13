async function runExample() {
    let x = [];

    for (let i = 0; i < 12; i++) {
        x[i] = parseFloat(document.getElementById(`box${i + 1}`).value);
    }

    let tensorX = new ort.Tensor('float32', x, [1, 12]);

    const session = await ort.InferenceSession.create('xgb_FI.onnx');

    // Replace these names based on actual model IO
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    let feeds = {};
    feeds[inputName] = tensorX;

    let result = await session.run(feeds);
    let outputData = result[outputName].data[0];

    let predictions = document.getElementById('predictions');

    predictions.innerHTML = `
        <hr> Got an output tensor with values: <br/>
        <table>
            <tr>
                <td>FI prediction</td>
                <td id="td0">${outputData.toFixed(2)}</td>
            </tr>
        </table>
    `;
}
