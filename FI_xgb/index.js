async function runExample() {

  const x = [];

  for (let i = 1; i <= 7; i++) {
    const boxId = `box${i}`;
    const element = document.getElementById(boxId);
    if (element) {
      x.push(parseFloat(element.value));
    } else {
      console.error(`Element with ID '${boxId}' not found.`);

      return;
    }
  }

  const tensorX = new ort.Tensor('float32', x, [1, 7]);

  const feeds = { input: tensorX };

  try {

    const session = await ort.InferenceSession.create('rfc_FI.onnx');

   
    const result = await session.run(feeds);


    const outputData = result.output_label.data;

    const prediction = parseFloat(outputData[0]).toFixed(2);


    const predictionsElement = document.getElementById('predictions');
    if (predictionsElement) {
      predictionsElement.innerHTML = `
        <hr> Got an output tensor with values: <br/>
        <table>
          <tr>
            <td>FI prediction</td>
            <td id="td0">${prediction}</td>
          </tr>
        </table>`;
    } else {
      console.error("Element with ID 'predictions' not found.");
    }

  } catch (e) {
    console.error(`Error during ONNX inference: ${e}`);
    const predictionsElement = document.getElementById('predictions');
    if (predictionsElement) {
      predictionsElement.innerHTML = `<hr><p style="color: red;">Error: Could not run prediction.</p>`;
    }
  }
}
