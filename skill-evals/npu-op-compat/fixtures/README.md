# Fixtures

Two tiny ONNX models, exported with `torch.onnx.export`, opset 13. Both are a
few kilobytes and exist only so the eval exercises the real protobuf parser
against real binary files rather than a hand-written op list.

| File | Graph | Why |
|---|---|---|
| `clean_cnn.onnx` | Conv, Relu, Conv, GlobalAveragePool, Flatten, Gemm, Softmax | Every op maps. Proves the tool does not invent a cut. |
| `leakyrelu_backbone.onnx` | 7x (Conv, LeakyRelu) then Conv | YOLO-shaped. The cut lands at index 1 and strands 93% of the graph. |

Ground truth for `clean_cnn.onnx`, from `onnx.load(...).graph.node`:

    ['Conv', 'Relu', 'Conv', 'GlobalAveragePool', 'Flatten', 'Gemm', 'Softmax']

Note the BatchNorm in the source module does not appear: the exporter folds it
into the preceding Conv. That is the behaviour `onnx-to-tflite.json` describes
under `depends.BatchNormalization`.
