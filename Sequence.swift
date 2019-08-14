//
//  main.swift
//  teste
//
//  Created by Celso Moraes on 07/08/19.
//  Copyright © 2019 Celso Moraes. All rights reserved.
//


import Foundation
import TensorFlow
import Python


// Import Python modules
let matplotlib = Python.import("matplotlib")
let np = Python.import("numpy")

//function to download the data from the tensorflow stoarage
func download(from sourceString: String, to destinationString: String) {
    let source = URL(string: sourceString)!
    let destination = URL(fileURLWithPath: destinationString)
    let data = try! Data.init(contentsOf: source)
    try! data.write(to: destination)
}

let traintxt = "seq.txt"
download(from:"https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt", to: traintxt)

//get the data from the repository and read as a string
let total_data = try String(contentsOfFile: traintxt, encoding: String.Encoding.utf8)

//set the conversion from char to int
let vocab = Set(total_data).sorted()
let char2idx = vocab.enumerated()
var conv: [Character : Int] = [:]
let idx2char = Array(vocab)
for (i,name) in char2idx{
    conv[name] = i
}
var txt: [Float] = []
for c in total_data {
    txt.append(Float(conv[c]!))
}

//Organize the elements for the creation of the dataset
let seq_length = 100
let examples_per_epoch = total_data.count/seq_length

let data_2 = np.array(txt, dtype: Int32.numpyScalarTypes.first!)
let data_set2 = Tensor<Int32>(numpy: data_2)!

//just created this for debugging, to see the first elements of the dataset
extension Sequence where Element == Tensor<Int32> {
    var first: Tensor<Int32>? {
        return first(where: {_ in true})
    }
}

//creating the dataset
let dataset_train: Dataset<Tensor<Int32>> = Dataset(elements: data_set2).batched(seq_length+1)

//function for splitting the dataset into a learning and target system
func split_input_target2(_ chunk: Tensor<Int32>)->(input_txt: Tensor<Int32>, output_txt: Tensor<Int32>){
    let np = Python.import("numpy")
    let temp = chunk.makeNumpyArray()
    let input_txt = Tensor<Int32>(numpy: np.delete(temp,-1))!
    let output_txt = Tensor<Int32>(numpy:np.delete(temp,0))!
    return (input_txt, output_txt)
}

//create the splited dataset to be feeded for the training
var data_split = dataset_train.map(split_input_target2)

let vocab_size = vocab.count
let embedding_dim = 256
let rnn_hid = 1024

//this was just for debugging, created an example of a embedding to test the model
var embeds = Embedding<Float>(vocabularySize: vocab_size, embeddingSize: embedding_dim)
//sample data for showing the output of the model
var sample = embeds.callAsFunction(data_split[0].input_txt)
var sample_label = data_split[0].output_txt



//create model, Shakes is from Shakespeare :)
struct Shakes: Layer{
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    var embeds = Embedding<Float>(vocabularySize: 65, embeddingSize: 256)
    var rnn = LSTMCell<Float>(inputSize: 256, hiddenSize: 1024)
    var ata = RNN<LSTMCell>(LSTMCell<Float>(inputSize: 256, hiddenSize: 1024))
    var logis = Dense<Float>(inputSize: 1024, outputSize: 65, activation: relu)

    
    //had to use this function to organize the output of the rn so that I could
    //feed it to the logits layer
    func duct2(_ inp: Array<LSTMCell<Float>.State>)->Tensor<Float>{
        let size = inp.count
        var saida = [Tensor<Float>]()
        var n = 0
        while n < size{
            saida.append(inp[n].hidden)
            n+=1
        }
        return Tensor<Float>(concatenating: saida, alongAxis: 0)
    }


    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        let size = input.shape
        var saida = [Tensor<Float>]()
        var n = 0
        let fd = TensorShape([1,256])
        while n < size[0]{
            saida.append(input[n].reshaped(to: fd))
            n+=1
        }
        let ata2 = ata.callAsFunction(saida)
        let test2 = duct2(ata2)
        let logis_out = logis.callAsFunction(test2)

        return logis_out
    }

}

//creates instance of model
var model = Shakes()

//for debugging, checking if the model is ok
var untrainedLogits = model.callAsFunction(sample)
//creates optimizer, chose SGD for simplicity
let optimizer = SGD(for: model, learningRate: 0.01)
//again, next lines are just for debugging, to see if the model can learning a bit
let untrainedLoss = softmaxCrossEntropy(logits: untrainedLogits, labels: sample_label)
print(untrainedLoss)
let (loss, grads) = model.valueWithGradient { model -> Tensor<Float> in
    let logits = model(sample)
    return softmaxCrossEntropy(logits: logits, labels: sample_label)
}

print("Current loss: \(loss)")
optimizer.update(&model.allDifferentiableVariables, along: grads)
let logitsAfterOneStep = model(sample)
let lossAfterOneStep = softmaxCrossEntropy(logits: logitsAfterOneStep, labels: sample_label)
print("Next loss: \(lossAfterOneStep)")
//results show that the model improved a tiny bit

let epochCount = 20
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []
//for the training
func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}


//training loop

for epoch in 1...epochCount {
    var epochLoss: Float = 0
    var epochAccuracy: Float = 0
    var batchCount: Int = 0
    for batch in data_split {
        let (loss, grad) = model.valueWithGradient { (model: Shakes) -> Tensor<Float> in
            let entre = embeds.callAsFunction(batch.input_txt)
            let logits = model(entre)
            return softmaxCrossEntropy(logits: logits, labels: batch.output_txt)
        }
        print("Current loss: \(loss), batch number \(batchCount)")
        let entre = embeds.callAsFunction(batch.input_txt)
        optimizer.update(&model.allDifferentiableVariables, along: grad)
        let logits = model(entre)
        epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.output_txt)
        epochLoss += loss.scalarized()
        batchCount += 1
    }
    epochAccuracy /= Float(batchCount)
    epochLoss /= Float(batchCount)
    trainAccuracyResults.append(epochAccuracy)
    trainLossResults.append(epochLoss)
    if epoch % 2 == 0 {
        print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
    }
}