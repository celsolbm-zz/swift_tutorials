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
let pk = Python.import("pickle")

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



var inicio = Tensor<Float>(zeros: [1,1024])
var stado = LSTMCell<Float>.State(cell: Tensor<Float>(zeros: [1,1024]), hidden: Tensor<Float>(zeros: [1,1024]))
var R_i = RNNCellInput<Tensor<Float>,LSTMCell<Float>.State>(input: inicio, state: stado)

//create model, Shakes is from Shakespeare :)
struct Shakes: Module{
    typealias Input = [Tensor<Float>]
    typealias Output = Tensor<Float>
    
    
    var embeds = Embedding<Float>(vocabularySize: 65, embeddingSize: 256)
    var rnn = LSTMCell<Float>(inputSize: 256, hiddenSize: 1024)
    var ata = RNN<LSTMCell>(LSTMCell<Float>(inputSize: 256, hiddenSize: 1024))
    var logis = Dense<Float>(inputSize: 1024, outputSize: 65, activation: relu)
    var weights = Tensor<Float>(glorotNormal: ([1024,65]))
    var bias = Tensor<Float>(glorotNormal: ([1,65]))

    var embeddings = Tensor<Float>(glorotNormal: ([65,256]))
    //var R_i = RNNCellInput<LSTMCell.State,LSTMCell.State>
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
        let ata2 = ata.callAsFunction(input)
        let resul = matmul(ata2[0].hidden,weights) + bias
        return resul
//        let test2 = duct2(ata2)
//        let logis_out = logis.callAsFunction(test2)
//        print(logis_out.shape)
//        return logis_out
    }

}

//creates instance of model
var model = Shakes()

//loading the model
//uncomment this part if model was already learned

//let fp = Python.open("save.p","rb")
//var loading = pk.load(fp)
//var input_w = loading[0]
//var update_w = loading[1]
//var forget_w = loading[2]
//var output_w = loading[3]
//var input_b = loading[4]
//var update_b = loading[5]
//var forget_b = loading[6]
//var output_b = loading[7]
//var embs = loading[8]
//var logis_w = loading[9]
//var logis_b = loading[10]
//
//
////
//model.ata.cell.inputWeight = Tensor<Float>(numpy: input_w)!
//model.ata.cell.updateWeight = Tensor<Float>(numpy: update_w)!
//model.ata.cell.forgetWeight = Tensor<Float>(numpy: forget_w)!
//model.ata.cell.outputWeight = Tensor<Float>(numpy: output_w)!
//model.ata.cell.inputBias = Tensor<Float>(numpy: input_b)!
//model.ata.cell.updateBias = Tensor<Float>(numpy: update_b)!
//model.ata.cell.forgetBias = Tensor<Float>(numpy: forget_b)!
//model.ata.cell.outputBias = Tensor<Float>(numpy: output_b)!
//model.embeds.embeddings = Tensor<Float>(numpy: embs)!
//model.weights = Tensor<Float>(numpy: logis_w)!
//model.bias = Tensor<Float>(numpy: logis_b)!










var one_hot = Tensor<Float>(oneHotAtIndices: data_split[0].input_txt, depth: 65)
var um = matmul(one_hot[0].reshaped(to: [1,65]), model.embeds.embeddings)
print(um.shape)
//for debugging, checking if the model is ok
var untrainedLogits = model.callAsFunction([um])
////creates optimizer, chose SGD for simplicity
print(untrainedLogits.shape)
print([sample_label[0]])
let optimizer = SGD(for: model, learningRate: 0.01)
////again, next lines are just for debugging, to see if the model can learning a bit
let untrainedLoss = softmaxCrossEntropy(logits: untrainedLogits, labels: sample_label[0].reshaped(to: [1]))
//print(untrainedLoss)
//var teste = Array(arrayLiteral: sample)
//print(teste.count)
//
var (loss, grads) = model.valueWithGradient { (model: Shakes) -> Tensor<Float> in
    let logits = model([um])
    return softmaxCrossEntropy(logits: logits, labels: sample_label[0].reshaped(to: [1]))
}









print("Current loss: \(loss)")
var plop = (model.ata.cell.inputWeight)
optimizer.update(&model.allDifferentiableVariables, along: grads)
let logitsAfterOneStep = model.callAsFunction([um])
let lossAfterOneStep = softmaxCrossEntropy(logits: logitsAfterOneStep, labels: sample_label[0].reshaped(to: [1]))
print("Next loss: \(lossAfterOneStep)")
print(plop == model.ata.cell.inputWeight)
////results show that the model improved a tiny bit
//
let epochCount = 20
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []
//for the training
func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}



//












var d = model.embeds.embeddings

var media = Tensor<Float>(0)
//training loop
//
for epoch in 1...epochCount {
    var epochLoss: Float = 0
    var epochAccuracy: Float = 0
    var batchCount: Int = 0
    for batch in data_split {
        let one_hot = Tensor<Float>(oneHotAtIndices: batch.input_txt, depth: 65)
        var n = 0
        var um = matmul(one_hot[n].reshaped(to: [1,65]), model.embeds.embeddings)
        while n < one_hot.shape[0]{
            um = matmul(one_hot[n].reshaped(to: [1,65]), model.embeds.embeddings)
            (loss, grads) = model.valueWithGradient { (model: Shakes) -> Tensor<Float> in
                let logits = model([um])
                return softmaxCrossEntropy(logits: logits, labels: batch.output_txt[n].reshaped(to: [1]) )
            }
            //print("Current loss: \(loss), batch number \(batchCount)")

            optimizer.update(&model.allDifferentiableVariables, along: grads)
            let logits = model([um])
            epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.output_txt[n].reshaped(to: [1]))
            epochLoss += loss.scalarized()
            media += loss
            n += 1
        }
        media = media/100
        print("Current loss: \(media), batch number \(batchCount)")
        media -= media
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


var input_w = model.ata.cell.inputWeight.makeNumpyArray()
var update_w = model.ata.cell.updateWeight.makeNumpyArray()
var forget_w = model.ata.cell.forgetWeight.makeNumpyArray()
var output_w = model.ata.cell.outputWeight.makeNumpyArray()
var input_b = model.ata.cell.inputBias.makeNumpyArray()
var update_b = model.ata.cell.updateBias.makeNumpyArray()
var forget_b = model.ata.cell.forgetBias.makeNumpyArray()
var output_b = model.ata.cell.outputBias.makeNumpyArray()
var embs = model.embeds.embeddings.makeNumpyArray()
var logis_w = model.weights.makeNumpyArray()
var logis_b = model.bias.makeNumpyArray()

pk.dump([input_w, update_w, forget_w,output_w,input_b,update_b, forget_b,output_b,embs,logis_w,logis_b], Python.open( "save.p", "wb" ))
