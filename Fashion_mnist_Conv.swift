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


public struct fashion_mnist {
    public let trainingImages: Tensor<Float>
    public let trainingLabels: Tensor<Int32>
    public let testImages: Tensor<Float>
    public let testLabels: Tensor<Int32>

    public let trainingSize: Int
    public let testSize: Int

    public let batchSize: Int

    public init(batchSize: Int, flattening: Bool = false, normalizing: Bool = false) {
        self.batchSize = batchSize

        let (trainingImages, trainingLabels) = readMNIST(
            imagesFile: "train-images-idx3-ubyte",
            labelsFile: "train-labels-idx1-ubyte",
            flattening: flattening,
            normalizing: normalizing)
        self.trainingImages = trainingImages
        self.trainingLabels = trainingLabels
        self.trainingSize = Int(trainingLabels.shape[0])

        let (testImages, testLabels) = readMNIST(
            imagesFile: "t10k-images-idx3-ubyte",
            labelsFile: "t10k-labels-idx1-ubyte",
            flattening: flattening,
            normalizing: normalizing)
        self.testImages = testImages
        self.testLabels = testLabels
        self.testSize = Int(testLabels.shape[0])
    }
}

extension Tensor {
    public func minibatch(at index: Int, batchSize: Int) -> Tensor {
        let start = index * batchSize
        return self[start..<start+batchSize]
    }
}

func readFile(_ path: String, possibleDirectories: [String]) -> [UInt8] {
//    for folder in possibleDirectories {
//        let parent = URL(fileURLWithPath: folder)
//        let filePath = parent.appendingPathComponent(path)
//        guard FileManager.default.fileExists(atPath: filePath.path) else {
//            continue
//        }
//        let data = try! Data(contentsOf: filePath, options: [])
//        return [UInt8](data)
//    }
    let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let filePath = dir.appendingPathComponent(path)
    let data = try! Data(contentsOf: filePath, options: [])
    return [UInt8](data)
//    print("File not found: \(path)")
//    exit(-1)
}

func readMNIST(imagesFile: String, labelsFile: String, flattening: Bool, normalizing: Bool) -> (
    images: Tensor<Float>,
    labels: Tensor<Int32>
) {
    print("Reading data from files: \(imagesFile), \(labelsFile).")
    let images = readFile(imagesFile, possibleDirectories: [".", "./Datasets/MNIST",""]).dropFirst(16)
        .map(Float.init)
    let labels = readFile(labelsFile, possibleDirectories: [".", "./Datasets/MNIST"]).dropFirst(8)
        .map(Int32.init)
    let rowCount = labels.count
    let imageHeight = 28
    let imageWidth = 28

    print("Constructing data tensors.")

    if flattening {
        var flattenedImages = Tensor(shape: [rowCount, imageHeight * imageWidth], scalars: images)
            / 255.0
        if normalizing {
            flattenedImages = flattenedImages * 2.0 - 1.0
        }
        return (images: flattenedImages, labels: Tensor(labels))
    } else {
        return (
            images: Tensor(shape: [rowCount, 1, imageHeight, imageWidth], scalars: images)
                .transposed(withPermutations: [0, 2, 3, 1]) / 255,  // NHWC
            labels: Tensor(labels)
        )
    }
}
//let file = "t10k-images-idx3-ubyte"
//let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
//let filePath = dir.appendingPathComponent(file)
//let data = try! Data(contentsOf: filePath, options: [])


let data = fashion_mnist(batchSize: 100, flattening: false, normalizing: true)
print(data.trainingImages.minibatch(at: 0, batchSize: 100).shape)
let x = Tensor<Float>(randomNormal: [10,16,16,3])
let y = Tensor<Int32>(rangeFrom: 0, to: 10, stride: 1)
var conv = Conv2D<Float>(filterShape: (5,5,3,6))

struct Fashion_conv: Module {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    
    var conv = Conv2D<Float>(filterShape: (4,4,1,1), strides: (1,1), activation: relu)
    var maxpool = MaxPool2D<Float>(poolSize: (2,2), strides: (1,1))
    var flat = Flatten<Float>()
    var FC = Dense<Float>(inputSize: 576, outputSize: 576)
    var Skip = Dense<Float>(inputSize: 576, outputSize: 10)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let tmp = input.sequenced(through: conv, maxpool, flat)
        let tmp2 = FC(tmp)
        return Skip(tmp+tmp2)
    }
}

var model = Fashion_conv()
var untrained = model(data.trainingImages.minibatch(at: 0, batchSize: 100))
print(untrained.shape)

let optimizer = SGD(for: model, learningRate: 0.01)
let untrainedLoss = softmaxCrossEntropy(logits: untrained, labels: data.trainingLabels.minibatch(at: 0, batchSize: 100))

var (loss, grads) = model.valueWithGradient { (model: Fashion_conv) -> Tensor<Float> in
    let logits = model(data.trainingImages.minibatch(at: 0, batchSize: 100))
    return softmaxCrossEntropy(logits: logits, labels: data.trainingLabels.minibatch(at: 0, batchSize: 100))
}

print("Current loss: \(loss)")
optimizer.update(&model.self, along: grads)


let logits_1 = model(data.trainingImages.minibatch(at: 0, batchSize: 100))
let trainedLoss = softmaxCrossEntropy(logits: logits_1, labels: data.trainingLabels.minibatch(at: 0, batchSize: 100))

print("Next loss: \(trainedLoss)")


let epochCount = 20
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []
//for the training
func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}
var batchSize = 100

for epoch in 1..<epochCount {
    for i in 0..<data.trainingSize/batchSize {
        let (loss, grads) = model.valueWithGradient { (model: Fashion_conv) -> Tensor<Float> in
            let logits = model(data.trainingImages.minibatch(at: i, batchSize: 100))
            return softmaxCrossEntropy(logits: logits, labels: data.trainingLabels.minibatch(at: i, batchSize: 100))
        }
        optimizer.update(&model.self, along: grads)
        print("minibatch loss: \(loss)")
        
    }
    print("currently at epoch \(epoch)")
}
