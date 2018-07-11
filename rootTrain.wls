#!/usr/bin/env wolframscript
(* ::Package:: *)

netPath = "/home/ubuntu/workdisk/net.wlnet";

trainingDataSet = {FileNames["/home/ubuntu/workdisk/binFiles/*.jpg"], FileNames["/home/ubuntu/workdisk/binFiles/*.bin"]};

checkpointsPath="/home/ubuntu/workdisk/checkpoint";

net=Import[netPath];

imageDataSet = Table[File[trainingDataSet[[1, i]]], {i, 18000}];
maskDataSet = Table[ReadByteArray[trainingDataSet[[2, i]]], {i,18000}];

data = RandomSample@Thread[imageDataSet -> maskDataSet];

partGenerator = Function[Block[{batch, dataSet, batchData},
    If[! ValueQ[partitionedData], partitionedData = Partition[Range[Length[data]], #BatchSize, #BatchSize, 1]];
    Print[Association@Thread[Keys[#] -> Values[#]]];
    batch = Mod[#AbsoluteBatch, Floor[18000/#BatchSize]];
    batchData = data[[partitionedData[[Echo@(batch + 1)]]]];
    <|"Input" -> Keys[batchData],"Target" -> 1 + BinaryDeserialize/@Values[batchData]|>
    ]
];

validationSet=partGenerator[<|"BatchSize"->300,"Round"->0,"AbsoluteBatch"->0|>];

Print[Length[data]];

ClearAll[partitionedData];

trainedNet = NetTrain[net, {partGenerator, "RoundLength" -> 3000}, All, ValidationSet->validationSet, TrainingProgressCheckpointing -> {"Directory", checkpointsPath, "Interval" -> Quantity[30, "Minutes"]}, TargetDevice -> "GPU", TrainingProgressReporting -> "Print"];

Export["model_"<>ToString[AbsoluteTime[]//Round]<>".mx",trainedNet];

Print["complete"];
