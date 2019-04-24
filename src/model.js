'use strict'

// Imports.
const tf = require('@tensorflow/tfjs');
const atrCalculator = require('../node_modules/quant-sdk/src/average-true-range.calculator');

// Training and testing data sets.
const trainingHistoricalPrices = require('./data/forex-aud-60min-training.data.json').results;
const testingHistoricalPrices = require('./data/forex-aud-60min-testing.data.json').results;

const MAX_EPOCH = 1500;
const LEARNING_RATE = 0.1;

function transformTrainingData(trainingHistoricalPrices) {
  let priceList = [...trainingHistoricalPrices];
  let dataList = [];

  // Map ATR and associated volume to new array. 
  for (let i = 0; i < trainingHistoricalPrices.length - 14; i++) {
    let averageTrueRange =
      atrCalculator.calculateAverageTrueRangePrice(
        priceList,
        4
      );

    dataList.push({
      averageTrueRange: averageTrueRange,
      volume: trainingHistoricalPrices[i].v
    });

    priceList.pop();
  }

  // Filter any outlier data (there is bad data in this set).
  return dataList.filter(x => x.averageTrueRange < 1);
}

function run(xList, yList) {
  const xs = tf.tensor1d(xList);
  const ys = tf.tensor1d(yList);

  const W = tf.scalar(0.1).variable();
  const b = tf.scalar(0.1).variable();

  // Define a linear regression model: f(x) = Wx + b.
  const f = x => W.mul(x).add(b);

  // Define loss function.
  const loss = (prediction, label) => prediction.sub(label).square().mean();

  // Define optimizer strategy.
  const optimizer = tf.train.adam(LEARNING_RATE);

  // Train the model.
  for (let i = 0; i < MAX_EPOCH; i++) {
    optimizer.minimize(() => loss(f(xs), ys));
  }

  // Make predictions and print result.
  const predictionList = f(testingHistoricalPrices.map(m => m.v)).dataSync();
  predictionList
    .forEach((p, i) => {
      console.log(`x: ${i}, prediction: ${p}`);
    });
}

let dataList = transformTrainingData(trainingHistoricalPrices);

let xList = dataList.map(m => parseFloat(m.volume));
let yList = dataList.map(m => parseFloat(m.averageTrueRange));

run(
  xList,
  yList
);
