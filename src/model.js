'use strict'

const tf = require('@tensorflow/tfjs');
const historicalPrices = require('./data/forex-aud-60min.data.json').results;
const atrCalculator = require('../node_modules/quant-sdk/src/average-true-range.calculator');

const MAX_EPOCH = 1500;
const LEARNING_RATE = 0.1;

function transformData(historicalPrices) {
  let priceList = [...historicalPrices];
  let dataList = [];

  // Map ATR and associated volume to new array 
  for (let i = 0; i < historicalPrices.length - 14; i++) {
    let averageTrueRange =
      atrCalculator.calculateAverageTrueRangePrice(
        priceList,
        4
      );

    dataList.push({
      averageTrueRange: averageTrueRange,
      volume: historicalPrices[i].v
    });

    priceList.pop();
  }

  // Filter any outlier data (there is bad data in this set)
  return dataList.filter(x => x.averageTrueRange < 1);
}

function run(xList, yList) {
  const xs = tf.tensor1d(xList);
  const ys = tf.tensor1d(yList);

  const W = tf.scalar(0.1).variable();
  const b = tf.scalar(0.1).variable();

  // Define a linear regression model: f(x) = Wx + b
  const f = x => W.mul(x).add(b);
  
  // Define loss function
  const loss = (prediction, label) => prediction.sub(label).square().mean();
  const optimizer = tf.train.adam(LEARNING_RATE);

  // Train the model.
  for (let i = 0; i < MAX_EPOCH; i++) {
    optimizer.minimize(() => loss(f(xs), ys));
  }

  // Make predictions and compare with actual result.
  const predictionList = f(xs).dataSync();
  predictionList
    .forEach((p, i) => {
      console.log(`x: ${i}, prediction: ${p}`);
      console.log(`x: ${i}, actual: ${yList[i]}`);
    });
}

let dataList = transformData(historicalPrices);

let xList = dataList.map(m => parseFloat(m.volume));
let yList = dataList.map(m => parseFloat(m.averageTrueRange));

run(
  xList,
  yList
);
