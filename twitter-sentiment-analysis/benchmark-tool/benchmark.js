import { pipeline } from '@xenova/transformers';
import { samples } from './data.js';

// Parse the samples to get all the inputs
let inputs = samples.map(sample => sample.input);

function calculateMetrics(predictions) {
    // define the confusion matrix elements
    // tp: true positive, fp: false positive, tn: true negative, fn: false negative
    let tp = 0, fp = 0, tn = 0, fn = 0;

    // Update the elements accordingly
    predictions.map((prediction, index) => {
        const predictionLabel = prediction === 'POSITIVE' ? 1 : 0;
        const trueLabel = samples[index].label === 'POSITIVE' ? 1 : 0;

        if (predictionLabel === 1 && trueLabel === 1) tp++;
        if (predictionLabel === 1 && trueLabel === 0) fp++;
        if (predictionLabel === 0 && trueLabel === 0) tn++;
        if (predictionLabel === 0 && trueLabel === 1) fn++;
    });

    // Calculate the metrics
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp);
    const recall = tp / (tp + fn);
    const f = 2 * (precision * recall) / (precision + recall);

    return { accuracy, precision, recall, f1 };
}

async function benchmarkModel(model) {
    // Instantiate a pipeline for sentiment-analysis using the model
    let classifier = await pipeline('sentiment-analysis', model);
    //console.log("PASSED")

    // Run the inputs through the model, time the inference time
    const startTime = Date.now();
    const results = await classifier(inputs);
    const endTime = Date.now();
    
    // Calculate the inference time
    const inference_time = (endTime - startTime) + ' ms';

    // Extract the predictions from the results
    const predictions = results.map(result => result.label);
    //console.log(predictions)

    // Calculate evaluation metrics
    const metrics = calculateMetrics(predictions);

    return { ...metrics, inference_time };
}

async function runBenchmarks(models) {
    for (const model of models) {
        try {
            const performance = await benchmarkModel(model);
            console.log(`\n\nPerformance for ${model}:`, performance);
        } catch(error) {
            console.error(`\n\nError benchmarking ${model}:`, error);
        }
    }
    console.log("\n");
}

// Models to benchmark
const models = [
    'tf_model.h5'
];

// Benchmark the models
runBenchmarks(models);