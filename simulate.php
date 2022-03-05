<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new NDJSON('dataset.ndjson'));

$blobs = [];

foreach ($dataset->describeByLabel() as $class => $dist) {
    $means = $stddevs = [];

    foreach ($dist as $stats) {
        if ($stats['type'] === 'continuous') {
            $means[] = $stats['mean'];
            $stddevs[] = $stats['standard deviation'];
        }
    }

    $blobs[$class] = new Blob($means, $stddevs);
}

$generator = new Agglomerate($blobs);

[$training, $testing] = $generator->generate(10000)->stratifiedSplit(0.8);

$estimator = new KDNeighbors(3);

$logger->info('Training');

$estimator->train($training);

$logger->info('Making predictions');

$predictions = $estimator->predict($testing);

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

$logger->info("Accuracy is $score");
