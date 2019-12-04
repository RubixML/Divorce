<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use League\Csv\Reader;

echo 'Loading data into memory ...' . PHP_EOL;

$reader = Reader::createFromPath('dataset.csv')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
    'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20',
    'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30',
    'Q31', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40',
    'Q41', 'Q42', 'Q43', 'Q44', 'Q45', 'Q46', 'Q47', 'Q48', 'Q49', 'Q50',
    'Q51', 'Q52', 'Q53', 'Q54',
]);

$labels = $reader->fetchColumn('Class');

$dataset = Labeled::fromIterator($samples, $labels);

$dataset->apply(new NumericStringConverter());

[$training, $testing] = $dataset->stratifiedSplit(0.8);

$estimator = new KNearestNeighbors(3);

echo 'Training ...' . PHP_EOL;

$estimator->train($dataset);

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($testing);

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo "Accuracy is $score" . PHP_EOL;
