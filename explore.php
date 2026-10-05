<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Extractors\CSV;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new NDJSON('dataset.ndjson'));

$dataset->apply(new FloatTypeConverter());

$embedder = new PrincipalComponentAnalysis(2);

$dataset->apply($embedder)->exportTo(new CSV('embedding.csv'), true);

$logger->info('Embedding saved to embedding.csv');
