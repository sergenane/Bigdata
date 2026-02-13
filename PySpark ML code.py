 from pyspark.sql import SparkSession
 from pyspark.ml.feature import VectorAssembler, StringIndexer
 from pyspark.ml.classification import RandomForestClassifier
 from pyspark.ml.evaluation import MulticlassClassificationEvaluator
 from pyspark.ml import Pipeline
 import happybase
 import datetime

 spark = SparkSession.builder \
     .appName("WineQuality_RandomForest_HBase") \
     .enableHiveSupport() \
     .getOrCreate()

 print("Loading wine quality data from Hive...")
 wine_df = spark.sql("""
     SELECT
         fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
         pH, sulphates, alcohol, quality
     FROM wine_quality
 """)

 print(f"Loaded {wine_df.count()} records")

 feature_columns = [
     'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
     'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
     'pH', 'sulphates', 'alcohol'
 ]
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = evaluator_accuracy.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

print("=" * 50)
print("MODEL EVALUATION METRICS")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print("=" * 50)

rf_model = model.stages[-1]
feature_importance = list(zip(feature_columns, rf_model.featureImportances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("\nTOP 5 MOST IMPORTANT FEATURES:")
for i, (feature, importance) in enumerate(feature_importance[:5], 1):
    print(f"{i}. {feature}: {importance:.4f}")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

 indexer = StringIndexer(inputCol="quality", outputCol="label", handleInvalid="keep")
 assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
 rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=10, impurity="gini")
 pipeline = Pipeline(stages=[indexer, assembler, rf])

 train_data, test_data = wine_df.randomSplit([0.7, 0.3], seed=42)
 print(f"Training data: {train_data.count()} records")
 print(f"Test data: {test_data.count()} records")

 print("Training Random Forest model...")
 model = pipeline.fit(train_data)
 predictions = model.transform(test_data)

 evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
 evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

 accuracy = evaluator_accuracy.evaluate(predictions)
 f1_score = evaluator_f1.evaluate(predictions)

 print("=" * 50)
 print("MODEL EVALUATION METRICS")
 print("=" * 50)
 print(f"Accuracy: {accuracy:.4f}")
 print(f"F1 Score: {f1_score:.4f}")
 print("=" * 50)

 rf_model = model.stages[-1]
 feature_importance = list(zip(feature_columns, rf_model.featureImportances))
 feature_importance.sort(key=lambda x: x[1], reverse=True)

 print("\nTOP 5 MOST IMPORTANT FEATURES:")
 for i, (feature, importance) in enumerate(feature_importance[:5], 1):
     print(f"{i}. {feature}: {importance:.4f}")

 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
 row_key = f"wine_metrics_{timestamp}"

 data = [
     (row_key, 'cf:accuracy', str(accuracy)),
     (row_key, 'cf:f1_score', str(f1_score)),
     (row_key, 'cf:num_trees', '50'),
     (row_key, 'cf:max_depth', '10'),
     (row_key, 'cf:test_records', str(test_data.count())),
     (row_key, 'cf:top_feature', feature_importance[0][0]),
     (row_key, 'cf:top_importance', str(feature_importance[0][1]))
 ]

 def write_to_hbase_partition(partition):
     connection = happybase.Connection('master', timeout=60000)
     connection.open()
     table = connection.table('wine_quality_metrics')
     for row in partition:
         row_key, column, value = row
         table.put(row_key.encode('utf-8'), {column.encode('utf-8'): value.encode('utf-8')})
     connection.close()

 print(f"\nWriting metrics to HBase with row key: {row_key}")
 rdd = spark.sparkContext.parallelize(data)
 rdd.foreachPartition(write_to_hbase_partition)
 print("Metrics successfully written to HBase!")

 spark.stop()