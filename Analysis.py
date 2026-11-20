
# Step 1: Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Step 2: Initialize SparkSession
spark = SparkSession.builder.appName(
    "StudyTime_ExamPerformance"
).getOrCreate()

# Step 3: Load the dataset
data = spark.read.csv(
    "C:/Users/hp Win 10/Desktop/LPU DOCS/Cluster Computing/docs/study_time_exam_messy_10_columns.csv",
    header=True,
    inferSchema=False
)

# Display first 5 records
data.show(5, truncate=False)

# Display schema
data.printSchema()

# Step 4: Clean the data

# Convert empty strings and invalid text into NULL
data = data.withColumn(
    "Study_Hours",
    regexp_replace(trim(col("Study_Hours")), " hrs|hours", "")
)

data = data.withColumn(
    "Study_Hours",
    col("Study_Hours").cast("double")
)

data = data.withColumn(
    "Exam_Marks",
    col("Exam_Marks").cast("double")
)

# Remove missing and invalid records
data_cleaned = data.dropna(
    subset=["Study_Hours", "Exam_Marks"]
)

data_cleaned = data_cleaned.filter(
    (col("Study_Hours") >= 0) &
    (col("Study_Hours") <= 24) &
    (col("Exam_Marks") >= 0) &
    (col("Exam_Marks") <= 100)
)

# Remove duplicate records
data_cleaned = data_cleaned.dropDuplicates()

print("Cleaned Dataset:")
data_cleaned.show(5)

print("Original Records:", data.count())
print("Cleaned Records:", data_cleaned.count())

# Step 5: Prepare the data

# Assemble Study_Hours into a single feature vector
assembler = VectorAssembler(
    inputCols=["Study_Hours"],
    outputCol="features"
)

data_assembled = assembler.transform(data_cleaned)

# Select only features and label
data_final = data_assembled.select(
    "features",
    col("Exam_Marks").alias("label")
)

# Display prepared data
data_final.show(5, truncate=False)

# Step 6: Split data into training and testing sets
train_data, test_data = data_final.randomSplit(
    [0.7, 0.3],
    seed=42
)

print("Training data count:", train_data.count())
print("Testing data count:", test_data.count())

# Step 7: Create Linear Regression model
lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

# Step 8: Train the Linear Regression model
model = lr.fit(train_data)

# Step 9: Make predictions on test data
predictions = model.transform(test_data)

# Display predictions
predictions.select(
    "features",
    "label",
    "prediction"
).show(10, truncate=False)

# Step 10: Evaluate the model using RMSE
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)

print("Root Mean Squared Error:", round(rmse, 2))

# Step 11: Calculate Correlation Coefficient
correlation = data_cleaned.stat.corr(
    "Study_Hours",
    "Exam_Marks"
)

print("Correlation Coefficient:", round(correlation, 4))

# Display model coefficients
print("Intercept:", model.intercept)
print("Coefficient:", model.coefficients[0])

# Step 12: Visualize Regression Line

plot_data = predictions.select(
    "features",
    "label",
    "prediction"
).collect()

study_hours = [
    float(row.features[0]) for row in plot_data
]

actual_marks = [
    float(row.label) for row in plot_data
]

predicted_marks = [
    float(row.prediction) for row in plot_data
]

# Sort data for regression line
sorted_data = sorted(
    zip(study_hours, predicted_marks)
)

x_line = [x[0] for x in sorted_data]
y_line = [x[1] for x in sorted_data]

plt.figure(figsize=(10, 6))

plt.scatter(
    study_hours,
    actual_marks,
    label="Actual Exam Marks"
)

plt.plot(
    x_line,
    y_line,
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Study Time vs Exam Performance")

plt.legend()
plt.grid(True)
plt.show()

# Step 13: Stop SparkSession
spark.stop()
